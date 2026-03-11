"""Real-Time Predictor for AeroSurrogate Digital Twin v3.0.

Bridges the TelemetryIngestion pipeline with the EnsemblePredictor to
provide live aerodynamic load estimation from flight telemetry.

Key features:
  - Sub-10ms single-sample inference latency
  - Async streaming prediction loop (asyncio)
  - Thread-safe prediction cache (deque ring buffer)
  - REST endpoints via FastAPI (mountable as sub-application)
  - SSE (Server-Sent Events) streaming for dashboard push
  - Load alert thresholds with callback hooks
  - Graceful degradation: caches last valid prediction on model error

Usage (standalone)::

    predictor = RealTimePredictor(artifacts_dir="artifacts")
    predictor.load()

    # Single inference
    record = ingestion.ingest(raw_telemetry)
    result = predictor.predict_from_record(record)

    # Mount SSE streaming app
    from fastapi import FastAPI
    app = FastAPI()
    app.mount("/twin", predictor.create_app())

Usage (streaming loop)::

    async for result in predictor.stream(telemetry_source):
        print(result.predictions, result.confidence)
"""

import asyncio
import json
import logging
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import AsyncIterator, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class LoadAlert:
    """Triggered when a prediction exceeds a defined threshold."""
    timestamp: float
    parameter: str           # e.g. "Cl", "Cd"
    predicted_value: float
    threshold: float
    direction: str           # "above" or "below"
    confidence: float


@dataclass
class RealTimePrediction:
    """Output of one real-time inference cycle."""
    timestamp: float
    predictions: Dict[str, Dict]          # {"Cl": {"value": ..., "std": ...}, ...}
    confidence: Dict                      # {"score": ..., "level": ..., "color": ...}
    cp_field: Optional[Dict] = None          # Cp(x) field, when available
    alerts: List[LoadAlert] = field(default_factory=list)
    latency_ms: float = 0.0
    source: str = "telemetry"             # "telemetry" | "replay" | "synthetic"


@dataclass
class StreamStats:
    """Running statistics for the streaming loop."""
    n_records: int = 0
    n_errors: int = 0
    n_alerts: int = 0
    mean_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    start_time: float = field(default_factory=time.time)

    def update(self, latency_ms: float, error: bool = False) -> None:
        self.n_records += 1
        if error:
            self.n_errors += 1
        alpha = 0.05  # EMA smoothing
        self.mean_latency_ms = (
            alpha * latency_ms + (1 - alpha) * self.mean_latency_ms
            if self.n_records > 1
            else latency_ms
        )
        self.max_latency_ms = max(self.max_latency_ms, latency_ms)

    @property
    def uptime_s(self) -> float:
        return time.time() - self.start_time

    @property
    def throughput_hz(self) -> float:
        return self.n_records / max(self.uptime_s, 1e-3)


# ── Threshold / alert configuration ───────────────────────────────────────────

DEFAULT_THRESHOLDS = {
    "Cl":  {"above": 2.0,  "below": -0.5},
    "Cd":  {"above": 0.15, "below": None},
    "Cm":  {"above": 0.5,  "below": -0.5},
    "K":   {"above": None, "below": 2.0},
}


# ── Main predictor class ───────────────────────────────────────────────────────

class RealTimePredictor:
    """Production real-time aerodynamic load predictor.

    Wraps EnsemblePredictor with:
      - async streaming loop
      - SSE broadcasting
      - alert detection
      - rolling prediction history

    Parameters
    ----------
    artifacts_dir : str
        Path to trained model artifacts.
    config_dir : str
        Path to YAML configs.
    history_size : int
        Number of recent predictions to keep in ring buffer.
    include_field : bool
        If True, compute Cp(x) field on each prediction (higher latency).
    field_every_n : int
        Compute field prediction every N records (latency tradeoff).
    thresholds : dict, optional
        Override DEFAULT_THRESHOLDS for alerts.
    n_workers : int, optional
        Parallel inference workers.
    """

    def __init__(
        self,
        artifacts_dir: str = "artifacts",
        config_dir: str = "configs",
        history_size: int = 1000,
        include_field: bool = False,
        field_every_n: int = 10,
        thresholds: Optional[Dict] = None,
        n_workers: Optional[int] = None,
    ):
        self.artifacts_dir = Path(artifacts_dir)
        self.config_dir = Path(config_dir)
        self.history_size = history_size
        self.include_field = include_field
        self.field_every_n = field_every_n
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.n_workers = n_workers

        self._predictor = None        # EnsemblePredictor
        self.is_loaded = False

        self._history: deque = deque(maxlen=history_size)
        self._last_result: Optional[RealTimePrediction] = None
        self._stats = StreamStats()
        self._record_count = 0

        # SSE subscribers: list of asyncio.Queue
        self._sse_queues: List[asyncio.Queue] = []
        self._sse_lock = threading.Lock()

        # Alert callbacks
        self._alert_callbacks: List[Callable] = []

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load EnsemblePredictor from disk."""
        t0 = time.time()
        try:
            from inference.ensemble_predictor import EnsemblePredictor
            self._predictor = EnsemblePredictor(
                artifacts_dir=str(self.artifacts_dir),
                config_dir=str(self.config_dir),
                n_workers=self.n_workers,
                use_fno=True,
            )
            self._predictor.load()
            self.is_loaded = True
            elapsed = (time.time() - t0) * 1000
            logger.info(
                f"RealTimePredictor loaded in {elapsed:.0f}ms | "
                f"field_model={self._predictor.field_model_type}"
            )
        except Exception as e:
            logger.error(f"RealTimePredictor load failed: {e}")
            raise

    # ── Core inference ─────────────────────────────────────────────────────────

    def predict_from_record(
        self,
        record,          # TelemetryRecord (duck-typed to avoid circular import)
        compute_field: bool = False,
    ) -> RealTimePrediction:
        """Run one inference cycle from a TelemetryRecord.

        Args:
            record: TelemetryRecord from TelemetryIngestion.ingest().
            compute_field: Override per-call field computation flag.

        Returns:
            RealTimePrediction with predictions, confidence, optional field.
        """
        t0 = time.time()

        # Convert record to feature dict
        input_dict = self._record_to_dict(record)

        raw = self._predictor.predict(input_dict)

        result = RealTimePrediction(
            timestamp=record.timestamp if hasattr(record, "timestamp") else time.time(),
            predictions=raw["predictions"],
            confidence=raw["confidence"],
            latency_ms=0.0,
            source="telemetry",
        )

        # Optional Cp field
        if compute_field or (self.include_field and self._record_count % self.field_every_n == 0):
            try:
                field_data = self._predictor.predict_field(input_dict, n_points=100)
                result.cp_field = field_data
            except Exception as e:
                logger.debug(f"Field prediction skipped: {e}")

        # Alert detection
        result.alerts = self._check_alerts(result.predictions, record.timestamp)

        result.latency_ms = (time.time() - t0) * 1000
        self._record_count += 1
        self._stats.update(result.latency_ms)
        self._history.append(result)
        self._last_result = result

        # Dispatch alerts
        if result.alerts:
            self._stats.n_alerts += len(result.alerts)
            self._fire_alert_callbacks(result.alerts)

        # Broadcast to SSE subscribers
        self._broadcast_sse(result)

        return result

    def predict_from_dict(self, input_dict: Dict) -> RealTimePrediction:
        """Direct dict inference (bypasses TelemetryIngestion)."""
        if not self.is_loaded:
            raise RuntimeError("RealTimePredictor not loaded")

        t0 = time.time()
        raw = self._predictor.predict(input_dict)

        result = RealTimePrediction(
            timestamp=time.time(),
            predictions=raw["predictions"],
            confidence=raw["confidence"],
            latency_ms=0.0,
            source="direct",
        )

        if self.include_field:
            try:
                result.cp_field = self._predictor.predict_field(input_dict, n_points=100)
            except Exception:
                pass

        result.alerts = self._check_alerts(result.predictions, result.timestamp)
        result.latency_ms = (time.time() - t0) * 1000
        self._history.append(result)
        self._last_result = result
        self._stats.update(result.latency_ms)
        self._broadcast_sse(result)
        return result

    # ── Async streaming ────────────────────────────────────────────────────────

    async def stream(
        self,
        source: AsyncIterator,         # yields TelemetryRecord objects
        max_records: Optional[int] = None,
    ) -> AsyncIterator[RealTimePrediction]:
        """Async streaming prediction loop.

        Args:
            source: Async iterator of TelemetryRecord (from TelemetryIngestion.stream).
            max_records: Optional stop condition.

        Yields:
            RealTimePrediction for every incoming record.
        """
        if not self.is_loaded:
            raise RuntimeError("Not loaded — call load() first")

        count = 0
        async for record in source:
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self.predict_from_record, record
                )
                yield result
            except Exception as e:
                self._stats.update(0.0, error=True)
                logger.warning(f"Stream prediction error (record {count}): {e}")
                if self._last_result is not None:
                    yield self._last_result  # return last good prediction
            count += 1
            if max_records is not None and count >= max_records:
                break

    # ── History and statistics ─────────────────────────────────────────────────

    def get_history(
        self,
        n: Optional[int] = None,
        as_arrays: bool = False,
    ) -> List[RealTimePrediction]:
        """Return recent prediction history.

        Args:
            n: Number of most recent records. None = all.
            as_arrays: If True, convert to dict of numpy arrays (for plotting).
        """
        history = list(self._history)
        if n is not None:
            history = history[-n:]
        if as_arrays:
            return self._history_to_arrays(history)
        return history

    def get_stats(self) -> Dict:
        """Return streaming statistics dict."""
        return {
            "n_records": self._stats.n_records,
            "n_errors": self._stats.n_errors,
            "n_alerts": self._stats.n_alerts,
            "mean_latency_ms": round(self._stats.mean_latency_ms, 2),
            "max_latency_ms": round(self._stats.max_latency_ms, 2),
            "throughput_hz": round(self._stats.throughput_hz, 2),
            "uptime_s": round(self._stats.uptime_s, 1),
            "field_model": (
                self._predictor.field_model_type
                if self._predictor else None
            ),
        }

    # ── Alerts ─────────────────────────────────────────────────────────────────

    def register_alert_callback(self, fn: Callable[[List[LoadAlert]], None]) -> None:
        """Register a function to be called when alerts are triggered."""
        self._alert_callbacks.append(fn)

    def set_threshold(self, parameter: str, above: Optional[float] = None,
                       below: Optional[float] = None) -> None:
        """Update alert threshold for a parameter."""
        if parameter not in self.thresholds:
            self.thresholds[parameter] = {}
        if above is not None:
            self.thresholds[parameter]["above"] = above
        if below is not None:
            self.thresholds[parameter]["below"] = below

    # ── SSE broadcasting ───────────────────────────────────────────────────────

    def subscribe_sse(self) -> asyncio.Queue:
        """Create and register a new SSE subscriber queue."""
        q: asyncio.Queue = asyncio.Queue(maxsize=50)
        with self._sse_lock:
            self._sse_queues.append(q)
        return q

    def unsubscribe_sse(self, q: asyncio.Queue) -> None:
        with self._sse_lock:
            if q in self._sse_queues:
                self._sse_queues.remove(q)

    # ── FastAPI sub-application ────────────────────────────────────────────────

    def create_app(self):
        """Create a mountable FastAPI sub-application for digital twin endpoints.

        Exposes:
          GET  /status        — predictor health and stats
          POST /predict       — single inference from raw dict
          GET  /history       — recent prediction history
          GET  /last          — most recent prediction
          GET  /stream        — SSE stream of live predictions
          POST /threshold     — update alert threshold
        """
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.responses import StreamingResponse, JSONResponse
            from pydantic import BaseModel
        except ImportError:
            raise ImportError("fastapi required: pip install fastapi")

        twin_app = FastAPI(title="AeroSurrogate Digital Twin", version="3.0.0")

        class InferenceRequest(BaseModel):
            data: Dict
            compute_field: bool = False

        class ThresholdRequest(BaseModel):
            parameter: str
            above: Optional[float] = None
            below: Optional[float] = None

        @twin_app.get("/status")
        async def status():
            return {
                "loaded": self.is_loaded,
                "stats": self.get_stats(),
                "thresholds": self.thresholds,
                "history_size": len(self._history),
                "sse_subscribers": len(self._sse_queues),
            }

        @twin_app.post("/predict")
        async def predict(req: InferenceRequest):
            if not self.is_loaded:
                raise HTTPException(503, "Predictor not loaded")
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.predict_from_dict(req.data)
                )
                return _result_to_json(result)
            except Exception as e:
                raise HTTPException(500, str(e))

        @twin_app.get("/last")
        async def last_prediction():
            if self._last_result is None:
                raise HTTPException(404, "No predictions yet")
            return _result_to_json(self._last_result)

        @twin_app.get("/history")
        async def history(n: int = 100):
            hist = self.get_history(n=n)
            return {
                "count": len(hist),
                "records": [_result_to_json(r) for r in hist],
            }

        @twin_app.post("/threshold")
        async def set_threshold(req: ThresholdRequest):
            self.set_threshold(req.parameter, req.above, req.below)
            return {"updated": req.parameter, "thresholds": self.thresholds}

        @twin_app.get("/stream")
        async def sse_stream():
            """Server-Sent Events stream of live predictions."""
            q = self.subscribe_sse()

            async def event_generator():
                try:
                    while True:
                        try:
                            result = await asyncio.wait_for(q.get(), timeout=30.0)
                            payload = json.dumps(_result_to_json(result))
                            yield f"data: {payload}\n\n"
                        except asyncio.TimeoutError:
                            yield "data: {\"heartbeat\": true}\n\n"
                except asyncio.CancelledError:
                    pass
                finally:
                    self.unsubscribe_sse(q)

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        return twin_app

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _record_to_dict(record) -> Dict:
        """Convert TelemetryRecord (dataclass) → feature dict."""
        fields_order = [
            "mach", "reynolds", "alpha", "beta", "altitude",
            "thickness_ratio", "camber", "camber_position",
            "leading_edge_radius", "trailing_edge_angle",
            "aspect_ratio", "taper_ratio", "sweep_angle",
            "twist_angle", "dihedral_angle",
        ]
        out = {}
        for f in fields_order:
            val = getattr(record, f, None)
            if val is not None:
                out[f] = float(val)
        return out

    def _check_alerts(
        self, predictions: Dict, timestamp: float
    ) -> List[LoadAlert]:
        alerts = []
        for param, limits in self.thresholds.items():
            if param not in predictions:
                continue
            val = predictions[param].get("value", 0.0)
            conf = 0.0
            if "confidence" in predictions:
                conf = predictions["confidence"].get("score", 0.0)

            if limits.get("above") is not None and val > limits["above"]:
                alerts.append(LoadAlert(
                    timestamp=timestamp,
                    parameter=param,
                    predicted_value=val,
                    threshold=limits["above"],
                    direction="above",
                    confidence=conf,
                ))
            if limits.get("below") is not None and val < limits["below"]:
                alerts.append(LoadAlert(
                    timestamp=timestamp,
                    parameter=param,
                    predicted_value=val,
                    threshold=limits["below"],
                    direction="below",
                    confidence=conf,
                ))
        return alerts

    def _fire_alert_callbacks(self, alerts: List[LoadAlert]) -> None:
        for cb in self._alert_callbacks:
            try:
                cb(alerts)
            except Exception as e:
                logger.warning(f"Alert callback error: {e}")

    def _broadcast_sse(self, result: RealTimePrediction) -> None:
        """Non-blocking SSE broadcast to all subscribers."""
        with self._sse_lock:
            stale = []
            for q in self._sse_queues:
                try:
                    q.put_nowait(result)
                except asyncio.QueueFull:
                    stale.append(q)  # slow consumer, drop
            for q in stale:
                self._sse_queues.remove(q)

    @staticmethod
    def _history_to_arrays(history: List[RealTimePrediction]) -> Dict:
        """Convert history list to dict of numpy arrays for easy plotting."""
        if not history:
            return {}
        targets = list(history[0].predictions.keys())
        out = {
            "timestamp": np.array([r.timestamp for r in history]),
            "latency_ms": np.array([r.latency_ms for r in history]),
            "confidence": np.array([r.confidence.get("score", 0) for r in history]),
        }
        for t in targets:
            out[f"{t}_value"] = np.array([
                r.predictions[t]["value"] for r in history
            ])
            out[f"{t}_std"] = np.array([
                r.predictions[t].get("std", 0) for r in history
            ])
        return out


# ── JSON serialization helper ─────────────────────────────────────────────────

def _result_to_json(result: RealTimePrediction) -> Dict:
    """Safely serialize RealTimePrediction to a JSON-compatible dict."""
    d = {
        "timestamp": result.timestamp,
        "predictions": result.predictions,
        "confidence": result.confidence,
        "latency_ms": round(result.latency_ms, 2),
        "source": result.source,
        "alerts": [
            {
                "parameter": a.parameter,
                "value": a.predicted_value,
                "threshold": a.threshold,
                "direction": a.direction,
            }
            for a in result.alerts
        ],
    }
    if result.cp_field is not None:
        d["field"] = result.cp_field
    return d


# ── Standalone demo ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    predictor = RealTimePredictor(
        artifacts_dir="artifacts",
        config_dir="configs",
        include_field=False,
    )

    try:
        predictor.load()
        logger.info("Predictor loaded in ML mode")
    except Exception as e:
        logger.warning(f"Could not load predictor: {e}. Running status-only mode.")

    app = FastAPI(title="AeroSurrogate Digital Twin Standalone", version="3.0.0")
    app.mount("/twin", predictor.create_app())

    @app.get("/health")
    async def health():
        return {"status": "ok", "twin_loaded": predictor.is_loaded}

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")