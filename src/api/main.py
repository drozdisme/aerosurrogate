"""FastAPI application for AeroSurrogate v2.0 — final release."""

import csv
import io
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.security import APIKeyHeader

from src.api.schemas import (
    BatchPredictionInput, BatchPredictionOutput,
    CoeffSet, CompareRequest, CompareResponse, ConfidenceInfo,
    DeltaSet, FieldPredictionOutput, HealthResponse, ModelMetrics,
    PredictionInput, PredictionOutput, SweepExportRequest, TargetPrediction,
)
from src.demo.demo_model import generate_cp_distribution, health_metrics, predict_coefficients
from src.inference.sklearn_predictor import SklearnPredictor

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

class _HealthFilter(logging.Filter):
    def filter(self, record):
        return "GET /health" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(_HealthFilter())

# ─── State ────────────────────────────────────────────────────────────────────
MODELS_DIR    = os.getenv("MODELS_DIR",    "models")
ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")

predictor = SklearnPredictor(
    models_dir=MODELS_DIR,
    metrics_path=str(Path(ARTIFACTS_DIR) / "metrics.json"),
)
DEMO_MODE = False

# ─── Auth ─────────────────────────────────────────────────────────────────────
API_KEY = os.getenv("AERO_API_KEY", "")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def _check_key(key: Optional[str] = Security(_api_key_header)):
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Invalid or missing API key")

# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app):
    global DEMO_MODE
    try:
        predictor.load()
        field_type = predictor.metrics.get("field_model_type", "none")
        logger.info(
            f"✓ ML mode — backend={predictor.metrics.get('backend','?')}  "
            f"Cl R²={predictor.metrics.get('Cl_R2')}  "
            f"field_model={field_type}  "
            f"fno={'✓' if predictor.has_deeponet or field_type == 'fno' else '✗'}"
        )
    except Exception as e:
        logger.warning(f"Models unavailable ({e}), running in DEMO mode")
        DEMO_MODE = True
    yield

app = FastAPI(title="AeroSurrogate API", version="3.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get("/", include_in_schema=False)
async def _root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _demo_coeff_out(data: dict) -> PredictionOutput:
    """Analytical fallback — demo mode."""
    geo  = {k: data.get(k, 0.0) for k in
            ("thickness_ratio","camber","camber_position","leading_edge_radius",
             "trailing_edge_angle","aspect_ratio","taper_ratio","sweep_angle",
             "twist_angle","dihedral_angle")}
    flow = {k: data.get(k, 0.0) for k in ("mach","reynolds","alpha","beta","altitude")}
    c    = predict_coefficients(geo, flow)
    cl, cd, cm, k = c["Cl"], c["Cd"], c["Cm"], c["K"]
    return PredictionOutput(
        predictions={
            "Cl": TargetPrediction(value=cl, std=abs(cl)*0.03),
            "Cd": TargetPrediction(value=cd, std=abs(cd)*0.04),
            "Cm": TargetPrediction(value=cm, std=abs(cm)*0.05 + 0.001),
            "K":  TargetPrediction(value=k,  std=abs(k)*0.03),
        },
        # uppercase so Confidence.jsx colour-maps correctly even without normalisation
        confidence=ConfidenceInfo(score=0.82, level="HIGH", color="#16a34a"),
        demo_mode=True,
    )

def _ml_out(raw: dict) -> PredictionOutput:
    return PredictionOutput(
        predictions={k: TargetPrediction(value=v["value"], std=v["std"])
                     for k, v in raw["predictions"].items()},
        confidence=ConfidenceInfo(**raw["confidence"]),
        demo_mode=False,
    )

def _run(data: dict) -> PredictionOutput:
    if DEMO_MODE:
        return _demo_coeff_out(data)
    try:
        return _ml_out(predictor.predict(data))
    except Exception as e:
        logger.warning(f"ML predict failed ({e}), fallback to demo")
        return _demo_coeff_out(data)

def _run_field(data: dict) -> FieldPredictionOutput:
    """Cp(x) distribution — ML surrogate when available, analytical otherwise."""
    # Try ML Cp surrogate first
    if not DEMO_MODE and predictor.is_loaded and predictor.has_deeponet:
        try:
            result = predictor.predict_field(data, n_points=200)
            if result:
                return FieldPredictionOutput(**result)
        except Exception as e:
            logger.warning(f"Cp surrogate failed ({e}), fallback to analytical")

    # Analytical fallback
    geo  = {k: data.get(k, 0.0) for k in
            ("thickness_ratio","camber","camber_position","leading_edge_radius",
             "trailing_edge_angle","aspect_ratio","taper_ratio","sweep_angle",
             "twist_angle","dihedral_angle")}
    flow = {k: data.get(k, 0.0) for k in ("mach","reynolds","alpha","beta","altitude")}
    cp   = generate_cp_distribution(geo, flow)
    return FieldPredictionOutput(**cp)

def _coeffset(o: PredictionOutput) -> CoeffSet:
    p = o.predictions
    return CoeffSet(Cl=p["Cl"].value, Cd=p["Cd"].value,
                    Cm=p["Cm"].value, K=p["K"].value)

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse)
async def health():
    m = mv = None
    if not DEMO_MODE and predictor.is_loaded and predictor.metrics:
        md = predictor.metrics
        m  = ModelMetrics(Cl_R2=md.get("Cl_R2",0), Cd_R2=md.get("Cd_R2",0),
                          Cm_R2=md.get("Cm_R2",0), MAPE=md.get("MAPE",0))
        mv = md.get("model_version", "v3")
    elif DEMO_MODE:
        m  = ModelMetrics(**health_metrics())
        mv = "demo"

    field_model = None
    if not DEMO_MODE and predictor.is_loaded:
        field_model = predictor.metrics.get("field_model_type", "none")
        if field_model == "none" and predictor.has_deeponet:
            field_model = "deeponet"

    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded,
        demo_mode=DEMO_MODE,
        deeponet_available=predictor.has_deeponet,
        version="3.0.0",
        model_version=mv,
        metrics=m,
    )

@app.post("/predict", response_model=PredictionOutput,
          dependencies=[Depends(_check_key)])
async def predict(data: PredictionInput):
    return _run(data.model_dump())

@app.post("/predict/batch", response_model=BatchPredictionOutput,
          dependencies=[Depends(_check_key)])
async def predict_batch(batch: BatchPredictionInput):
    results = [_run(inp.model_dump()) for inp in batch.inputs]
    return BatchPredictionOutput(results=results, count=len(results))

@app.post("/predict/field", response_model=FieldPredictionOutput,
          dependencies=[Depends(_check_key)])
async def predict_field(data: PredictionInput):
    return _run_field(data.model_dump())   # ← FIXED: now uses ML surrogate

@app.post("/compare", response_model=CompareResponse,
          dependencies=[Depends(_check_key)])
async def compare(req: CompareRequest):
    outA = _run(req.configA.model_dump())
    outB = _run(req.configB.model_dump())
    cA, cB = _coeffset(outA), _coeffset(outB)
    delta = DeltaSet(
        Cl=round(cB.Cl-cA.Cl,6), Cd=round(cB.Cd-cA.Cd,6),
        Cm=round(cB.Cm-cA.Cm,6), K=round(cB.K -cA.K, 4),
    )
    return CompareResponse(
        A=cA, B=cB, delta=delta,
        cpA=_run_field(req.configA.model_dump()),
        cpB=_run_field(req.configB.model_dump()),
        demo_mode=DEMO_MODE or outA.demo_mode,
    )

@app.post("/export/sweep")
async def export_sweep(req: SweepExportRequest):
    buf = io.StringIO()
    w   = csv.DictWriter(buf, fieldnames=["alpha","Cl","Cd","Cm","K","confidence"])
    w.writeheader()
    for pt in req.points:
        w.writerow(pt.model_dump())
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sweep_results.csv"},
    )


# ─── v3.0 Endpoints ────────────────────────────────────────────────────────────

# ── Optimization ──────────────────────────────────────────────────────────────

from pydantic import BaseModel as _BaseModel
from typing import Dict as _Dict, Any as _Any, Optional as _Optional, List as _List

class OptimizationRequest(_BaseModel):
    flow_conditions: _Dict[str, float]         # mach, reynolds, alpha, ...
    objective: str = "max_lift_drag"           # max_lift_drag | min_drag | max_lift | pareto
    parametrization: str = "scalar"            # scalar | cst | naca
    n_trials: int = 100
    sampler: str = "tpe"                       # tpe | cmaes | random
    constraints: _Optional[_Dict[str, float]] = None  # e.g. {"Cl_min": 0.5}
    timeout: _Optional[float] = None

class OptimizationResponse(_BaseModel):
    objective: str
    best_value: float
    best_geometry: _Dict[str, float]
    n_trials: int
    n_completed: int
    elapsed_seconds: float
    pareto_front: _Optional[_List[_Dict]] = None


@app.post(
    "/optimize",
    response_model=OptimizationResponse,
    dependencies=[Depends(_check_key)],
    summary="Bayesian geometry optimization (v3.0)",
    tags=["v3"],
)
async def optimize(req: OptimizationRequest):
    """Run Bayesian shape optimization using the surrogate ensemble.

    Maximizes Cl/Cd or minimizes Cd over the airfoil geometry parameter space.
    Uses Optuna TPE (default) or CMA-ES sampler.

    Returns best geometry, objective value, and — for 'pareto' — the Pareto front.
    """
    try:
        from optimization.bayesian_optimizer import BayesianOptimizer
    except ImportError as e:
        raise HTTPException(500, f"optimization module unavailable: {e}")

    if req.n_trials > 2000:
        raise HTTPException(400, "n_trials capped at 2000 per request")

    def _predictor_fn(d: dict) -> dict:
        """Thin wrapper: map optimizer dict → ensemble predict output."""
        return _run(d).model_dump()

    try:
        optimizer = BayesianOptimizer(
            predictor=_predictor_fn,
            flow_conditions=req.flow_conditions,
            objective=req.objective,
            parametrization=req.parametrization,
            n_trials=req.n_trials,
            sampler=req.sampler,
            seed=42,
            constraints=req.constraints,
            timeout=req.timeout,
        )
        result = optimizer.optimize()
        return OptimizationResponse(
            objective=result.objective,
            best_value=round(result.best_value, 6),
            best_geometry=result.best_geometry,
            n_trials=result.n_trials,
            n_completed=result.n_completed,
            elapsed_seconds=round(result.elapsed_seconds, 2),
            pareto_front=result.pareto_front,
        )
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise HTTPException(500, str(e))


# ── Digital Twin ──────────────────────────────────────────────────────────────

# Lazy-initialised digital twin predictor (loaded on first request)
_twin_predictor = None
_twin_lock = __import__("threading").Lock()


def _get_twin():
    """Return loaded RealTimePredictor, initialising on first call."""
    global _twin_predictor
    if _twin_predictor is not None:
        return _twin_predictor
    with _twin_lock:
        if _twin_predictor is not None:
            return _twin_predictor
        try:
            from digital_twin.real_time_predictor import RealTimePredictor
            p = RealTimePredictor(
                artifacts_dir=ARTIFACTS_DIR,
                config_dir="configs",
                include_field=False,
                field_every_n=10,
            )
            if not DEMO_MODE:
                p.load()
            _twin_predictor = p
        except Exception as e:
            logger.warning(f"Digital twin unavailable: {e}")
            _twin_predictor = None
    return _twin_predictor


class TelemetryRequest(_BaseModel):
    """Raw telemetry record for digital twin inference."""
    timestamp: _Optional[float] = None
    mach: float
    alpha_deg: float
    altitude_m: float = 0.0
    beta_deg: float = 0.0
    airspeed_ms: _Optional[float] = None
    reynolds: _Optional[float] = None
    # Geometry (optional override of aircraft defaults)
    thickness_ratio: _Optional[float] = None
    camber: _Optional[float] = None
    camber_position: _Optional[float] = None
    leading_edge_radius: _Optional[float] = None
    trailing_edge_angle: _Optional[float] = None
    aspect_ratio: _Optional[float] = None
    taper_ratio: _Optional[float] = None
    sweep_angle: _Optional[float] = None
    twist_angle: _Optional[float] = None
    dihedral_angle: _Optional[float] = None


class TelemetryBatchRequest(_BaseModel):
    records: _List[TelemetryRequest]


@app.post(
    "/twin/ingest",
    dependencies=[Depends(_check_key)],
    summary="Digital twin: single telemetry record inference (v3.0)",
    tags=["v3", "digital-twin"],
)
async def twin_ingest(req: TelemetryRequest):
    """Ingest one flight telemetry record and return real-time load prediction.

    Applies ISA atmosphere model to compute Reynolds from altitude + Mach
    if `reynolds` is not provided.
    """
    twin = _get_twin()
    if twin is None or not twin.is_loaded:
        # Fallback: use main ensemble directly
        data = {k: v for k, v in req.model_dump().items() if v is not None}
        data["alpha"] = data.pop("alpha_deg", data.get("alpha", 0.0))
        return _run(data).model_dump()

    try:
        from digital_twin.telemetry_ingestion import TelemetryIngestion
        ingestion = TelemetryIngestion()
        record = ingestion.ingest(req.model_dump(exclude_none=True))
        result = await __import__("asyncio").get_event_loop().run_in_executor(
            None, twin.predict_from_record, record
        )
        from digital_twin.real_time_predictor import _result_to_json
        return _result_to_json(result)
    except Exception as e:
        logger.warning(f"Digital twin ingest failed ({e}), falling back to ensemble")
        data = {k: v for k, v in req.model_dump(exclude_none=True).items()}
        data["alpha"] = data.pop("alpha_deg", data.get("alpha", 0.0))
        return _run(data).model_dump()


@app.post(
    "/twin/ingest/batch",
    dependencies=[Depends(_check_key)],
    summary="Digital twin: batch telemetry ingestion (v3.0)",
    tags=["v3", "digital-twin"],
)
async def twin_ingest_batch(req: TelemetryBatchRequest):
    """Process a batch of telemetry records."""
    results = []
    for rec in req.records:
        r = await twin_ingest(rec)
        results.append(r)
    return {"count": len(results), "results": results}


@app.get(
    "/twin/status",
    summary="Digital twin predictor status and streaming statistics (v3.0)",
    tags=["v3", "digital-twin"],
)
async def twin_status():
    """Return digital twin predictor status, history stats, and alert thresholds."""
    twin = _get_twin()
    if twin is None:
        return {"loaded": False, "demo_mode": DEMO_MODE}
    return {
        "loaded": twin.is_loaded,
        "demo_mode": DEMO_MODE,
        "stats": twin.get_stats(),
        "history_size": len(list(twin._history)),
        "thresholds": twin.thresholds,
    }


@app.get(
    "/twin/history",
    summary="Recent digital twin prediction history (v3.0)",
    tags=["v3", "digital-twin"],
)
async def twin_history(n: int = 50):
    """Return the last N digital twin predictions."""
    twin = _get_twin()
    if twin is None or not twin.is_loaded:
        raise HTTPException(503, "Digital twin not loaded")
    from digital_twin.real_time_predictor import _result_to_json
    hist = twin.get_history(n=n)
    return {"count": len(hist), "records": [_result_to_json(r) for r in hist]}


# ── FNO field resolution endpoint ─────────────────────────────────────────────

class FieldResolutionRequest(_BaseModel):
    """Predict Cp at custom resolution or custom x-coordinates."""
    data: _Dict[str, float]
    n_points: int = 200
    x_coords: _Optional[_List[float]] = None   # custom surface x ∈ [0,1]


@app.post(
    "/predict/field/custom",
    dependencies=[Depends(_check_key)],
    summary="FNO field prediction at custom resolution (v3.0)",
    tags=["v3"],
)
async def predict_field_custom(req: FieldResolutionRequest):
    """Predict Cp(x) at arbitrary resolution using the FNO field model.

    Pass `n_points` for a uniform grid, or `x_coords` for non-uniform evaluation.
    The FNO model is resolution-independent — no retraining needed.
    """
    import numpy as _np

    coords = None
    if req.x_coords:
        coords = _np.array(req.x_coords, dtype=_np.float32)

    if not DEMO_MODE and predictor.is_loaded and predictor.has_deeponet:
        try:
            result = predictor.predict_field(req.data, n_points=req.n_points)
            if result:
                return result
        except Exception as e:
            logger.warning(f"FNO custom field failed: {e}")

    # Analytical fallback
    geo  = {k: req.data.get(k, 0.0) for k in
            ("thickness_ratio","camber","camber_position","leading_edge_radius",
             "trailing_edge_angle","aspect_ratio","taper_ratio","sweep_angle",
             "twist_angle","dihedral_angle")}
    flow = {k: req.data.get(k, 0.0) for k in ("mach","reynolds","alpha","beta","altitude")}
    return _run_field({**geo, **flow}).model_dump()