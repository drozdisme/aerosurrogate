"""
AeroSurrogate v3.0 — Digital Twin standalone runner.
Запускается через entrypoint_twin.sh из /app, поэтому sys.path[0] = /app
и все пакеты проекта (digital_twin, inference, src, ...) находятся.
"""
import logging
import os
import uvicorn
from fastapi import FastAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "artifacts")
CONFIG_DIR    = os.getenv("CONFIG_DIR",    "configs")

app = FastAPI(title="AeroSurrogate Digital Twin", version="3.0.0")
predictor = None

try:
    from digital_twin.real_time_predictor import RealTimePredictor
    predictor = RealTimePredictor(
        artifacts_dir=ARTIFACTS_DIR,
        config_dir=CONFIG_DIR,
    )
    predictor.load()
    app.mount("/twin", predictor.create_app())
    logging.info("Digital twin loaded OK ✓")
except Exception as e:
    logging.warning(f"Digital twin load failed: {e}")
    logging.warning("Running in degraded mode — /health endpoint only")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded": predictor.is_loaded if predictor else False,
    }


@app.get("/")
def root():
    return {
        "service": "AeroSurrogate Digital Twin",
        "version": "3.0.0",
        "loaded": predictor.is_loaded if predictor else False,
        "docs": "/docs",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
