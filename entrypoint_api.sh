#!/bin/bash
# Used ONLY for local docker-compose (where model_weights volume may be empty).
# On Railway: CMD in Dockerfile is used directly (models baked in at build).
set -e

MODELS_DIR="${MODELS_DIR:-models}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts}"
mkdir -p "$MODELS_DIR" "$ARTIFACTS_DIR" dataset

NEED_TRAIN=false
for f in "${MODELS_DIR}/cl_model.pkl" \
          "${MODELS_DIR}/cd_model.pkl" \
          "${MODELS_DIR}/cm_model.pkl" \
          "${MODELS_DIR}/cp_surrogate.pkl"; do
  [ ! -f "$f" ] && NEED_TRAIN=true && break
done

if [ "$NEED_TRAIN" = "true" ]; then
  echo "→ First run: training models (this only happens once)..."
  python train_v3.py
  echo "✓ Training complete"
else
  echo "→ Models found, skipping training"
fi

exec uvicorn src.api.main:app \
     --host 0.0.0.0 \
     --port "${PORT:-8000}" \
     --workers 1 \
     --log-level info
