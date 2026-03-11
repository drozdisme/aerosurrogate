#!/bin/bash
# AeroSurrogate v3.0 — Digital Twin service
# Запускаем twin_runner.py из /app чтобы sys.path[0]=/app
# и все пакеты проекта (digital_twin, inference, src) были доступны.
set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   AeroSurrogate v3.0 — Digital Twin on :8001"
echo "   ARTIFACTS_DIR: ${ARTIFACTS_DIR:-artifacts}"
echo "   MODELS_DIR:    ${MODELS_DIR:-models}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd /app
exec python twin_runner.py
