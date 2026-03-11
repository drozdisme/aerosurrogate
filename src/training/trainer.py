"""Training pipeline for AeroSurrogate v3.0.

v3 changes over v2:
  - FNO field model replaces DeepONet (resolution-independent Cp prediction)
  - Data pipeline integration: v3 dataset builder (XFOIL + UIUC) used when
    no pre-built dataset is available
  - MLflow experiment renamed to aerosurrogate_v3
  - meta.json includes fno_trained flag and field_model_type key
  - DeepONet kept as fallback import (backward compat with old checkpoints)
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

from src.data.ingestion import DataIngestion
from src.data.validator import DataValidator
from src.data.splitter import split_data
from src.features.engineer import FeatureEngineer
from src.features.scaler_store import ScalerStore
from src.models.ensemble import EnsembleModel
from src.uncertainty.scorer import ConfidenceScorer

# FNO is the v3 default field model; DeepONet kept for backward compatibility
try:
    from models.fno.fno2d import FNOModel
    _HAS_FNO = True
except ImportError:
    _HAS_FNO = False

try:
    from src.models.deeponet import DeepONetModel
    _HAS_DEEPONET = True
except ImportError:
    _HAS_DEEPONET = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_dir: str = "configs") -> dict:
    p = Path(config_dir)
    with open(p / "model_config.yaml") as f:
        model_cfg = yaml.safe_load(f)
    with open(p / "feature_config.yaml") as f:
        feature_cfg = yaml.safe_load(f)
    return {"model": model_cfg, "feature": feature_cfg}


def train(
    data_path: str = None,
    config_dir: str = "configs",
    data_sources_path: str = "configs/data_sources.yaml",
) -> None:
    """Full v2.0 training pipeline.

    1. Ingest data from all configured sources (or single CSV)
    2. Validate
    3. Feature engineering (v2: expanded derived features)
    4. Split
    5. Scale
    6. Optional: Optuna hyperparameter tuning
    7. Train ensemble (XGB + LGBM + N×MLP per target)
    8. Optional: Train DeepONet on field data
    9. Fit uncertainty scorer
    10. Evaluate and save
    """
    cfg = load_config(config_dir)
    model_cfg = cfg["model"]
    feature_cfg = cfg["feature"]
    targets = model_cfg["targets"]["integral"]

    artifacts_dir = Path(model_cfg.get("artifacts_dir", "artifacts"))
    models_dir = artifacts_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ── MLflow (best-effort) ───────────────────────────────────
    mlflow_active = False
    try:
        import mlflow
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("aerosurrogate_v3")
        mlflow.start_run(run_name="v3_training")
        mlflow_active = True
    except Exception:
        pass

    # ── 1. Load data ───────────────────────────────────────────
    if data_path and Path(data_path).exists():
        logger.info(f"Loading single CSV: {data_path}")
        from src.data.loader import load_csv
        df = load_csv(data_path)
    else:
        logger.info("Running data ingestion pipeline...")
        ingestion = DataIngestion(data_sources_path)
        df = ingestion.load_all()
        ingestion.print_summary()

    # ── 2. Validate ────────────────────────────────────────────
    required = feature_cfg.get("required_columns_min", ["mach", "reynolds", "alpha", "Cl"])
    validator = DataValidator(required_columns=required)
    df = validator.validate(df)

    # Ensure all targets exist
    for t in targets:
        if t not in df.columns:
            if t == "K" and "Cl" in df.columns and "Cd" in df.columns:
                df["K"] = df["Cl"] / df["Cd"].clip(lower=1e-8)
            elif t == "Cm" and "Cl" in df.columns:
                df["Cm"] = -0.25 * df["Cl"]
                logger.warning(f"Target '{t}' missing, using approximation")
            else:
                logger.error(f"Target '{t}' missing and cannot be derived")
                targets = [tt for tt in targets if tt in df.columns or tt == "K"]

    # ── 3. Feature engineering ─────────────────────────────────
    engineer = FeatureEngineer(feature_config=feature_cfg)
    df = engineer.transform(df)

    feature_cols = engineer.get_feature_columns(df, targets)
    logger.info(f"Features ({len(feature_cols)}): {feature_cols[:10]}...")
    logger.info(f"Targets ({len(targets)}): {targets}")
    logger.info(f"Dataset: {len(df)} samples")

    # ── 4. Split ───────────────────────────────────────────────
    split = split_data(
        df, feature_cols=feature_cols, target_cols=targets,
        test_size=model_cfg["training"]["test_size"],
        val_size=model_cfg["training"]["val_size"],
        random_state=model_cfg["training"]["random_state"],
    )

    # ── 5. Scale ───────────────────────────────────────────────
    scaler = ScalerStore()
    X_train_s = scaler.fit_feature_scaler(split.X_train)
    X_val_s = scaler.transform_features(split.X_val)
    X_test_s = scaler.transform_features(split.X_test)
    scaler.save(str(artifacts_dir))

    meta = {
        "feature_names": feature_cols,
        "target_names": targets,
        "n_features": len(feature_cols),
        "version": "2.0",
        "n_train": len(split.X_train),
    }
    with open(artifacts_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── 6. Optional tuning ─────────────────────────────────────
    tuning_cfg = model_cfg.get("tuning", {})
    if tuning_cfg.get("enabled", False):
        logger.info("Running hyperparameter tuning...")
        from src.training.tuner import run_tuning
        target_idx = targets.index(tuning_cfg.get("target", "Cl"))
        best_params = run_tuning(
            X_train_s, split.y_train, X_val_s, split.y_val,
            target_idx=target_idx,
            n_trials=tuning_cfg.get("n_trials", 100),
            timeout=tuning_cfg.get("timeout", 3600),
            metric=tuning_cfg.get("metric", "mape"),
        )
        # Merge tuned params with base config
        for key in ["xgboost", "lightgbm", "mlp"]:
            if key in best_params and best_params[key]:
                model_cfg["models"][key].update(best_params[key])
        logger.info("Tuned parameters applied")

    # ── 7. Train ensemble ──────────────────────────────────────
    mlp_cfg = model_cfg["models"]["mlp"]
    n_seeds = mlp_cfg.get("n_seeds", 5)
    base_seed = mlp_cfg.get("base_seed", 42)

    ensemble = EnsembleModel(
        target_names=targets,
        weights=model_cfg["ensemble"]["weights"],
        xgb_params=model_cfg["models"]["xgboost"],
        lgbm_params=model_cfg["models"]["lightgbm"],
        mlp_params=mlp_cfg,
        input_dim=len(feature_cols),
        n_seeds=n_seeds,
        base_seed=base_seed,
    )
    ensemble.fit(X_train_s, split.y_train, X_val_s, split.y_val)

    # ── 8. Field model: FNO (v3 default) ──────────────────────
    #       Falls back to DeepONet if FNO not available or if
    #       explicit deeponet config is set in model_config.yaml.
    field_model_type = None
    field_trained = False

    # --- Try to get field data ---
    try:
        ingestion_obj = DataIngestion(data_sources_path)
        field_result = ingestion_obj.load_field_data(
            n_points=200
        )
    except Exception as e:
        logger.warning(f"Could not load field data: {e}")
        field_result = None

    if field_result is not None:
        params_df, Y_fields = field_result
        logger.info(f"Field dataset: {len(Y_fields)} samples × {Y_fields.shape[1]} points")

        # Re-engineer + scale the field parameter block
        params_df = engineer.transform(params_df)
        for col in feature_cols:
            if col not in params_df.columns:
                params_df[col] = 0.0
        X_field = params_df[feature_cols].values.astype(np.float32)
        X_field_s = scaler.transform_features(X_field)

        n_field = len(X_field_s)
        n_fval = max(1, int(n_field * 0.15))
        X_ft, X_fv = X_field_s[:-n_fval], X_field_s[-n_fval:]
        Y_ft, Y_fv = Y_fields[:-n_fval], Y_fields[-n_fval:]

        # ── v3: FNO field model (preferred) ──────────────────
        force_deeponet = model_cfg.get("models", {}).get("deeponet", {}).get("force", False)

        if _HAS_FNO and not force_deeponet:
            fno_cfg = model_cfg.get("models", {}).get("fno", {})
            logger.info("Training FNO field model (v3)...")
            fno = FNOModel(param_dim=len(feature_cols), params=fno_cfg)
            fno.fit(X_ft, Y_ft, X_fv, Y_fv)
            fno.save(str(models_dir / "fno.pt"))
            field_model_type = "fno"
            field_trained = True
            meta["fno_trained"] = True
            meta["field_model_type"] = "fno"

            if mlflow_active:
                import mlflow
                if fno.train_history:
                    mlflow.log_metric("fno_best_val_loss",
                        min(h["val_loss"] for h in fno.train_history))
                mlflow.log_param("fno_n_modes", fno.hparams["n_modes"])
                mlflow.log_param("fno_width", fno.hparams["width"])
                mlflow.log_param("fno_n_layers", fno.hparams["n_layers"])

        elif _HAS_DEEPONET:
            # ── Fallback: DeepONet ────────────────────────────
            deeponet_cfg = model_cfg.get("models", {}).get("deeponet", {})
            logger.info("Training DeepONet field model (fallback)...")
            deeponet = DeepONetModel(param_dim=len(feature_cols), params=deeponet_cfg)
            deeponet.fit(X_ft, Y_ft, X_fv, Y_fv)
            deeponet.save(str(models_dir / "deeponet.pt"))
            field_model_type = "deeponet"
            field_trained = True
            meta["deeponet_trained"] = True
            meta["field_model_type"] = "deeponet"
        else:
            logger.warning("No field model available (FNO and DeepONet both unavailable)")

        if field_trained:
            with open(artifacts_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

    # ── 9. Fit uncertainty scorer ──────────────────────────────
    _, train_std = ensemble.predict(X_train_s)

    scorer = ConfidenceScorer(
        variance_weight=model_cfg["uncertainty"]["variance_weight"],
        distance_weight=model_cfg["uncertainty"]["distance_weight"],
        high_threshold=model_cfg["uncertainty"]["thresholds"]["high"],
        medium_threshold=model_cfg["uncertainty"]["thresholds"]["medium"],
    )
    scorer.fit(X_train_s, train_std)

    # ── 10. Evaluate on test set ───────────────────────────────
    test_preds, test_std = ensemble.predict(X_test_s)

    for i, target in enumerate(targets):
        y_true = split.y_test[:, i]
        y_pred = test_preds[:, i]

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mask = np.abs(y_true) > 1e-8
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]))) * 100 if mask.sum() > 0 else 0.0
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2) + 1e-8
        r2 = float(1.0 - ss_res / ss_tot)

        logger.info(f"[{target}] MAE={mae:.6f}  RMSE={rmse:.6f}  MAPE={mape:.2f}%  R2={r2:.4f}")

        if mlflow_active:
            import mlflow
            mlflow.log_metric(f"{target}_mae", mae)
            mlflow.log_metric(f"{target}_rmse", rmse)
            mlflow.log_metric(f"{target}_mape", mape)
            mlflow.log_metric(f"{target}_r2", r2)

    # ── 11. Save everything ────────────────────────────────────
    ensemble.save(str(models_dir))

    scorer_params = scorer.get_params()
    with open(artifacts_dir / "uncertainty_params.json", "w") as f:
        json.dump(scorer_params, f, indent=2,
                  default=lambda x: float(x) if isinstance(x, (np.floating,)) else x)

    if mlflow_active:
        import mlflow
        mlflow.log_params({"n_train": len(split.X_train), "n_features": len(feature_cols),
                           "n_seeds": n_seeds, "field_model": field_model_type or "none"})
        mlflow.end_run()

    logger.info("=" * 60)
    logger.info("AeroSurrogate v3.0 training complete!")
    logger.info(f"Artifacts: {artifacts_dir}")
    logger.info(f"Models: {2 + n_seeds} per target × {len(targets)} targets = "
                f"{(2 + n_seeds) * len(targets)} estimators")
    if field_trained:
        logger.info(f"Field model: {field_model_type} (trained)")
    logger.info("=" * 60)


if __name__ == "__main__":
    data_file = sys.argv[1] if len(sys.argv) > 1 else None
    train(data_path=data_file)