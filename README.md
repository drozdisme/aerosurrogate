# AeroSurrogate v2.0

Surrogate model for aerodynamic coefficient and surface field prediction.
Supports 3D configurations, multi-source data ingestion, and DeepONet field models.

## What's new in v2.0

- **Multi-source data ingestion** — plug in CSV, SU2, OpenFOAM, VLM (AVL/VSPAERO), VTK field data via `configs/data_sources.yaml`
- **Expanded ensemble** — 5 MLP seeds (configurable) for better variance estimation; 28 total estimators
- **DeepONet** — branch-trunk neural operator for surface Cp(x) prediction (when field data available)
- **Physics-informed loss** — Cd positivity, Cl monotonicity constraints
- **Optuna tuning** — optional hyperparameter optimization
- **3D features** — sweep-Mach interaction, compressibility factor, reduced frequency, CST support
- **Field prediction API** — `POST /predict/field` returns Cp distribution

## Quick Start

```bash
docker-compose up --build
```

Services: API (8000), UI (8501), MLflow (5000).

## Adding Your Own Data

Edit `configs/data_sources.yaml`:

```yaml
sources:
  my_cfd_data:
    adapter: csv
    path: data/my_dataset.csv
    enabled: true
    column_map:
      mach: Mach_Number
      reynolds: Re
      alpha: AoA_deg
      Cl: CL
      Cd: CD
```

Then retrain:

```bash
python -m src.training.trainer
```

### Supported adapters

| Adapter   | Format                           | Use case                    |
|-----------|----------------------------------|-----------------------------|
| `csv`     | CSV/TSV with column mapping      | Purchased datasets, XFOIL   |
| `su2`     | SU2 case directories             | SU2 CFD results             |
| `openfoam`| OpenFOAM case directories        | OpenFOAM RANS results       |
| `vlm`     | AVL .st files, VSPAERO .polar    | Vortex lattice methods      |
| `vtk`     | VTK/numpy field arrays           | DeepONet training (Cp fields)|

### Training on purchased datasets

1. Place your CSV/data in the `data/` directory
2. Add an entry to `configs/data_sources.yaml` with `column_map` matching your column names to the standard names
3. Run `python -m src.training.trainer`
4. Check metrics in MLflow at `localhost:5000`

### Enabling DeepONet (field predictions)

Add a `vtk`-type source with surface Cp data:

```yaml
sources:
  airfrans:
    adapter: vtk
    path: data/airfrans/
    enabled: true
    field_name: Cp
```

Provide either:
- `manifest.csv` + `fields/0000.npy` (numpy arrays)
- Case directories with `params.csv` + `surface.csv`

## API Endpoints

| Method | Path             | Description              |
|--------|------------------|--------------------------|
| GET    | /health          | Status + capability check|
| POST   | /predict         | Single integral prediction|
| POST   | /predict/batch   | Batch integral prediction|
| POST   | /predict/field   | Cp(x) field prediction   |

## Project Structure

```
src/
├── data/
│   ├── adapters/        # CSV, SU2, OpenFOAM, VLM, VTK parsers
│   ├── ingestion.py     # Multi-source orchestrator
│   ├── loader.py        # CSV loader + synthetic generator
│   ├── validator.py     # Physical bounds checking
│   └── splitter.py      # Train/val/test split
├── features/
│   ├── engineer.py      # Derived features (v2: expanded)
│   ├── scaler_store.py  # Scaler persistence
│   └── cst.py           # CST airfoil parameterization
├── models/
│   ├── base_model.py    # Abstract interface
│   ├── xgb_model.py     # XGBoost
│   ├── lgbm_model.py    # LightGBM
│   ├── mlp_model.py     # PyTorch MLP
│   ├── deeponet.py      # DeepONet field model
│   ├── ensemble.py      # Multi-seed weighted ensemble
│   └── physics_loss.py  # Physics-informed penalties
├── training/
│   ├── trainer.py       # Full pipeline
│   └── tuner.py         # Optuna hyperparameter search
├── uncertainty/         # Variance + Mahalanobis scoring
├── inference/           # Predictor (integral + field)
├── api/                 # FastAPI
└── ui/                  # Streamlit
```

## Configuration

- `configs/model_config.yaml` — hyperparameters, ensemble weights, physics loss, tuning
- `configs/feature_config.yaml` — feature lists, CST config, scaling
- `configs/data_sources.yaml` — data source registry (add your datasets here)
