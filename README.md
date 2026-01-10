# AutoMLP — Leakage-Audited AutoML Benchmarking & Deployment Toolkit

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)](Project/streamlit_leaderboard.py)

> End-to-end AutoML system featuring **leakage-safe cross-validation**, **guardrail audits**, **multi-framework benchmarking**, **SHAP explanations**, and **production deployment** (FastAPI + Docker + Prometheus).

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [Key Features](#key-features)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Frameworks & Baselines](#frameworks--baselines)
6. [Quickstart](#quickstart)
7. [Configuration](#configuration)
8. [Output Artifacts](#output-artifacts)
9. [Deployment](#deployment)
10. [Notebooks](#notebooks)
11. [Tests](#tests)
12. [Contributing](#contributing)

---

## Overview

AutoMLP is a **leakage-audited, multimodal AutoML benchmarking system** that evaluates tabular, vision, and audio ML pipelines under identical conditions. It enforces:

- **Leakage-safe CV**: All preprocessing fitted strictly inside CV folds (train-only)
- **Fair comparison**: Identical time budgets (300–600s), seeds, and splits across frameworks
- **Guardrail audits**: Automatic detection of duplicate rows, proxy leakage, path/token leakage
- **Production metrics**: Inference latency (p50/p95), model size, Pareto-optimal picks
- **Reproducibility**: Deterministic seeds, JSON configs, complete artifact logging

---

## Repository Structure

```
AutoMLP/
├── scripts/                    # Main CLI entrypoints
│   ├── run_all.py              # Master orchestrator (runs full pipeline per dataset)
│   ├── run_boosting_suite.py   # XGBoost/LightGBM/CatBoost experiments
│   ├── run_automl_suite.py     # AutoGluon/LightAutoML/FLAML/H2O experiments
│   ├── run_feature_ablation.py # Feature engineering ablations
│   ├── run_guardrails.py       # Leakage detection audit
│   └── run_sensitivity_analysis.py
│
├── Project/
│   ├── trainers/               # Individual framework trainers
│   │   ├── train_boosters.py   # XGBoost + LightGBM baseline
│   │   ├── train_catboost.py   # CatBoost baseline
│   │   ├── train_flaml.py      # FLAML AutoML
│   │   └── train_h2o.py        # H2O AutoML
│   ├── experiments/            # Reusable experiment infrastructure
│   │   ├── runner.py           # ExperimentRunner with CV, metrics, artifact saving
│   │   ├── preprocessing.py    # PreprocessingConfig (scaling, binning, VIF)
│   │   ├── boosting.py         # Boosting suite with optional Optuna tuning
│   │   ├── automl.py           # AutoML wrappers (AutoGluon, LightAutoML, FLAML, H2O)
│   │   └── ablations.py        # Feature ablation variants
│   ├── analysis/               # Post-training analysis
│   │   ├── analyze_stats.py    # CI computation, paired t-tests
│   │   ├── explain_shap.py     # SHAP/LIME explanations
│   │   ├── plot_comparisons.py # Visualization generation
│   │   └── summarize_all.py    # Aggregate leaderboards
│   ├── deeplearning/           # Neural network baselines
│   │   ├── tabular_keras.py    # Keras MLP for tabular
│   │   ├── image_cnn_torch.py  # ResNet for vision (CIFAR10)
│   │   └── audio_cnn_torch.py  # CNN for audio
│   ├── utils/                  # Shared utilities (io, sanitize, memory)
│   └── src/data/               # Demo datasets location
│
├── Deploy/
│   ├── api/
│   │   ├── serve/app.py        # FastAPI prediction service
│   │   ├── Dockerfile          # Container build
│   │   └── requirements*.txt   # API dependencies
│   ├── docker-compose.yml      # API + Prometheus + Grafana stack
│   ├── monitoring/prometheus.yml
│   └── k8s/                    # Kubernetes manifests (optional)
│
├── reports/                    # Generated outputs (metrics, leaderboards)
├── figures/                    # Generated plots (SHAP, violin, Pareto)
├── artifacts/                  # Saved models, pipelines
├── runs/                       # Per-dataset snapshots
├── notebooks/                  # Jupyter tutorials (01–14)
├── tests/                      # Smoke tests
├── Makefile                    # Common targets
└── pyproject.toml              # Package metadata
```

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Guardrails Audit** | Detects duplicate rows, high-cardinality proxies, path/token leakage before training |
| **Leakage-Safe CV** | `PreprocessingConfig` + `ExperimentRunner` fit transforms on train folds only |
| **Multi-Framework** | AutoGluon, LightAutoML, FLAML, H2O, (auto-sklearn, TPOT optional) |
| **Boosting Baselines** | XGBoost, LightGBM, CatBoost with optional Optuna/RandomSearch tuning |
| **Feature Ablations** | Polynomial features, VIF filtering, quantile binning, no-scaling variants |
| **SHAP Explanations** | Global feature importance + local sample explanations |
| **Statistical Validation** | 95% CIs, paired t-tests, multi-seed CV |
| **Deployment Ready** | FastAPI + Prometheus metrics + Docker Compose + K8s manifests |

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         scripts/run_all.py                               │
│  (Discovers datasets, orchestrates per-dataset pipeline, collects runs) │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         ▼                          ▼                          ▼
┌─────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│ run_guardrails  │     │  train_boosters.py  │     │  train_flaml.py  │
│   (leakage      │     │  train_catboost.py  │     │  train_h2o.py    │
│    detection)   │     │  (XGB/LGB/CB)       │     │  (AutoML)        │
└─────────────────┘     └─────────────────────┘     └──────────────────┘
         │                          │                          │
         ▼                          ▼                          ▼
┌─────────────────────────────────────────────────────────────────────────┐
│              Project/experiments/runner.py (ExperimentRunner)            │
│  • StratifiedKFold / KFold with configurable seeds                      │
│  • Preprocessing inside fold (fit on train, transform on test)          │
│  • Metric computation (accuracy, F1, ROC-AUC, RMSE, MAE, R²)            │
│  • Artifact saving (joblib pipelines, CSVs, JSON configs)               │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        Project/analysis/*                                │
│  • analyze_stats.py   → reports/summary_ci.csv, paired_tests.csv        │
│  • explain_shap.py    → figures/shap/*, reports/shap_*.csv              │
│  • plot_comparisons.py → figures/*.png (violin, Pareto, leaderboard)    │
│  • summarize_all.py   → reports/leaderboard.csv, framework_summary.csv  │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                            Output Artifacts                              │
│  • runs/{dataset}/reports/  — per-dataset leaderboard + metrics         │
│  • reports/                 — aggregated leaderboards                   │
│  • figures/                 — all generated plots                       │
│  • artifacts/experiments/   — saved pipelines (.joblib)                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### Leakage Prevention

1. **PreprocessingConfig** (`Project/experiments/preprocessing.py`): Defines imputation, scaling, polynomial features, VIF filtering — all wrapped in sklearn `Pipeline`
2. **ExperimentRunner** (`Project/experiments/runner.py`): Clones the preprocessor per fold, fits only on `X_train[train_idx]`
3. **Guardrails** (`scripts/run_guardrails.py`): Pre-training audit for duplicate rows, deterministic proxies, path/token leakage

---

## Frameworks & Baselines

| Framework | Type | Script | Notes |
|-----------|------|--------|-------|
| **XGBoost** | Boosting | `train_boosters.py`, `run_boosting_suite.py` | GPU optional |
| **LightGBM** | Boosting | `train_boosters.py`, `run_boosting_suite.py` | Fast histogram-based |
| **CatBoost** | Boosting | `train_catboost.py`, `run_boosting_suite.py` | Native categorical support |
| **FLAML** | AutoML | `train_flaml.py`, `run_automl_suite.py` | Fast lightweight AutoML |
| **H2O AutoML** | AutoML | `train_h2o.py`, `run_automl_suite.py` | Stacking, Java backend |
| **AutoGluon** | AutoML | `run_automl_suite.py` | Multi-layer stacking |
| **LightAutoML** | AutoML | `run_automl_suite.py` | Sber's AutoML |
| **auto-sklearn** | AutoML | `run_automl_suite.py` | Optional (extra deps) |
| **TPOT** | AutoML | `run_automl_suite.py` | Optional (extra deps) |
| **Keras MLP** | Deep Learning | `tabular_keras.py` | Simple dense network |
| **ResNet** | Vision | `image_cnn_torch.py` | CIFAR10/Caltech101 |

---

## Quickstart

### 1. Install Dependencies

```bash
# Minimal (boosters + FLAML)
pip install -r Project/requirements_min.txt

# Full (includes H2O, SHAP)
pip install -r Project/requirements.txt

# Streamlit dashboard
pip install -r Project/requirements_streamlit.txt
```

### 2. Run Full Pipeline

```bash
# From repository root
python scripts/run_all.py --max-datasets 3

# With specific settings
python scripts/run_all.py \
  --max-datasets 5 \
  --flaml-time-budget 120 \
  --prefer-small-datasets
```

### 3. Run Individual Suites

```bash
# Boosting suite only
python scripts/run_boosting_suite.py --models xgboost lightgbm catboost

# AutoML suite
python scripts/run_automl_suite.py --frameworks flaml h2o --time-limit 300

# Feature ablation
python scripts/run_feature_ablation.py --estimators xgboost catboost
```

### 4. View Results

```bash
# Streamlit leaderboard
streamlit run Project/streamlit_leaderboard.py

# Or check CSVs directly
cat reports/leaderboard.csv
cat reports/summary_ci.csv
```

### 5. Serve API

```bash
# Local
make serve
# or
uvicorn Deploy.api.serve.app:app --port 8000

# Docker
make demo
# or
docker compose -f Deploy/docker-compose.yml up
```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CSV_PATH` | auto-discovered | Path to dataset CSV |
| `TARGET` | auto-inferred | Target column name |
| `SEED` | `42` | Random seed |
| `N_SPLITS` | `5` | CV folds |
| `FLAML_TIME_BUDGET` | `60` | FLAML search budget (seconds) |
| `RUN_ALL_MAX_DATASETS` | `3` | Max datasets for run_all.py |
| `LOW_MEMORY_MODE` | `1` | Enable memory optimizations |
| `SHAP_GLOBAL_SAMPLES` | `400` | Samples for SHAP computation |
| `API_ENABLE_H2O` | `false` | Load H2O model in API |

### CLI Arguments (`run_all.py`)

```
--max-datasets N         Maximum datasets to process
--flaml-time-budget S    FLAML time budget (seconds)
--step-timeout S         Timeout per pipeline step (0=disabled)
--prefer-small-datasets  Sort datasets by size ascending
--frameworks F1 F2       Filter to specific frameworks
--skip-guardrails        Skip leakage audit step
```

---

## Output Artifacts

### Reports (`reports/`)

| File | Description |
|------|-------------|
| `leaderboard.csv` | Aggregated framework comparison |
| `summary_ci.csv` | 95% confidence intervals per framework |
| `paired_tests.csv` | Paired t-test results |
| `framework_summary.json` | Detailed framework metadata |
| `shap_global_summary.csv` | SHAP feature importances |
| `metrics/*.csv` | Per-fold metrics for each experiment |
| `guardrails/*.json` | Leakage audit results |

### Figures (`figures/`)

| File | Description |
|------|-------------|
| `classifier_accuracy_precision.png` | Scatter: accuracy vs precision |
| `pareto_accuracy_runtime.png` | Pareto frontier: accuracy vs runtime |
| `metric_violin_*.png` | Distribution plots per metric |
| `leaderboard_*.png` | Bar charts of framework performance |
| `shap/*.png` | SHAP summary plots |
| `feature_importance/**/*.png` | Per-fold feature importance |

### Artifacts (`artifacts/`)

| Path | Contents |
|------|----------|
| `experiments/{name}/seed_{s}/fold_{f}_pipeline.joblib` | Fitted sklearn pipelines |
| `flaml/` | FLAML model artifacts |
| `h2o/` | H2O MOJO exports |
| `keras_dense/` | Keras model weights |

---

## Deployment

### FastAPI Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/readyz` | GET | Readiness probe |
| `/version` | GET | API version |
| `/metrics` | GET | Prometheus metrics |
| `/predict` | POST | Model inference |

### Docker Compose Stack

```bash
docker compose -f Deploy/docker-compose.yml up
```

Services:
- **api** (port 8000): FastAPI prediction service
- **prometheus** (port 9090): Metrics collection
- **grafana** (port 3000): Visualization dashboards

### Kubernetes

Basic manifests available in `Deploy/k8s/` for production deployment.

---

## Notebooks

| Notebook | Topic |
|----------|-------|
| `01_data_cleaning.ipynb` | Data preprocessing walkthrough |
| `02_modeling_metrics.ipynb` | Metric computation deep-dive |
| `05_boosting_walkthrough.ipynb` | XGBoost/LightGBM/CatBoost tutorial |
| `06_automl_comparison.ipynb` | AutoML framework comparison |
| `07_explainability.ipynb` | SHAP/LIME explanations |
| `08_calibration_fairness.ipynb` | Calibration & fairness analysis |
| `09_leaderboard_analysis.ipynb` | Results analysis |
| `13_vision_pipeline.ipynb` | Image classification pipeline |
| `14_audio_pipeline.ipynb` | Audio classification pipeline |

---

## Tests

```bash
# Run smoke tests
pytest tests/test_smoke.py -v

# Or via pytest.ini
pytest
```

Tests verify:
- Demo CSV existence
- Core module imports
- Analysis script imports
- FastAPI app import
- Reports directory structure

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Commit changes (`git commit -m 'Add amazing feature'`)
5. Push to branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{autompl2025,
  title = {AutoMLP: Leakage-Audited AutoML Benchmarking Toolkit},
  author = {AutoML Pro Contributors},
  year = {2025},
  url = {https://github.com/Pratikn03/AutoMLP}
}
```
