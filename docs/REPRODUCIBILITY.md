# Reproducibility Guide

This document explains how to reproduce all results reported in this repository.

---

## Environment Setup

### System Requirements

- **Python**: 3.9, 3.10, or 3.11
- **OS**: Linux, macOS, Windows (Linux recommended for H2O)
- **RAM**: 8GB minimum, 16GB recommended for H2O/AutoGluon
- **Disk**: ~5GB for datasets + artifacts

### Installation

```bash
# Clone repository
git clone https://github.com/Pratikn03/AutoMLP.git
cd AutoMLP

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install dependencies (choose one)
pip install -r Project/requirements_min.txt   # Minimal (XGBoost, LightGBM, FLAML)
pip install -r Project/requirements.txt       # Full (includes H2O, SHAP)
```

### Verified Package Versions

From `Project/requirements_min.txt`:

```
numpy==1.26.4
pandas==2.2.2
scikit-learn==1.4.2
xgboost==2.1.2
lightgbm==4.5.0
flaml==2.3.3
shap==0.45.1
h2o==3.46.0.1
```

---

## Reproducing Results

### Full Pipeline (Recommended)

```bash
# From repository root
python scripts/run_all.py --max-datasets 3 --flaml-time-budget 60

# For complete reproduction with all datasets
python scripts/run_all.py --max-datasets 50 --flaml-time-budget 300
```

**Expected outputs:**
- `reports/leaderboard.csv` — aggregated framework comparison
- `reports/summary_ci.csv` — 95% confidence intervals
- `reports/paired_tests.csv` — statistical significance tests
- `figures/*.png` — all visualization plots
- `runs/{dataset}/` — per-dataset snapshots

### Individual Components

#### 1. Guardrails Audit

```bash
python scripts/run_guardrails.py --data-root src/data/datasets/tabular
```

Output: `reports/guardrails/*.json`

#### 2. Boosting Suite

```bash
python scripts/run_boosting_suite.py \
  --models xgboost lightgbm catboost \
  --seeds 42 77 \
  --splits 5
```

Output: `reports/metrics/boosting_suite_*.csv`

#### 3. AutoML Suite

```bash
python scripts/run_automl_suite.py \
  --frameworks flaml h2o autogluon lightautoml \
  --time-limit 600 \
  --seeds 42
```

Output: `reports/metrics/automl_suite_*.csv`

#### 4. Feature Ablations

```bash
python scripts/run_feature_ablation.py \
  --estimators xgboost lightgbm catboost \
  --seeds 42
```

Output: `reports/metrics/feature_ablation_*.csv`, `reports/feature_ablation_summary.csv`

#### 5. SHAP Explanations

```bash
python Project/analysis/explain_shap.py
```

Output: `figures/shap/*.png`, `reports/shap_global_summary.csv`

#### 6. Statistical Analysis

```bash
python Project/analysis/analyze_stats.py
```

Output: `reports/summary_ci.csv`, `reports/paired_tests.csv`

#### 7. Visualization

```bash
python Project/analysis/plot_comparisons.py
```

Output: `figures/*.png` (violin plots, Pareto frontier, leaderboards)

---

## Random Seeds

All experiments use deterministic seeding:

| Component | Default Seeds | Environment Variable |
|-----------|--------------|---------------------|
| Train/test splits | 42 | `SEED` |
| CV outer loop | 42, 77 | `--seeds` CLI |
| FLAML internal | 42 | Set in AutoML config |
| NumPy/sklearn | 42 | Set in each script |

To reproduce with different seeds:

```bash
SEED=123 python scripts/run_boosting_suite.py --seeds 123 456
```

---

## Cross-Validation Protocol

The repository enforces **leakage-safe cross-validation**:

1. **Outer CV**: `StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)` for classification, `KFold` for regression
2. **Preprocessing**: `PreprocessingConfig` is cloned per fold; `fit()` called only on train indices
3. **Multi-seed**: Default runs use seeds `[42, 77]` for variance estimation

### Verification

Check `Project/experiments/runner.py` lines 100-200 for the CV implementation:

```python
# Preprocessing fitted inside fold
preprocessor = clone(self.config.preprocessing.build())
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)  # transform only
```

---

## Dataset Requirements

### Expected Location

Datasets are discovered in:
1. `Project/src/data/*.csv`
2. `src/data/*.csv`
3. `src/data/datasets/tabular/*.csv`
4. Environment variable `DATASET_PATHS`

### Demo Dataset

A demo dataset must exist at one of:
- `Project/src/data/modeldata_demo.csv`
- `src/data/modeldata_demo.csv`

Verify with:

```bash
pytest tests/test_smoke.py::test_demo_csv_exists -v
```

---

## Time Budgets

AutoML frameworks are constrained to identical time budgets:

| Framework | Default Budget | CLI Override |
|-----------|---------------|--------------|
| FLAML | 60s | `--flaml-time-budget` or `FLAML_TIME_BUDGET` |
| H2O AutoML | 600s | `--time-limit` |
| AutoGluon | 600s | `--time-limit` |
| LightAutoML | 600s | `--time-limit` |

For fair comparison in papers, use `--time-limit 600` across all AutoML frameworks.

---

## Hardware Considerations

### Memory

- **Low memory mode**: Enabled by default (`LOW_MEMORY_MODE=1`)
- **H2O**: Requires Java; may consume significant heap
- **AutoGluon**: Multi-layer stacking can be memory-intensive

### GPU

- XGBoost/LightGBM: Set `tree_method='gpu_hist'` for GPU acceleration
- PyTorch models (`image_cnn_torch.py`): Auto-detect CUDA

---

## Validation Checklist

After running the full pipeline, verify:

```bash
# 1. Leaderboard exists and is non-empty
test -s reports/leaderboard.csv && echo "✓ Leaderboard OK"

# 2. Statistical summaries exist
test -s reports/summary_ci.csv && echo "✓ CI summary OK"
test -s reports/paired_tests.csv && echo "✓ Paired tests OK"

# 3. SHAP outputs exist
test -s reports/shap_global_summary.csv && echo "✓ SHAP summary OK"
ls figures/shap/*.png 2>/dev/null | head -1 && echo "✓ SHAP figures OK"

# 4. Per-dataset runs archived
ls runs/*/reports/leaderboard.csv 2>/dev/null | wc -l

# 5. Run smoke tests
pytest tests/test_smoke.py -v
```

---

## Known Limitations

1. **H2O requires Java**: Install JDK 8+ or 11+ before running H2O experiments
2. **AutoGluon disk usage**: Creates temp directories; clean with `--skip-model-save`
3. **auto-sklearn/TPOT**: Require additional dependencies not in `requirements_min.txt`
4. **Vision/Audio pipelines**: Require PyTorch and dataset downloads (see `notebooks/13_vision_pipeline.ipynb`)

---

## Troubleshooting

### H2O fails to start

```bash
# Check Java
java -version

# Set H2O memory
export H2O_MEMORY=4g
```

### SHAP import error

```bash
pip install shap==0.45.1
```

### Out of memory

```bash
# Enable low-memory mode
LOW_MEMORY_MODE=1 python scripts/run_all.py --max-datasets 1
```

### Guardrails not running

```bash
# Ensure guardrails script exists and is wired in
python scripts/run_guardrails.py --data-root Project/src/data
```

---

## Reproducing Specific Tables/Figures

| Output | Command |
|--------|---------|
| Table 1 (Leaderboard) | `python Project/analysis/summarize_all.py` |
| Figure 1 (Pareto) | `python Project/analysis/plot_comparisons.py` |
| Figure 2 (SHAP) | `python Project/analysis/explain_shap.py` |
| Table 2 (CIs) | `python Project/analysis/analyze_stats.py` |

---

## Contact

For reproducibility issues, open a GitHub issue with:
1. Python version (`python --version`)
2. Package versions (`pip freeze`)
3. Full error traceback
4. Command that failed
