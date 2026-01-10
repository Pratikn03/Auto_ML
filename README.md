# AutoML: Automating the Traditional Machine Learning Methodology
## A Leakage-Audited AutoML Benchmark (Guardrails + Fair Comparison + Explainability + Deployment)

**Author:** Pratik Niroula  
**Instructor:** Dr. Suboh M. Alkhushayni  
**Type:** Research Paper + Reproducible Benchmark Code  
**Keywords:** AutoML, Data Leakage, Cross-Validation, Boosting, Explainability, Deployment, Reproducibility

---

## Quick links

| Resource | Location |
|----------|----------|
| 📄 **Paper (DOCX)** | [`paper/AutoML - CIS_Final.docx`](paper/AutoML%20-%20CIS_Final.docx) |
| 📄 **Paper (PDF)** | [`docs/AutoML.pdf`](docs/AutoML.pdf) |
| 🔁 **Reproducibility** | [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) |
| 🧾 **Datasets** | [`docs/DATASETS.md`](docs/DATASETS.md) |
| 🏗️ **Architecture** | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| ▶️ **Run demo** | `python scripts/run_all.py --max-datasets 3` |
| 📊 **Leaderboard** | `streamlit run Project/streamlit_leaderboard.py` |

---

## What this research is solving (the real problems)

AutoML can produce high accuracy quickly, but real-world ML often fails because results are **not trustworthy** or **not reproducible**. This research targets the practical problems that make AutoML results unreliable:

1. **Data leakage** makes models look "better than reality" (duplicate rows across folds, group/ID leakage, time leakage, and label leakage hidden in filenames/folder paths).
2. **Unfair comparisons** occur when frameworks are compared with different splits, seeds, budgets, preprocessing, or evaluation protocols.
3. **Evaluation is incomplete** when papers report only accuracy/F1 but ignore deployment constraints (latency, model size, runtime).
4. **Results are not reproducible** when seeds, folds, configs, and artifacts are not saved.
5. **Black-box models** are difficult to justify without explainability (global + local explanations).

This repo provides a leakage-audited benchmark protocol + implementation that addresses all five.

---

## Main contribution (what I built)

A reproducible AutoML benchmarking pipeline with:

- **AutoML Guardrails**: automated leakage auditing before training.
- **Leakage-safe cross-validation**: preprocessing fit only on training folds.
- **Fair benchmarking**: same splits, seeds, and time budgets across models.
- **Multiple model families**: AutoML frameworks + strong boosting baselines.
- **Ablation studies**: quantify impact of feature engineering choices.
- **Statistical reporting**: confidence intervals + paired tests.
- **Explainability**: SHAP-based interpretation artifacts.
- **Deployment readiness**: FastAPI inference + Streamlit leaderboard + Docker/monitoring.

---

## Techniques used (what methods/algorithms are inside)

### 1) Leakage prevention & evaluation integrity
- Fold-safe preprocessing using **scikit-learn Pipelines** and **ColumnTransformer**
- Consistent CV protocol (KFold / StratifiedKFold; configurable)
- Guardrails checks for:
  - cross-fold duplicates / near-duplicates
  - group/entity leakage (ID appears in train and validation)
  - time/look-ahead leakage for time series tasks
  - label/path leakage for vision datasets (tokens inside folder/file names)

### 2) Feature engineering & preprocessing techniques
- Imputation (numeric/categorical)
- Scaling/normalization (when applicable)
- Polynomial feature expansion (ablation)
- Quantile/binning (KBins discretization; ablation)
- Multicollinearity filtering via **VIF** (ablation)
- Categorical handling depending on model family (CatBoost native; others encoded)

### 3) Models compared (families)

**AutoML frameworks**
- AutoGluon
- LightAutoML
- H2O AutoML
- FLAML
- (Optional if enabled): auto-sklearn, TPOT

**Boosting baselines**
- XGBoost
- LightGBM
- CatBoost

### 4) Explainability
- SHAP global feature importance
- Optional local explanations for specific samples

### 5) Deployment & MLOps techniques
- Model packaging (saved pipelines/artifacts)
- FastAPI serving endpoints
- Docker Compose deployment
- Prometheus metrics endpoint (monitoring)
- CI smoke tests to ensure the pipeline runs in automation

---

## Task breakdown (Task 1–11) — what we did across the full work

### Task 1 — Define research objective & benchmark protocol
- Defined the research question: "How to compare AutoML fairly while preventing leakage?"
- Set evaluation requirements: consistent splits, seeds, budgets, metrics.

### Task 2 — Dataset intake + task typing
- Structured dataset intake for tabular (classification/regression), plus optional vision/audio.
- Defined dataset format expectations and where data should be placed (see [`docs/DATASETS.md`](docs/DATASETS.md)).

### Task 3 — AutoML Guardrails (leakage audit)
- Implemented pre-training leakage checks (duplicates, group/time leakage, token/path leakage).
- Output: guardrail reports saved for transparency.
- Script: `scripts/run_guardrails.py`

### Task 4 — Leakage-safe preprocessing design
- Implemented fold-safe preprocessing so transforms are trained only on training folds.
- Output: pipelines are reproducible and consistent across models.
- Utilities: `Project/utils/standardize.py`, `Project/utils/sanitize.py`

### Task 5 — Baseline suite (Boosting models)
- Implemented XGBoost/LightGBM/CatBoost training + evaluation.
- Logged metrics per fold, saved artifacts.
- Scripts: `Project/trainers/train_boosters.py`, `train_catboost.py`

### Task 6 — AutoML suite (Framework comparisons)
- Ran AutoML frameworks under identical budgets/seeds/splits.
- Logged per-framework outputs consistently.
- Scripts: `Project/trainers/train_flaml.py`, `train_h2o.py`

### Task 7 — Feature ablation studies
- Ran controlled ablations:
  - no-scaling vs scaling
  - polynomial features on/off
  - binning on/off
  - VIF filtering on/off
- Purpose: quantify what preprocessing truly improves performance.
- Script: `scripts/run_feature_ablation.py`
- Analysis: `Project/analysis/analyze_feature_ablations.py`

### Task 8 — Metrics + deployment-aware evaluation
- Computed standard ML metrics (Accuracy/F1/ROC-AUC etc.)
- Logged operational metrics (latency, model size) when applicable.
- Analysis: `Project/analysis/analyze_stats.py`

### Task 9 — Statistical validity
- Aggregated per-fold results.
- Generated 95% confidence intervals and paired comparisons.
- Script: `scripts/run_sensitivity_analysis.py`
- Output: `reports/` contains CSV summaries

### Task 10 — Explainability artifacts
- Generated SHAP feature importance summaries.
- Saved explainability outputs for reporting and transparency.
- Script: `Project/analysis/explain_shap.py`
- Output: `figures/feature_importance/`

### Task 11 — Packaging + presentation + reproducibility
- Streamlit leaderboard for result visualization: `Project/streamlit_leaderboard.py`
- FastAPI service for deployment demonstration: `Deploy/api/serve/app.py`
- Reproducibility docs + dataset docs + CI smoke tests.

---

## Repository layout (actual structure)

```text
.
├── README.md                      # This file
├── Makefile                       # Build/run shortcuts
├── pyproject.toml                 # Package metadata
├── pytest.ini                     # Test configuration
├── requirements.txt               # Dependencies (CI)
│
├── paper/                         # 📄 Research paper (main deliverable)
│   └── AutoML - CIS_Final.docx
│
├── docs/                          # 📚 Documentation
│   ├── ARCHITECTURE.md            # System design
│   ├── DATASETS.md                # Dataset registry & formats
│   ├── REPRODUCIBILITY.md         # Reproduction guide
│   ├── MEMORY_OPTIMIZATION.md     # Memory tuning tips
│   ├── RELATED_WORK.md            # Literature context
│   ├── FIGURE_GUIDE.md            # Figure explanations
│   ├── sensitivity_analysis.md    # Sensitivity study notes
│   ├── AutoML.pdf                 # Paper PDF export
│   └── README_RUN.md              # Quick run guide
│
├── Project/                       # 🔧 Core implementation
│   ├── trainers/                  # Model training scripts
│   │   ├── train_boosters.py      #   XGBoost/LightGBM/CatBoost
│   │   ├── train_catboost.py      #   CatBoost standalone
│   │   ├── train_flaml.py         #   FLAML AutoML
│   │   └── train_h2o.py           #   H2O AutoML
│   ├── analysis/                  # Post-training analysis
│   │   ├── analyze_stats.py       #   Statistical summaries
│   │   ├── analyze_feature_ablations.py
│   │   ├── explain_shap.py        #   SHAP explanations
│   │   ├── plot_comparisons.py    #   Visualization
│   │   └── summarize_all.py       #   Aggregate results
│   ├── utils/                     # Shared utilities
│   │   ├── io.py                  #   File I/O helpers
│   │   ├── memory.py              #   Memory optimization
│   │   ├── sanitize.py            #   Data cleaning
│   │   ├── standardize.py         #   Preprocessing
│   │   └── system.py              #   System utilities
│   ├── deeplearning/              # Neural network demos
│   │   ├── image_cnn_torch.py     #   Vision CNN (PyTorch)
│   │   ├── audio_cnn_torch.py     #   Audio CNN (PyTorch)
│   │   └── tabular_keras.py       #   Tabular NN (Keras)
│   ├── anomaly/                   # Anomaly detection
│   │   └── tabular_anomaly.py
│   ├── timeseries/                # Time series forecasting
│   │   └── forecast_baseline.py
│   ├── nlp/                       # NLP demos
│   │   └── train_sms_spam.py      #   SMS spam classifier
│   ├── experiments/               # Experiment configs
│   ├── artifacts/                 # Saved models
│   ├── src/data/                  # Demo datasets
│   ├── streamlit_leaderboard.py   # 📊 Interactive dashboard
│   └── requirements*.txt          # Dependencies
│
├── scripts/                       # 🚀 CLI entrypoints
│   ├── run_all.py                 # Master orchestrator
│   ├── run_all_trainers.py        # Run all trainers
│   ├── run_automl_suite.py        # AutoML frameworks only
│   ├── run_boosting_suite.py      # Boosting baselines only
│   ├── run_feature_ablation.py    # Ablation experiments
│   ├── run_guardrails.py          # Leakage detection
│   ├── run_sensitivity_analysis.py# Sensitivity studies
│   ├── run_full_pipeline.sh       # Full pipeline (shell)
│   ├── download_datasets.py       # Dataset downloader
│   ├── stage_datasets.py          # Dataset preparation
│   ├── collect_dataset_stats.py   # Dataset statistics
│   ├── extract_audio_features.py  # Audio preprocessing
│   ├── generate_classifier_figures.py
│   ├── generate_readme_assets.py
│   └── plot_score_vs_time.py      # Performance plots
│
├── Deploy/                        # 🐳 Deployment
│   ├── api/                       # FastAPI service
│   │   ├── serve/app.py           #   Main API
│   │   ├── Dockerfile
│   │   └── requirements.txt
│   ├── k8s/                       # Kubernetes manifests
│   ├── monitoring/                # Prometheus/Grafana
│   └── docker-compose.yml         # Container orchestration
│
├── notebooks/                     # 📓 Jupyter notebooks
│   ├── 01_data_cleaning.ipynb
│   ├── 02_modeling_metrics.ipynb
│   ├── 03_deployment_ops.ipynb
│   ├── 05_boosting_walkthrough.ipynb
│   ├── 06_automl_comparison.ipynb
│   ├── 07_explainability.ipynb
│   ├── 08_calibration_fairness.ipynb
│   ├── 09_leaderboard_analysis.ipynb
│   ├── 10_anomaly_timeseries.ipynb
│   ├── 11_streamlit_demo.ipynb
│   ├── 12_final_report.ipynb
│   ├── 13_vision_pipeline.ipynb
│   ├── 14_audio_pipeline.ipynb
│   └── quick_run.ipynb
│
├── reports/                       # 📈 Results output
│   ├── leaderboard.csv            # Main results table
│   ├── runtime.json               # Timing data
│   ├── ablations/                 # Ablation results
│   └── metrics/                   # Per-model metrics
│
├── figures/                       # 📊 Generated plots
│   ├── ablations/                 # Ablation figures
│   └── feature_importance/        # SHAP plots
│
├── runs/                          # 🗂️ Per-dataset artifacts
│   ├── adult_income/
│   ├── breast_cancer/
│   ├── california_housing/
│   ├── cifar10/
│   └── ... (30+ datasets)
│
├── artifacts/                     # 💾 Model artifacts
│   ├── catboost/
│   ├── flaml/
│   ├── lightgbm/
│   ├── xgboost/
│   └── keras_dense/
│
├── src/                           # Source data
│   └── data/modeldata_demo.csv    # Demo dataset for CI
│
└── tests/                         # ✅ Test suite
    └── test_smoke.py              # CI smoke tests
```

---

## How to run (quick demo)

```bash
# Install minimal dependencies
pip install -r Project/requirements_min.txt

# Run benchmark on 3 datasets
python scripts/run_all.py --max-datasets 3

# View interactive leaderboard
streamlit run Project/streamlit_leaderboard.py
```

---

## For reviewers (recommended reading order)

1. **Read the paper:** [`paper/AutoML - CIS_Final.docx`](paper/AutoML%20-%20CIS_Final.docx)
2. **Reproducibility checklist:** [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md)
3. **Dataset formatting and registry:** [`docs/DATASETS.md`](docs/DATASETS.md)
4. **Architecture overview:** [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
5. **Run a small benchmark** to verify the pipeline works.

---

## Key results (high-level)

From the reported experimental figures:
- **H2O AutoML** achieves the strongest mean accuracy (≈ **97%** range reported in the discussion), with **XGBoost (poly2 ablation)** very close behind.
- **LightGBM** and **CatBoost** form a strong and consistent next tier.
- Statistical analysis includes 95% confidence intervals and paired significance tests.

---

## Authorship statement

This research paper, benchmark methodology, implementation, experiments, and documentation were created entirely by **Pratik Niroula**.

---

## Citation

```bibtex
@misc{niroula_automl_leakage_audited_2026,
  title  = {AutoML: Automating the Traditional Machine Learning Methodology},
  author = {Niroula, Pratik},
  year   = {2026},
  note   = {Leakage-audited AutoML benchmarking with guardrails, explainability, and deployment readiness}
}
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.
