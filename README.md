# AutoML: Automating the Traditional Machine Learning Methodology
## A Leakage-Audited AutoML Benchmark with Guardrails, Explainability, and Deployment Readiness

**Author:** Pratik Niroula (sole author)  
**Instructor:** Dr. Suboh M. Alkhushayni  
**Type:** Research Paper (final)  
**Keywords:** AutoML, Data Leakage, Cross-Validation, Boosting, Explainability, Deployment, Reproducibility

---

## What this repository contains

This GitHub repository hosts my **final research paper** and supplementary materials (figures, configs, result tables, and implementation code) referenced in the paper.

- ✅ **Main deliverable:** `AutoML - CIS_Final.docx` (the full research paper)
- ✅ **Research focus:** a **leakage-audited AutoML benchmark** that compares modern AutoML frameworks and strong boosting baselines under identical conditions
- ✅ **Core contribution:** a practical, reproducible pipeline design that prevents "fake good" results caused by data leakage and evaluates models with both quality and deployment metrics

---

## Abstract (what the paper proves)

Machine Learning is widely used across domains, but building reliable models still requires expertise and time. AutoML reduces this barrier by automating preprocessing, feature engineering, model selection, hyperparameter tuning, and deployment. However, real-world ML suffers from **data leakage** and poor reproducibility.

This paper presents an AutoML pipeline that includes:
- **AutoML Guardrails** (cross-fold duplicates, group leakage, temporal leakage, and image path/token leakage)
- **Fold-aware preprocessing** (imputation, scaling, polynomial features, binning, and VIF filtering fit strictly inside each training fold)
- Standardized benchmarking across **AutoGluon, LightAutoML, H2O AutoML, FLAML** and **XGBoost, LightGBM, CatBoost**
- Evaluation with both ML metrics and deployment metrics (latency and model size)
- Statistical rigor (per-fold results, bootstrap 95% CI, paired tests)
- Explainability (SHAP + LIME) and deployment readiness (model packaging + FastAPI + Streamlit leaderboard)

---

## Research questions answered

1. **How can AutoML be made more trustworthy in real-world settings?**  
   By preventing leakage using guardrails + fold-aware preprocessing.

2. **How do modern AutoML frameworks compare to boosting baselines fairly?**  
   Run them on identical splits, seeds, and time budgets.

3. **How do we evaluate models beyond accuracy?**  
   Record inference latency and model size, then analyze tradeoffs.

---

## Methodology summary (pipeline)

### 1) Guardrails / Leakage Control (pre-training)
The pipeline checks for:
- Cross-fold duplicates
- Group/ID leakage
- Temporal/look-ahead leakage
- Image path/token leakage (label information hidden in folders/filenames)

### 2) Leakage-safe preprocessing (inside each fold)
All transformations are fit ONLY on the training fold:
- Imputation
- Scaling
- Polynomial feature generation
- KBins discretization (binning)
- VIF-based multicollinearity filtering

Implemented via scikit-learn Pipelines / ColumnTransformers to keep CV honest.

### 3) Fair benchmarking rules
- Same CV splits and random seeds
- Same time budgets (typically 300–600 seconds)
- Same datasets and tasks across frameworks/baselines

---

## Models compared (as described in the paper)

### AutoML frameworks
| Framework | Description |
|-----------|-------------|
| **AutoGluon** | Multi-layer stacking with automated ensembling |
| **LightAutoML** | Sber's production AutoML system |
| **H2O AutoML** | Java-based with stacking and metalearning |
| **FLAML** | Fast lightweight AutoML from Microsoft |

### Strong boosting baselines
| Framework | Description |
|-----------|-------------|
| **XGBoost** | Gradient boosting with regularization |
| **LightGBM** | Histogram-based gradient boosting |
| **CatBoost** | Native categorical feature support |

---

## Datasets & tasks covered

The paper describes a mixed benchmark that includes:
- **Tabular** classification and regression datasets
- **Vision**: CIFAR-10 (image classification)
- **Audio**: lightweight speech/audio datasets (e.g., FSDD / Speech Commands)

---

## Evaluation metrics

### Classification
- Accuracy, Precision, Recall, F1
- ROC-AUC (and PR-AUC for imbalanced settings when applicable)

### Regression
- RMSE (and/or MAE, R² where relevant)

### Deployment-relevant
- Inference latency
- Model size
- Training time / resource usage when available

### Statistical rigor
- Per-fold results
- Mean ± standard deviation
- Bootstrap 95% confidence intervals
- Paired significance tests (e.g., Wilcoxon signed-rank test)

---

## Key results (high-level)

From the reported experimental figures:
- **H2O AutoML** achieves the strongest mean accuracy (≈ **97%** range reported in the discussion), with **XGBoost (poly2 ablation)** very close behind.
- **LightGBM** and **CatBoost** form a strong and consistent next tier.
- Lower-ranked frameworks/baselines show modest but consistent gaps across Accuracy/Precision/Recall/F1.

---

## Explainability

Explainability is integrated as a standard output:
- **SHAP** global summary plots (feature importance across models)
- Optional **local explanations** for individual predictions
- Outputs stored for easy reporting and comparison

---

## Deployment readiness (as demonstrated)

The pipeline demonstrates real deployment outputs:
- Model packaging: `.pkl`, `.json`, `.yaml`
- API serving: **FastAPI**
- Dashboard: **Streamlit leaderboard**
- Optional containerization for portability (Docker/Compose)

---

## Repository layout

```
.
├── paper/
│   └── AutoML - CIS_Final.docx   # Full research paper
├── Project/                       # Implementation code
│   ├── trainers/                  # Model training scripts
│   ├── experiments/               # CV runner, preprocessing
│   ├── analysis/                  # Statistical analysis, SHAP
│   └── utils/                     # Shared utilities
├── scripts/                       # CLI entrypoints
│   ├── run_all.py                 # Master orchestrator
│   ├── run_guardrails.py          # Leakage detection
│   └── run_boosting_suite.py      # Boosting experiments
├── Deploy/                        # Deployment artifacts
│   ├── api/                       # FastAPI service
│   └── docker-compose.yml         # Container orchestration
├── figures/                       # Exported figures from paper
├── reports/                       # CSV result summaries
├── notebooks/                     # Jupyter tutorials
└── README.md
```

---

## How to use this repository (for reviewers)

1. **Read the paper:** `paper/AutoML - CIS_Final.docx`
2. **Key sections to focus on:**
   - **Introduction** (motivation: leakage + reproducibility)
   - **Methods** (guardrails + fold-safe preprocessing + fairness rules)
   - **Experimental Results** (framework comparison + figures)
   - **Conclusion** (trustworthy AutoML benchmark design)

3. **Run the implementation (optional):**
   ```bash
   pip install -r Project/requirements_min.txt
   python scripts/run_all.py --max-datasets 3
   streamlit run Project/streamlit_leaderboard.py
   ```

---

## Authorship statement

This research paper (writing, methodology, design, and analysis) was completed independently by **Pratik Niroula**.

---

## Citation

If you reference this work:

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
