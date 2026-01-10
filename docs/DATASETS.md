# Datasets Guide

This document describes the datasets used in the AutoMLP benchmark and how to add new datasets.

---

## Dataset Discovery

The orchestrator (`scripts/run_all.py`) automatically discovers datasets from:

1. **Primary paths** (in order):
   - `Project/src/data/*.csv`
   - `src/data/*.csv`
   - `Project/src/data/datasets/*.csv`
   - `src/data/datasets/tabular/*.csv`

2. **Environment override**:
   ```bash
   export DATASET_PATHS="/path/to/data1.csv:/path/to/data2.csv"
   ```

3. **CLI override**:
   ```bash
   python scripts/run_boosting_suite.py --data-path /path/to/custom.csv
   ```

---

## Included Datasets

### Tabular (Classification)

| Dataset | Rows | Features | Classes | Target Column | Source |
|---------|------|----------|---------|---------------|--------|
| `breast_cancer` | 569 | 30 | 2 | `diagnosis` | sklearn |
| `iris` | 150 | 4 | 3 | `species` | sklearn |
| `wine` | 178 | 13 | 3 | `class` | sklearn |
| `titanic` | 891 | 11 | 2 | `survived` | Kaggle |
| `heart_statlog` | 270 | 13 | 2 | `target` | UCI |
| `adult_income` | 48,842 | 14 | 2 | `income` | UCI |
| `bank_marketing` | 45,211 | 16 | 2 | `y` | UCI |
| `credit_german` | 1,000 | 20 | 2 | `class` | UCI |
| `mushroom` | 8,124 | 22 | 2 | `class` | UCI |
| `car_evaluation` | 1,728 | 6 | 4 | `class` | UCI |
| `modeldata_demo` | ~500 | 10+ | 2 | `IsInsurable` | Synthetic |

### Tabular (Regression)

| Dataset | Rows | Features | Target | Source |
|---------|------|----------|--------|--------|
| `california_housing` | 20,640 | 8 | `median_house_value` | sklearn |
| `diabetes_regression` | 442 | 10 | `target` | sklearn |
| `bike_sharing` | 17,379 | 16 | `cnt` | UCI |
| `energy_efficiency` | 768 | 8 | `Heating_Load` | UCI |
| `wine_quality` | 6,497 | 11 | `quality` | UCI |
| `abalone` | 4,177 | 8 | `Rings` | UCI |

### Time Series

| Dataset | Records | Frequency | Target | Notes |
|---------|---------|-----------|--------|-------|
| `airpassengers` | 144 | Monthly | `Passengers` | Classic benchmark |
| `sunspots` | 2,820 | Monthly | `Sunspots` | Long history |
| `electricity_load` | 370,000+ | Hourly | `MT_001` | Multi-client |
| `household_power` | 2M+ | Minute | `Global_active_power` | High frequency |

### Vision

| Dataset | Images | Classes | Resolution | Source |
|---------|--------|---------|------------|--------|
| `cifar10` | 60,000 | 10 | 32×32 | torchvision |
| `caltech101` | 9,146 | 101 | Variable | torchvision |
| `fashion_mnist` | 70,000 | 10 | 28×28 | torchvision |
| `stl10` | 13,000 | 10 | 96×96 | torchvision |
| `oxford_flowers102` | 8,189 | 102 | Variable | torchvision |
| `food101` | 101,000 | 101 | Variable | torchvision |

### Audio

| Dataset | Samples | Classes | Duration | Source |
|---------|---------|---------|----------|--------|
| `speech_commands_v2` | 105,829 | 35 | 1s | Google |
| `esc50` | 2,000 | 50 | 5s | GitHub |
| `urbansound8k` | 8,732 | 10 | ≤4s | NYU |
| `gtzan` | 1,000 | 10 | 30s | Music genre |
| `crema_d` | 7,442 | 6 | Variable | Emotion |
| `ravdess` | 7,356 | 8 | Variable | Emotion |
| `fsdd` | 3,000 | 10 | Variable | Digits |

---

## Dataset Format Requirements

### Tabular CSV

```
feature1,feature2,feature3,target
1.5,categorical_A,100,0
2.3,categorical_B,200,1
...
```

**Requirements:**
- UTF-8 encoding
- Header row with column names
- Target column: named `target`, `class`, `label`, or last column with ≤50 unique values
- Missing values: empty cells or `NA`/`NaN`

### Target Column Detection

The system auto-infers the target column by checking (in order):
1. Environment variable `TARGET`
2. Columns named: `target`, `class`, `label`, `variety`, `species`, `diagnosis`, `survived`, `salary`
3. Last column if it has ≤50 unique values (classification) or is numeric (regression)

Override with:
```bash
TARGET=my_target_column python scripts/run_all.py
```

---

## Adding New Datasets

### Method 1: Drop CSV in Data Directory

```bash
# Copy your dataset
cp /path/to/my_dataset.csv Project/src/data/

# Run pipeline (will auto-discover)
python scripts/run_all.py
```

### Method 2: Use Environment Variable

```bash
export CSV_PATH=/path/to/custom_dataset.csv
export TARGET=my_target_column
python scripts/run_boosting_suite.py
```

### Method 3: CLI Argument

```bash
python scripts/run_boosting_suite.py \
  --data-path /path/to/dataset.csv \
  --target my_target_column
```

### Method 4: Stage Script

For datasets requiring download/preprocessing:

```bash
# Download and stage datasets
python scripts/stage_datasets.py

# Or download specific datasets
python scripts/download_datasets.py --datasets cifar10 esc50
```

---

## Dataset Validation

### Guardrails Audit

Before training, run the guardrails checker:

```bash
python scripts/run_guardrails.py --data-root Project/src/data
```

**Checks performed:**
- ✓ Duplicate rows (leakage across CV folds)
- ✓ High-cardinality proxy columns (deterministic mappings to target)
- ✓ Path/token leakage (target class names in file paths)
- ✓ Temporal columns (potential look-ahead leakage)

Output: `reports/guardrails/{dataset}.json`

### Schema Validation

```python
import pandas as pd
from Project.utils.io import load_dataset, guess_target_column
from Project.utils.sanitize import sanitize_columns

# Load and validate
df = load_dataset(path="Project/src/data/my_dataset.csv")
df = sanitize_columns(df)
target = guess_target_column(df)
print(f"Target: {target}, Shape: {df.shape}")
```

---

## Dataset Registry

After running the pipeline, a registry is created:

**`reports/dataset_registry.json`**:
```json
[
  {
    "name": "breast_cancer",
    "rows": 569,
    "columns": 31,
    "target": "diagnosis",
    "task_type": "classification",
    "n_classes": 2,
    "path": "Project/src/data/breast_cancer.csv"
  },
  ...
]
```

**`reports/dataset_registry.csv`**: Same data in tabular format for easy viewing.

---

## Data Splits

### Default Split Strategy

| Task | Splitter | Parameters |
|------|----------|------------|
| Classification | `StratifiedKFold` | `n_splits=5, shuffle=True, random_state=seed` |
| Regression | `KFold` | `n_splits=5, shuffle=True, random_state=seed` |

### Multi-Seed Evaluation

Default seeds: `[42, 77]`

Total evaluations per model: `n_seeds × n_folds = 2 × 5 = 10`

### Holdout for Quick Testing

For rapid iteration (not for final results):

```bash
# Single 80/20 split
python Project/trainers/train_flaml.py  # Uses train_test_split internally
```

---

## Memory Considerations

| Dataset Size | Recommendation |
|--------------|----------------|
| < 10K rows | Full pipeline, all frameworks |
| 10K–100K rows | Enable `LOW_MEMORY_MODE=1` |
| 100K–1M rows | Subsample or use single framework |
| > 1M rows | Use chunked loading or sampling |

```bash
# Enable memory optimization
LOW_MEMORY_MODE=1 python scripts/run_all.py --max-datasets 3
```

---

## Vision Dataset Setup

Vision datasets require download:

```bash
# Download CIFAR-10
python -c "from torchvision.datasets import CIFAR10; CIFAR10('data', download=True)"

# Run vision pipeline
python Project/deeplearning/image_cnn_torch.py
```

Outputs:
- `reports/leaderboard_vision.csv`
- `reports/vision_metrics.json`

---

## Audio Dataset Setup

```bash
# Extract audio features (requires librosa)
pip install librosa soundfile

# Stage audio datasets
python scripts/extract_audio_features.py

# Run audio pipeline
python Project/deeplearning/audio_cnn_torch.py
```

Outputs:
- `reports/leaderboard_audio.csv`
- Pre-computed features in `data/features/`

---

## Dataset Sources

| Source | URL |
|--------|-----|
| UCI ML Repository | https://archive.ics.uci.edu/ml |
| Kaggle | https://www.kaggle.com/datasets |
| sklearn.datasets | https://scikit-learn.org/stable/datasets |
| torchvision.datasets | https://pytorch.org/vision/stable/datasets.html |
| torchaudio.datasets | https://pytorch.org/audio/stable/datasets.html |
| OpenML | https://www.openml.org |

---

## Licensing

Datasets have various licenses. Key ones:

| Dataset | License |
|---------|---------|
| sklearn datasets | BSD-3 |
| UCI datasets | CC BY 4.0 |
| CIFAR-10 | MIT |
| ImageNet derivatives | Research only |
| Speech Commands | CC BY 4.0 |

Always verify licensing before commercial use.
