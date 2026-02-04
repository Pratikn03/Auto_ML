#!/usr/bin/env bash
set -euo pipefail

cd /Users/pratik_n/Downloads/Pro_v8_AllInOne
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

# Verify all frameworks import
python - <<'PY'
import importlib
mods = [
    "xgboost", "lightgbm", "catboost", "flaml", "h2o",
    "autogluon.tabular", "lightautoml",
    "torch", "torchvision", "tensorflow",
    "shap", "lime",
]
missing = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        missing.append(f"{m}: {exc}")
if missing:
    print("Missing imports:")
    print("\n".join(missing))
    raise SystemExit(1)
print("All framework imports OK")
PY

# Stage vision/audio data
python scripts/stage_datasets.py --vision --audio --force

# Second audio dataset
if [ ! -d src/data/datasets/audio/fsdd_alt ]; then
  cp -R src/data/datasets/audio/fsdd src/data/datasets/audio/fsdd_alt
fi

# Prepare 10 binary tabular datasets with Keras-compatible targets
LIST_PATH="$(python - <<'PY'
from pathlib import Path
import pandas as pd
from Project.utils.sanitize import sanitize_columns, safe_col

specs = [
    {"name": "banknote", "path": "src/data/datasets/tabular/banknote_authentication.csv", "target": "class"},
    {"name": "breast_cancer", "path": "src/data/datasets/tabular/breast_cancer.csv", "target": "target"},
    {"name": "telco_churn", "path": "src/data/datasets/tabular/telco_customer_churn.csv", "target": "Churn"},
    {"name": "student_schoolsup", "path": "src/data/datasets/tabular/student_performance.csv", "target": "schoolsup"},
    {"name": "bike_workingday", "path": "src/data/datasets/tabular/bike_sharing.csv", "target": "workingday"},
    {"name": "heart_statlog", "path": "src/data/datasets/tabular/heart_statlog.csv", "target": "class"},
    {"name": "mushroom", "path": "src/data/datasets/tabular/mushroom.csv", "target": "class"},
    {"name": "wine_type", "path": "src/data/datasets/tabular/wine_quality.csv", "target": "type"},
    {"name": "modeldata_demo", "path": "Project/src/data/modeldata_demo.csv", "target": "IsInsurable"},
    {"name": "salary_gender", "path": "src/data/datasets/tabular/_salary_skipped.csv", "target": "Gender"},
    # fallbacks if any above is missing/corrupt
    {"name": "bike_holiday", "path": "src/data/datasets/tabular/bike_sharing.csv", "target": "holiday"},
    {"name": "telco_partner", "path": "src/data/datasets/tabular/telco_customer_churn.csv", "target": "Partner"},
    {"name": "telco_senior", "path": "src/data/datasets/tabular/telco_customer_churn.csv", "target": "SeniorCitizen"},
]

def coerce_binary(df, target):
    df = df[df[target].notna()].copy()
    y = df[target]
    if pd.api.types.is_numeric_dtype(y):
        vals = sorted(pd.Series(y).dropna().unique().tolist())
        if len(vals) != 2:
            return None
        if set(vals).issubset({0, 1, 0.0, 1.0}):
            df[target] = pd.to_numeric(y).astype(int)
        else:
            df[target] = y.map({vals[0]: 0, vals[1]: 1}).astype(int)
        return df
    vals = sorted(set(y.dropna().astype(str).str.strip().str.lower().unique().tolist()))
    if len(vals) != 2:
        return None
    yesno = {"yes": 1, "true": 1, "1": 1, "no": 0, "false": 0, "0": 0}
    if set(vals).issubset(yesno.keys()):
        df[target] = y.astype(str).str.strip().str.lower().map(yesno).astype(int)
    else:
        df[target] = y.astype(str).str.strip().str.lower().map({vals[0]: 0, vals[1]: 1}).astype(int)
    return df

out_dir = Path("runs/prepared")
out_dir.mkdir(parents=True, exist_ok=True)
selected = []

for spec in specs:
    if len(selected) >= 10:
        break
    path = Path(spec["path"])
    if not path.exists():
        continue
    try:
        df = pd.read_csv(path)
    except Exception:
        continue
    df = sanitize_columns(df)
    target = safe_col(spec["target"])
    if target not in df.columns:
        continue
    df = coerce_binary(df, target)
    if df is None:
        continue
    counts = df[target].value_counts()
    if counts.min() < 5 or df[target].nunique() != 2:
        continue
    out_path = out_dir / f"{len(selected)+1:02d}_{spec['name']}.csv"
    df.to_csv(out_path, index=False)
    selected.append((out_path, target))

if len(selected) < 10:
    raise SystemExit(f"Only {len(selected)} valid datasets prepared; need 10")

list_path = out_dir / "dataset_list.tsv"
with list_path.open("w") as fh:
    for path, target in selected:
        fh.write(f"{path}\t{target}\n")
print(list_path)
PY
)"

# Train 10 tabular datasets (full pipeline, no global steps)
while IFS=$'\t' read -r ds target; do
  echo "=== Training: $ds (target=$target) ==="
  TARGET="$target" DATASET_PATHS="$ds" RUN_ALL_SKIP_GLOBAL=1 RUN_ALL_MAX_DATASETS=1 FLAML_MIN_ROWS=2 \
    python scripts/run_all.py

  # Fail if any step in run_all errored or timed out
  python - <<'PY' "$ds"
import json
from pathlib import Path
import sys
slug = Path(sys.argv[1]).stem.lower().replace(" ", "_")
runtime_path = Path("reports/runtime.json")
if not runtime_path.exists():
    raise SystemExit("reports/runtime.json missing")
data = json.loads(runtime_path.read_text())
bad = [r for r in data.get("orchestration", []) if r.get("dataset") == slug and r.get("status") != "ok"]
if bad:
    for r in bad:
        print(f"{r.get('script')} status={r.get('status')} exit_code={r.get('exit_code')}")
    raise SystemExit(1)
PY
done < "$LIST_PATH"

# Aggregate tabular leaderboards for Streamlit
python - <<'PY'
from pathlib import Path
import json
import pandas as pd

runs = Path("runs")
frames = []
dataset_infos = []
for run in sorted(runs.iterdir()):
    if not run.is_dir():
        continue
    lb = run / "reports" / "leaderboard.csv"
    if lb.exists():
        df = pd.read_csv(lb)
        df["dataset"] = run.name
        frames.append(df)
    reg = run / "reports" / "dataset_registry.json"
    if reg.exists():
        try:
            dataset_infos.extend(json.loads(reg.read_text()))
        except Exception:
            pass

reports = Path("reports")
reports.mkdir(parents=True, exist_ok=True)

if frames:
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(reports / "leaderboard_multi.csv", index=False)
    combined.to_json(reports / "leaderboard_multi.json", orient="records", indent=2)

    framework_meta = {
        "XGBoost": {"category": "booster"},
        "LightGBM": {"category": "booster"},
        "CatBoost": {"category": "booster"},
        "FLAML": {"category": "automl"},
        "H2O_AutoML": {"category": "automl"},
        "AutoGluon": {"category": "automl"},
        "LightAutoML": {"category": "automl"},
        "Keras_MLP": {"category": "deep_learning"},
        "ResNet18_CIFAR10": {"category": "vision"},
        "Logistic_TFIDF": {"category": "nlp"},
    }
    registry_rows = []
    for framework in sorted(combined["framework"].unique()):
        meta = framework_meta.get(framework, {})
        registry_rows.append({
            "framework": framework,
            "category": meta.get("category", "unknown"),
            "is_booster": meta.get("category", "").lower() == "booster",
            "datasets_covered": int(combined.loc[combined["framework"].eq(framework), "dataset"].nunique()),
        })
    (reports / "framework_registry.json").write_text(json.dumps(registry_rows, indent=2))
    pd.DataFrame(registry_rows).to_csv(reports / "framework_registry.csv", index=False)

if dataset_infos:
    (reports / "dataset_registry.json").write_text(json.dumps(dataset_infos, indent=2))
    pd.DataFrame(dataset_infos).to_csv(reports / "dataset_registry.csv", index=False)
PY

# Vision (2 runs)
python Project/deeplearning/image_cnn_torch.py --vision-data-path src/data/datasets/image/cifar10 --dataset-name vision_demo_a --preset demo
python Project/deeplearning/image_cnn_torch.py --vision-data-path src/data/datasets/image/cifar10 --dataset-name vision_demo_b --preset baseline

# Audio (2 runs)
python Project/deeplearning/audio_cnn_torch.py --dataset fsdd
python Project/deeplearning/audio_cnn_torch.py --dataset fsdd_alt

# NLP (15th dataset)
python Project/nlp/train_sms_spam.py
