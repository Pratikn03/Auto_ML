
import os
import time
import pandas as pd
import subprocess
from pathlib import Path

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(BASE_DIR, 'src', 'data', 'datasets', 'tabular')
RUNS_ROOT = os.path.join(BASE_DIR, 'runs')

# Try common target column names
TARGET_CANDIDATES = [
    'target', 'label', 'y', 'class', 'output', 'category', 'survived', 'income', 'quality', 'churn',
    'outcome', 'diagnosis', 'species', 'default.payment.next.month', 'is_fraud', 'attrition', 'exited',
]

BOOSTERS = {'XGBoost', 'LightGBM', 'CatBoost'}


def find_target_column(df: pd.DataFrame) -> str:
    for col in df.columns:
        if col.lower() in {c.lower() for c in TARGET_CANDIDATES}:
            return col
    return df.columns[-1]


def get_smallest_csv_datasets(n: int = 5):
    files = []
    if not os.path.isdir(DATA_ROOT):
        return []
    for fname in os.listdir(DATA_ROOT):
        if fname.lower().endswith('.csv'):
            fpath = os.path.join(DATA_ROOT, fname)
            try:
                size = os.path.getsize(fpath)
            except OSError:
                size = 0
            files.append((fname.replace('.csv', ''), size))
    files.sort(key=lambda x: x[1])
    return [f[0] for f in files[:n]]


def check_frameworks(dataset: str) -> bool:
    # Prefer the per-run copied leaderboard under runs/<dataset>/reports/leaderboard.csv
    lb_path = os.path.join(RUNS_ROOT, dataset, 'reports', 'leaderboard.csv')
    if not os.path.exists(lb_path):
        # fallback to top-level reports (older behaviour)
        lb_path = os.path.join(BASE_DIR, 'reports', 'leaderboard.csv')
    if not os.path.exists(lb_path):
        print(f"[WARN] {dataset}: leaderboard.csv not found at expected locations.")
        return False
    try:
        df = pd.read_csv(lb_path)
    except Exception as e:
        print(f"[WARN] {dataset}: failed reading leaderboard.csv: {e}")
        return False
    # Some leaderboards don't include a dataset column; try to filter if present
    if 'dataset' in df.columns:
        df = df[df['dataset'].str.lower() == dataset.lower()]
    frameworks = set(df['framework'].dropna().astype(str).unique())
    boosters = frameworks & BOOSTERS
    if len(frameworks) < 3 or not boosters:
        print(f"[FAIL] {dataset}: frameworks found={len(frameworks)} ({', '.join(sorted(frameworks)) if frameworks else 'none'}); boosters: {', '.join(sorted(boosters)) if boosters else 'None'}")
        return False
    print(f"[OK] {dataset}: frameworks: {', '.join(sorted(frameworks))} (boosters: {', '.join(sorted(boosters))})")
    return True


def run_command(cmd, env=None, cwd=None, desc=None):
    if desc:
        print(f"[RUN] {desc}: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, env=env, cwd=cwd or BASE_DIR, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] command failed: {e}")
        return False


def clean_and_train(dataset: str):
    csv_path = os.path.join(DATA_ROOT, f'{dataset}.csv')
    if not os.path.exists(csv_path):
        print(f"[SKIP] {dataset}: data file missing.")
        return
    df = pd.read_csv(csv_path)
    target_col = find_target_column(df)
    before = len(df)
    df_clean = df[df[target_col].notnull()].copy()
    after = len(df_clean)
    if after < before:
        print(f"[CLEAN] {dataset}: removed {before - after} rows with missing target '{target_col}'.")
        df_clean.to_csv(csv_path, index=False)
    else:
        print(f"[CLEAN] {dataset}: no missing target values found (target='{target_col}').")

    # Run full pipeline for this dataset. Request common AutoML frameworks explicitly so
    # run_automl_suite receives a useful default set. Do NOT skip guardrails.
    time_budget = 90
    cmd = [
        'python', 'scripts/run_all.py',
        '--max-datasets', '1',
        '--flaml-time-budget', str(time_budget),
        '--prefer-small-datasets',
        '--frameworks', 'flaml', 'h2o', 'autogluon'
    ]
    env = os.environ.copy()
    env['RUN_ALL_DATASET_FILTER'] = dataset
    # Give the process some timeouts internally; run and then validate outputs.
    print(f"[TRAIN] Starting run_all for dataset={dataset} (time_budget={time_budget}s)")
    ok = run_command(cmd, env=env, desc=f'run_all ({dataset})')
    if not ok:
        print(f"[WARN] initial run_all failed for {dataset}; will attempt a boosters-only run next.")

    # Validate frameworks presence; if insufficient, attempt a boosters-only run and re-check.
    if not check_frameworks(dataset):
        print(f"[RETRY] Running boosting suite for {dataset} to ensure boosters are present.")
        boost_cmd = [
            'python', 'scripts/run_boosting_suite.py',
            '--data-path', os.path.join(DATA_ROOT, f'{dataset}.csv'),
            '--experiment-name', f'boosting_retry_{dataset}'
        ]
        run_command(boost_cmd, desc=f'run_boosting_suite ({dataset})')
        # small pause to allow reports copying
        time.sleep(2)
        if not check_frameworks(dataset):
            print(f"[FAIL] {dataset}: after retries still fewer than 3 frameworks or no boosters. See runs/{dataset}/reports for details.")
        else:
            print(f"[OK] {dataset}: boosters added after retry.")
    else:
        print(f"[OK] {dataset}: sufficient frameworks present.")


def run_vision_pipeline():
    print("[VISION] Running vision pipeline (image CNN)...")
    cmd = [
        'python', 'Project/deeplearning/image_cnn_torch.py'
    ]
    run_command(cmd, desc='image_cnn_torch')


if __name__ == "__main__":
    datasets = get_smallest_csv_datasets(5)
    if not datasets:
        print("No CSV datasets found under src/data/datasets/tabular")
    for dataset in datasets:
        clean_and_train(dataset)
    # Run vision pipeline after tabular runs
    run_vision_pipeline()
    print("Done: cleaned + attempted training 5 tabular datasets and ran vision pipeline.")
