import os
import subprocess

RUNS_DIR = os.path.join(os.path.dirname(__file__), '..', 'runs')
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'datasets', 'tabular')

# 1. Find all dataset slugs in runs/
all_datasets = [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))]

# 2. Identify which are already trained (COMPLETE = has both artifacts/ and reports/)
trained = set()
for d in all_datasets:
    dpath = os.path.join(RUNS_DIR, d)
    if os.path.isdir(os.path.join(dpath, 'artifacts')) and os.path.isdir(os.path.join(dpath, 'reports')):
        trained.add(d)

# 3. Find all tabular CSVs
csvs = [f for f in os.listdir(DATA_ROOT) if f.endswith('.csv')]
slug_from_csv = lambda f: os.path.splitext(f)[0]

# 4. For each CSV, if not trained, run pipeline
for csv in csvs:
    slug = slug_from_csv(csv)
    if slug in trained:
        print(f"[SKIP] {slug} already trained.")
        continue
    csv_path = os.path.join(DATA_ROOT, csv)
    print(f"[TRAIN] {slug} ...")
    # Call run_all.py for this dataset only
    cmd = [
        'python', 'scripts/run_all.py',
        '--max-datasets', '1',
        '--prefer-small-datasets',
        '--skip-guardrails'
    ]
    # Set env var to force only this dataset
    env = os.environ.copy()
    env['RUN_ALL_DATASET_FILTER'] = slug
    subprocess.run(cmd, env=env, check=False)

print("Done. Only missing datasets were trained.")
