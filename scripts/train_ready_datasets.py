import os
import subprocess

DATASETS = [
    'airpassengers',
    'energy_efficiency',
    'sunspots',
    'banknote_authentication',
    'car_evaluation',
    'bike_sharing',
    'student_performance',
    'abalone',
    'mushroom',
    'wine_quality',
    'telco_customer_churn',
    'appliances_energy_prediction',
]

DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', 'src', 'data', 'datasets', 'tabular')

# Start with 30s, increase by 30s for each dataset
base_time = 30
for i, dataset in enumerate(DATASETS):
    csv_path = os.path.join(DATA_ROOT, f'{dataset}.csv')
    if not os.path.exists(csv_path):
        print(f"[SKIP] {dataset}: data file missing.")
        continue
    time_budget = base_time * (i + 1)
    print(f"[TRAIN] {dataset} (time budget: {time_budget}s)...")
    cmd = [
        'python', 'scripts/run_all.py',
        '--max-datasets', '1',
        '--flaml-time-budget', str(time_budget),
        '--prefer-small-datasets',
        '--skip-guardrails'
    ]
    # Set env var to force only this dataset
    env = os.environ.copy()
    env['RUN_ALL_DATASET_FILTER'] = dataset
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {dataset}: {e}")

print("Done training all ready datasets.")
