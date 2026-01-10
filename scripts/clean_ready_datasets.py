import os
import pandas as pd

# List of datasets to clean (same as in train_ready_datasets.py)
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

# Try common target column names
TARGET_CANDIDATES = [
    'target', 'label', 'y', 'class', 'output', 'Category', 'Survived', 'income', 'quality', 'Churn', 'Outcome', 'diagnosis', 'species', 'default.payment.next.month', 'is_fraud', 'Attrition', 'Exited', 'Class', 'Result', 'Response', 'SalePrice', 'price', 'score', 'target_column', 'Target', 'TARGET', 'labels', 'Label', 'LABEL', 'OutcomeType', 'Loan_Status', 'Loan_Status_Y', 'Loan_Status_N', 'Loan_Status_YN', 'Loan_Status_Binary', 'Loan_Status_Encoded', 'Loan_Status_Label', 'Loan_Status_OneHot', 'Loan_Status_Ordinal', 'Loan_Status_BinaryEncoded', 'Loan_Status_LabelEncoded', 'Loan_Status_OneHotEncoded', 'Loan_Status_OrdinalEncoded', 'Loan_Status_BinaryLabel', 'Loan_Status_LabelLabel', 'Loan_Status_OneHotLabel', 'Loan_Status_OrdinalLabel', 'Loan_Status_BinaryOneHot', 'Loan_Status_LabelOneHot', 'Loan_Status_OneHotOneHot', 'Loan_Status_OrdinalOneHot', 'Loan_Status_BinaryOrdinal', 'Loan_Status_LabelOrdinal', 'Loan_Status_OneHotOrdinal', 'Loan_Status_OrdinalOrdinal', 'Loan_Status_BinaryOrdinalLabel', 'Loan_Status_LabelOrdinalLabel', 'Loan_Status_OneHotOrdinalLabel', 'Loan_Status_OrdinalOrdinalLabel', 'Loan_Status_BinaryOrdinalOneHot', 'Loan_Status_LabelOrdinalOneHot', 'Loan_Status_OneHotOrdinalOneHot', 'Loan_Status_OrdinalOrdinalOneHot', 'Loan_Status_BinaryOrdinalOrdinal', 'Loan_Status_LabelOrdinalOrdinal', 'Loan_Status_OneHotOrdinalOrdinal', 'Loan_Status_OrdinalOrdinalOrdinal', 'Loan_Status_BinaryOrdinalOrdinalLabel', 'Loan_Status_LabelOrdinalOrdinalLabel', 'Loan_Status_OneHotOrdinalOrdinalLabel', 'Loan_Status_OrdinalOrdinalOrdinalLabel', 'Loan_Status_BinaryOrdinalOrdinalOneHot', 'Loan_Status_LabelOrdinalOrdinalOneHot', 'Loan_Status_OneHotOrdinalOrdinalOneHot', 'Loan_Status_OrdinalOrdinalOrdinalOneHot', 'Loan_Status_BinaryOrdinalOrdinalOrdinal', 'Loan_Status_LabelOrdinalOrdinalOrdinal', 'Loan_Status_OneHotOrdinalOrdinalOrdinal', 'Loan_Status_OrdinalOrdinalOrdinalOrdinal', 'Loan_Status_BinaryOrdinalOrdinalOrdinalLabel', 'Loan_Status_LabelOrdinalOrdinalOrdinalLabel', 'Loan_Status_OneHotOrdinalOrdinalOrdinalLabel', 'Loan_Status_OrdinalOrdinalOrdinalOrdinalLabel', 'Loan_Status_BinaryOrdinalOrdinalOrdinalOneHot', 'Loan_Status_LabelOrdinalOrdinalOrdinalOneHot', 'Loan_Status_OneHotOrdinalOrdinalOrdinalOneHot', 'Loan_Status_OrdinalOrdinalOrdinalOrdinalOneHot', 'Loan_Status_BinaryOrdinalOrdinalOrdinalOrdinal', 'Loan_Status_LabelOrdinalOrdinalOrdinalOrdinal', 'Loan_Status_OneHotOrdinalOrdinalOrdinalOrdinal', 'Loan_Status_OrdinalOrdinalOrdinalOrdinalOrdinal', 'Loan_Status_BinaryOrdinalOrdinalOrdinalOrdinalLabel', 'Loan_Status_LabelOrdinalOrdinalOrdinalOrdinalLabel', 'Loan_Status_OneHotOrdinalOrdinalOrdinalOrdinalLabel', 'Loan_Status_OrdinalOrdinalOrdinalOrdinalOrdinalLabel', 'Loan_Status_BinaryOrdinalOrdinalOrdinalOrdinalOneHot', 'Loan_Status_LabelOrdinalOrdinalOrdinalOrdinalOneHot', 'Loan_Status_OneHotOrdinalOrdinalOrdinalOrdinalOneHot', 'Loan_Status_OrdinalOrdinalOrdinalOrdinalOrdinalOneHot', 'Loan_Status_BinaryOrdinalOrdinalOrdinalOrdinalOrdinal', 'Loan_Status_LabelOrdinalOrdinalOrdinalOrdinalOrdinal', 'Loan_Status_OneHotOrdinalOrdinalOrdinalOrdinalOrdinal', 'Loan_Status_OrdinalOrdinalOrdinalOrdinalOrdinalOrdinal', 'Loan_Status_BinaryOrdinalOrdinalOrdinalOrdinalOrdinalLabel', 'Loan_Status_LabelOrdinalOrdinalOrdinalOrdinalOrdinalLabel', 'Loan_Status_OneHotOrdinalOrdinalOrdinalOrdinalOrdinalLabel', 'Loan_Status_OrdinalOrdinalOrdinalOrdinalOrdinalOrdinalLabel', 'Loan_Status_BinaryOrdinalOrdinalOrdinalOrdinalOrdinalOneHot', 'Loan_Status_LabelOrdinalOrdinalOrdinalOrdinalOrdinalOneHot', 'Loan_Status_OneHotOrdinalOrdinalOrdinalOrdinalOrdinalOneHot', 'Loan_Status_OrdinalOrdinalOrdinalOrdinalOrdinalOrdinalOneHot', 'Loan_Status_BinaryOrdinalOrdinalOrdinalOrdinalOrdinalOrdinal', 'Loan_Status_LabelOrdinalOrdinalOrdinalOrdinalOrdinalOrdinal', 'Loan_Status_OneHotOrdinalOrdinalOrdinalOrdinalOrdinalOrdinal', 'Loan_Status_OrdinalOrdinalOrdinalOrdinalOrdinalOrdinalOrdinal', 'Loan_Status_BinaryOrdinalOrdinalOrdinalOrdinalOrdinalOrdinalLabel', 'Loan_Status_LabelOrdinalOrdinalOrdinalOrdinalOrdinalOrdinalLabel', 'Loan_Status_OneHotOrdinalOrdinalOrdinalOrdinalOrdinalOrdinalLabel', 'Loan_Status_OrdinalOrdinalOrdinalOrdinalOrdinalOrdinalOrdinalLabel', 'Loan_Status_BinaryOrdinalOrdinalOrdinalOrdinalOrdinalOrdinalOneHot', 'Loan_Status_LabelOrdinalOrdinalOrdinalOrdinalOrdinalOrdinalOneHot', 'Loan_Status_OneHotOrdinalOrdinalOrdinalOrdinalOrdinalOrdinalOneHot', 'Loan_Status_OrdinalOrdinalOrdinalOrdinalOrdinalOrdinalOrdinalOneHot', 'Loan_Status_BinaryOrdinalOrdinalOrdinalOrdinalOrdinalOrdinalOrdinal', 'Loan_Status_LabelOrdinalOrdinalOrdinalOrdinalOrdinalOrdinalOrdinal', 'Loan_Status_OneHotOrdinalOrdinalOrdinalOrdinalOrdinalOrdinalOrdinal', 'Loan_Status_OrdinalOrdinalOrdinalOrdinalOrdinalOrdinalOrdinalOrdinal']

def find_target_column(df):
    for col in df.columns:
        if col.lower() in [c.lower() for c in TARGET_CANDIDATES]:
            return col
    # fallback: last column
    return df.columns[-1]

def clean_dataset(dataset):
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
    else:
        print(f"[CLEAN] {dataset}: no missing target values found.")
    df_clean.to_csv(csv_path, index=False)

if __name__ == "__main__":
    for dataset in DATASETS:
        clean_dataset(dataset)
    print("Done cleaning all ready datasets.")
