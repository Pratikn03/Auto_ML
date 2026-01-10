# Sensitivity Analysis

The new `scripts/run_sensitivity_analysis.py` script rounds up a synthetic binary classification dataset (12 numeric features, 800 rows, 50% positive) and re-runs the XGBoost feature-ablation variants while injecting controlled perturbations so you can see how the pipeline reacts when (a) features get noisy, (b) 20% of the numeric cells go missing, or (c) the positive class is under-represented (downsampled to a 20% positive ratio). Each run still reuses the same seeds/splits, records the same training metrics, and writes a condensed summary to `reports/sensitivity/sensitivity_summary.csv`.

## Variant definitions

| Variant | Preprocessing change |
| --- | --- |
| `baseline` | Median/most-frequent imputation + scaling, no engineered transforms. |
| `poly2` | Adds 2nd-degree polynomial interactions on the numeric branch. |
| `quantile_bins` | Inserts `KBinsDiscretizer(n_bins=10, encode='onehot-dense')` followed by a denseifier to avoid sparse output. |
| `vif_filter` | Drops numeric features whose variance inflation factor exceeds **10**, keeping the guardrail at the conventional “VIF≥10” threshold; this lets linear/logistic candidates stay stable while trees remain unchanged.

> The script configures `PreprocessingConfig(use_pandas_output=False)` so the denseifier can safely convert the discretized buckets, and it stores the final records in `reports/sensitivity/sensitivity_summary.csv` (four scenarios × four variants × accuracy/F1 metrics).

## Accuracy under each scenario

| Scenario | Baseline | Polynomial | Quantile bins | VIF filter |
| --- | --- | --- | --- | --- |
| clean | **0.9225 ±0.0243** | 0.9088 ±0.0184 | 0.8675 ±0.0303 | 0.9225 ±0.0243 |
| feature noise (Gaussian 0.5×std) | 0.8425 ±0.0199 | **0.8500 ±0.0261** | 0.8137 ±0.0195 | 0.8425 ±0.0199 |
| missingness (20% of numerics masked) | 0.8400 ±0.0117 | **0.8425 ±0.0185** | 0.8175 ±0.0210 | 0.8400 ±0.0117 |
| class imbalance (positives = 20%) | **0.9162 ±0.0208** | 0.9062 ±0.0285 | 0.8662 ±0.0315 | 0.9162 ±0.0208 |

Table values correspond to the `accuracy` rows in `reports/sensitivity/sensitivity_summary.csv`; standard deviations reflect the 3-fold CV repeats in the synthetic run. The VIF filter preserves the baseline score in every scenario, showing it only removes collinear numerics and does not penalize the signal even when noise/missingness is introduced. Polynomial expansions slightly boost robustness against noise and missingness, whereas quantile binning consistently lowers accuracy because it oversmooths the signal in this synthetic dataset.

## Reproducing the analysis

```bash
PYTHONPATH=. python scripts/run_sensitivity_analysis.py
```

The script reuses `run_feature_ablation_suite` so all supporting artifacts land under `reports/metrics` and `artifacts/experiments`; the condensed CSV at `reports/sensitivity/sensitivity_summary.csv` is safe to quote in reports. Update the scalars (noise scale, missing fraction, class ratio) by editing the scenario lambdas at the top of the script if you want to stress-test different levels.
