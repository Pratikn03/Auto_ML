"""Run sensitivity experiments under noise, missingness, and imbalance perturbations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from Project.experiments.ablations import FeatureVariant, run_feature_ablation_suite
from Project.experiments.boosting import make_boosting_factory
from Project.experiments.preprocessing import PreprocessingConfig
from Project.experiments.runner import ExperimentConfig


TARGET_COL = "target"


def make_synthetic_dataset(
    *, n_samples: int, n_features: int, n_informative: int, random_state: int
) -> pd.DataFrame:
    """Create a small classification dataset to exercise sensitivity experiments."""

    x, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_clusters_per_class=2,
        flip_y=0.01,
        class_sep=1.0,
        random_state=random_state,
    )
    columns = [f"feature_{i}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=columns)
    df[TARGET_COL] = y
    return df


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [col for col in df.select_dtypes(include=np.number).columns if col != TARGET_COL]


def add_gaussian_noise(df: pd.DataFrame, rng: np.random.Generator, scale: float) -> pd.DataFrame:
    frame = df.copy()
    cols = _numeric_cols(frame)
    stds = frame[cols].std().replace(0, 1.0)
    noise = rng.standard_normal(size=frame[cols].shape) * (stds.values * scale)
    frame.loc[:, cols] += noise
    return frame


def add_missingness(df: pd.DataFrame, rng: np.random.Generator, fraction: float) -> pd.DataFrame:
    frame = df.copy()
    cols = _numeric_cols(frame)
    mask = rng.random(size=frame[cols].shape) < fraction
    frame.loc[:, cols] = frame.loc[:, cols].mask(mask)
    return frame


def apply_class_imbalance(
    df: pd.DataFrame, rng: np.random.Generator, positive_fraction: float
) -> pd.DataFrame:
    frame = df.copy()
    positives = frame[frame[TARGET_COL] == 1]
    negatives = frame[frame[TARGET_COL] == 0]
    if positives.empty or negatives.empty:
        return frame
    desired_neg = max(1, int(len(positives) * (1 / positive_fraction - 1)))
    desired_neg = min(desired_neg, len(negatives))
    neg_sample = negatives.sample(n=desired_neg, random_state=int(rng.integers(0, 2**31 - 1)))
    out = pd.concat([positives, neg_sample], ignore_index=True)
    shuffle_state = int(rng.integers(0, 2**31 - 1))
    return out.sample(frac=1.0, random_state=shuffle_state).reset_index(drop=True)


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    modifier: Callable[[pd.DataFrame, np.random.Generator], pd.DataFrame]


def build_variants() -> tuple[FeatureVariant, ...]:
    """Return the subset of feature variants relevant to the sensitivity analysis."""

    return (
        FeatureVariant(
            name="baseline",
            display_name="Baseline",
            description="Impute + scaling; no extra transforms.",
            preprocessing_overrides={},
            metadata={"feature_variant_key": "baseline"},
            is_baseline=True,
        ),
        FeatureVariant(
            name="poly2",
            display_name="Polynomial (deg=2)",
            description="Numeric polynomial features for interaction sensitivity.",
            preprocessing_overrides={"poly_degree": 2},
            metadata={"feature_variant_key": "poly2"},
        ),
        FeatureVariant(
            name="quantile_bins",
            display_name="Quantile Binning",
            description="Discretize numeric inputs into quantile buckets.",
            preprocessing_overrides={"binning_strategy": "quantile", "n_bins": 10, "bin_encode": "onehot"},
            metadata={"feature_variant_key": "quantile_bins"},
        ),
        FeatureVariant(
            name="vif_filter",
            display_name="VIF Filter",
            description="Drop columns with VIF>10 (numeric only).",
            preprocessing_overrides={"vif_threshold": 10.0},
            metadata={"feature_variant_key": "vif_filter"},
        ),
    )


def collect_metrics(payload: dict[str, dict[str, dict[str, object]]], scenario: Scenario, df: pd.DataFrame) -> Iterator[dict[str, object]]:
    """Yield metric records for the summary table."""

    for estimator, variants in payload.items():
        for variant_key, variant_payload in variants.items():
            summary = variant_payload.get("summary")
            if summary is None or summary.empty:
                continue
            for metric in ("accuracy", "f1_macro"):
                if "metric" in summary.columns:
                    row = summary[summary["metric"] == metric]
                    if row.empty:
                        continue
                    mean = float(row["mean"].iloc[0])
                    std = float(row["std"].iloc[0])
                else:
                    mean_col = f"{metric}_mean"
                    std_col = f"{metric}_std"
                    if mean_col not in summary.columns:
                        continue
                    mean = float(summary[mean_col].iloc[0])
                    std = float(summary[std_col].iloc[0])
                yield {
                    "scenario": scenario.name,
                    "variant": variant_key,
                    "estimator": estimator,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "dataset_rows": len(df),
                    "positive_ratio": float(df[TARGET_COL].mean()),
                    "description": scenario.description,
                }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sensitivity experiments for feature variants.")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/sensitivity"), help="Where to write the summary CSV.")
    parser.add_argument("--n-samples", type=int, default=800, help="Number of synthetic samples to generate per scenario.")
    parser.add_argument("--n-features", type=int, default=12, help="Total features to synthesize.")
    parser.add_argument("--n-informative", type=int, default=6, help="Number of informative features.")
    parser.add_argument("--splits", type=int, default=3, help="Cross-validation folds per run.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0], help="Seeds used by each experiment runner.")
    parser.add_argument("--random-seed", type=int, default=0, help="Seed for data perturbations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_seed)
    base_df = make_synthetic_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_informative=args.n_informative,
        random_state=args.random_seed,
    )

    scenarios: list[Scenario] = [
        Scenario(name="baseline", description="Original synthetic data", modifier=lambda df, _: df),
        Scenario(
            name="feature_noise",
            description="Gaussian noise added (scale=0.5×std).",
            modifier=lambda df, rng: add_gaussian_noise(df, rng, scale=0.5),
        ),
        Scenario(
            name="missingness",
            description="20% random entries masked on numeric columns.",
            modifier=lambda df, rng: add_missingness(df, rng, fraction=0.2),
        ),
        Scenario(
            name="imbalance",
            description="Downsample majority class to 20% positives.",
            modifier=lambda df, rng: apply_class_imbalance(df, rng, positive_fraction=0.2),
        ),
    ]

    variants = build_variants()
    estimators = {"xgboost": make_boosting_factory("xgboost", tuning_strategy=None)}
    preprocessing = PreprocessingConfig(use_pandas_output=False)
    base_config = ExperimentConfig(
        experiment_name="sensitivity",
        seeds=tuple(args.seeds),
        n_splits=args.splits,
        preprocessing=preprocessing,
        output_dir=Path("reports/metrics"),
        artifact_dir=Path("artifacts/experiments"),
        figure_dir=Path("figures/feature_importance"),
        metadata={"sensitivity_run": True},
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []

    for scenario in scenarios:
        df = scenario.modifier(base_df.copy(), rng)
        print(f"Running scenario {scenario.name} ({scenario.description}); rows={len(df)}; pos_ratio={df[TARGET_COL].mean():.2f}")
        results = run_feature_ablation_suite(
            base_config,
            estimators=estimators,
            variants=variants,
            df=df,
            target_override=TARGET_COL,
        )
        records.extend(collect_metrics(results, scenario, df))

    summary_df = pd.DataFrame(records)
    summary_path = output_dir / "sensitivity_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
