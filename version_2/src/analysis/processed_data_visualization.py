"""
Processed Data Visualization and Analytics

Generates dataset diagnostics for the ICU processed data used by training:
- Distribution plots
- Temporal curves
- Missingness and correlation diagnostics
- Drift statistics between eICU and PhysioNet tensors
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp


FEATURES_24H = ["heartrate", "respiration", "sao2"]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_series(values: np.ndarray) -> np.ndarray:
    """Return finite values only for robust stats/plots."""
    values = np.asarray(values).reshape(-1)
    return values[np.isfinite(values)]


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    x = safe_series(x)
    y = safe_series(y)
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = np.sqrt(((len(x) - 1) * vx + (len(y) - 1) * vy) / (len(x) + len(y) - 2))
    if pooled == 0:
        return 0.0
    return float((np.mean(x) - np.mean(y)) / pooled)


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Compute PSI using quantile bins from reference distribution."""
    reference = safe_series(reference)
    current = safe_series(current)
    if len(reference) < 10 or len(current) < 10:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.quantile(reference, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    eps = 1e-8
    ref_pct = np.clip(ref_pct, eps, 1)
    cur_pct = np.clip(cur_pct, eps, 1)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def flatten_feature(arr: np.ndarray, feat_idx: int) -> np.ndarray:
    return arr[:, :, feat_idx].reshape(-1)


def basic_stats(values: np.ndarray) -> Dict[str, float]:
    values = safe_series(values)
    if len(values) == 0:
        return {"count": 0}
    q1, q3 = np.percentile(values, [25, 75])
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q1": float(q1),
        "q3": float(q3),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def add_ci_curve(ax, x: np.ndarray, y_matrix: np.ndarray, label: str, color: str) -> None:
    """Plot mean curve with 95% confidence interval across samples."""
    mean_curve = np.nanmean(y_matrix, axis=0)
    std_curve = np.nanstd(y_matrix, axis=0)
    n = np.maximum(np.sum(np.isfinite(y_matrix), axis=0), 1)
    ci95 = 1.96 * std_curve / np.sqrt(n)

    ax.plot(x, mean_curve, label=label, color=color, linewidth=2)
    ax.fill_between(x, mean_curve - ci95, mean_curve + ci95, color=color, alpha=0.2)


def generate_analysis(
    hourly_path: Path,
    outcomes_path: Path,
    x_eicu_path: Path,
    x_physio_path: Path,
    output_dir: Path,
) -> Dict:
    ensure_dir(output_dir)
    fig_dir = output_dir / "figures"
    ensure_dir(fig_dir)

    hourly = pd.read_csv(hourly_path)
    outcomes = pd.read_csv(outcomes_path)
    x_eicu = np.load(x_eicu_path)
    x_physio = np.load(x_physio_path)

    # Merge mortality labels to hourly records for stratified analysis.
    outcome_cols = ["patientunitstayid", "mortality"]
    outcomes_small = outcomes[outcome_cols].drop_duplicates("patientunitstayid")
    hourly = hourly.merge(outcomes_small, on="patientunitstayid", how="left")

    vital_cols = [c for c in FEATURES_24H if c in hourly.columns]
    all_numeric_cols = [c for c in hourly.columns if c not in ["patientunitstayid", "hour"]]

    results: Dict[str, Dict] = {
        "dataset_overview": {
            "hourly_rows": int(len(hourly)),
            "hourly_patients": int(hourly["patientunitstayid"].nunique()),
            "outcome_rows": int(len(outcomes)),
            "outcome_patients": int(outcomes["patientunitstayid"].nunique()),
            "mortality_rate": float(outcomes["mortality"].mean()),
            "x_eicu_shape": list(x_eicu.shape),
            "x_physio_shape": list(x_physio.shape),
        }
    }

    # Missingness summary.
    missing_rate = hourly[all_numeric_cols].isna().mean().sort_values(ascending=False)
    missing_df = pd.DataFrame({"feature": missing_rate.index, "missing_rate": missing_rate.values})
    missing_df.to_csv(output_dir / "missingness_summary.csv", index=False)
    results["missingness"] = {row.feature: float(row.missing_rate) for row in missing_df.itertuples()}

    # Descriptive stats for key vital signs.
    desc = []
    for col in vital_cols:
        overall = basic_stats(hourly[col].values)
        surv = basic_stats(hourly.loc[hourly["mortality"] == 0, col].values)
        nonsurv = basic_stats(hourly.loc[hourly["mortality"] == 1, col].values)
        desc.append({"feature": col, "group": "overall", **overall})
        desc.append({"feature": col, "group": "survivor", **surv})
        desc.append({"feature": col, "group": "non_survivor", **nonsurv})

    desc_df = pd.DataFrame(desc)
    desc_df.to_csv(output_dir / "descriptive_stats_vitals.csv", index=False)

    # Drift metrics between eICU and PhysioNet tensors.
    drift_rows: List[Dict] = []
    for i, feat in enumerate(FEATURES_24H):
        e = flatten_feature(x_eicu, i)
        p = flatten_feature(x_physio, i)

        e = safe_series(e)
        p = safe_series(p)

        # Downsample for speed while keeping stable estimates.
        max_n = 250000
        if len(e) > max_n:
            e = np.random.default_rng(42).choice(e, size=max_n, replace=False)
        if len(p) > max_n:
            p = np.random.default_rng(42).choice(p, size=max_n, replace=False)

        ks_stat, ks_p = ks_2samp(e, p)
        drift_rows.append(
            {
                "feature": feat,
                "eicu_mean": float(np.mean(e)),
                "physio_mean": float(np.mean(p)),
                "eicu_std": float(np.std(e)),
                "physio_std": float(np.std(p)),
                "cohen_d": cohen_d(e, p),
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_p),
                "psi": population_stability_index(e, p, bins=10),
            }
        )

    drift_df = pd.DataFrame(drift_rows)
    drift_df.to_csv(output_dir / "drift_metrics.csv", index=False)
    results["drift_metrics"] = drift_df.to_dict(orient="records")

    # --------------------------
    # Visualization section
    # --------------------------

    # 1. Mortality class distribution.
    fig, ax = plt.subplots(figsize=(6, 4))
    class_counts = outcomes["mortality"].value_counts().sort_index()
    labels = ["Survivor (0)", "Non-survivor (1)"]
    ax.bar(labels, class_counts.values, color=["#4C72B0", "#C44E52"])
    ax.set_title("Mortality Class Distribution")
    ax.set_ylabel("Patients")
    for i, v in enumerate(class_counts.values):
        ax.text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(fig_dir / "mortality_class_distribution.png", dpi=160)
    plt.close(fig)

    # 2. Missingness bar chart.
    fig, ax = plt.subplots(figsize=(10, 5))
    top_missing = missing_df.head(15)
    x_idx = np.arange(len(top_missing))
    ax.bar(x_idx, top_missing["missing_rate"] * 100, color="#DD8452")
    ax.set_title("Top 15 Features by Missingness")
    ax.set_ylabel("Missing Rate (%)")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(top_missing["feature"], rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(fig_dir / "missingness_top15.png", dpi=160)
    plt.close(fig)

    # 3. Feature distributions for core vitals by mortality.
    fig, axes = plt.subplots(1, len(vital_cols), figsize=(16, 4))
    if len(vital_cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, vital_cols):
        a = safe_series(hourly.loc[hourly["mortality"] == 0, col].values)
        b = safe_series(hourly.loc[hourly["mortality"] == 1, col].values)
        ax.hist(a, bins=50, alpha=0.5, density=True, label="Survivor", color="#4C72B0")
        ax.hist(b, bins=50, alpha=0.5, density=True, label="Non-survivor", color="#C44E52")
        ax.set_title(f"{col} distribution")
        ax.set_xlabel(col)
    axes[0].set_ylabel("Density")
    axes[0].legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "vital_distributions_by_mortality.png", dpi=160)
    plt.close(fig)

    # 4. Correlation heatmap for numeric features.
    corr_cols = [c for c in all_numeric_cols if hourly[c].dtype.kind in "fi"]
    corr_cols = corr_cols[:20]  # keep figure readable
    corr = hourly[corr_cols].corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(corr_cols)))
    ax.set_yticks(np.arange(len(corr_cols)))
    ax.set_xticklabels(corr_cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr_cols, fontsize=8)
    ax.set_title("Feature Correlation Heatmap (first 20 numeric features)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(fig_dir / "correlation_heatmap.png", dpi=160)
    plt.close(fig)

    # 5. Temporal 24h curves from tensor data.
    hours = np.arange(24)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharex=True)
    for i, feat in enumerate(FEATURES_24H):
        ax = axes[i]
        add_ci_curve(ax, hours, x_eicu[:, :, i], label="eICU", color="#4C72B0")
        add_ci_curve(ax, hours, x_physio[:, :, i], label="PhysioNet", color="#55A868")
        ax.set_title(f"24h Trend: {feat}")
        ax.set_xlabel("Hour")
        if i == 0:
            ax.set_ylabel("Normalized value")
            ax.legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "temporal_trends_24h_dataset_comparison.png", dpi=160)
    plt.close(fig)

    # 6. Dataset drift boxplots (flattened features).
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for i, feat in enumerate(FEATURES_24H):
        e = safe_series(flatten_feature(x_eicu, i))
        p = safe_series(flatten_feature(x_physio, i))
        # Subsample for plotting clarity.
        rng = np.random.default_rng(42)
        if len(e) > 30000:
            e = rng.choice(e, size=30000, replace=False)
        if len(p) > 30000:
            p = rng.choice(p, size=30000, replace=False)

        axes[i].boxplot([e, p], tick_labels=["eICU", "PhysioNet"], showfliers=False)
        axes[i].set_title(f"{feat}: dataset shift")
    plt.tight_layout()
    fig.savefig(fig_dir / "dataset_shift_boxplots.png", dpi=160)
    plt.close(fig)

    with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def main() -> None:
    base = Path(".")
    output_dir = base / "results" / "processed_data_analysis"

    summary = generate_analysis(
        hourly_path=base / "data" / "processed" / "eicu_hourly_all_features.csv",
        outcomes_path=base / "data" / "processed" / "eicu_outcomes.csv",
        x_eicu_path=base / "X_eicu_24h.npy",
        x_physio_path=base / "X_physio_24h.npy",
        output_dir=output_dir,
    )

    print("Analysis complete")
    print("Output directory:", output_dir)
    print("Mortality rate:", f"{summary['dataset_overview']['mortality_rate']:.2%}")


if __name__ == "__main__":
    main()
