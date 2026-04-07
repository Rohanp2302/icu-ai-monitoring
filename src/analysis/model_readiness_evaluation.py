"""
Model readiness evaluation on processed ICU data.

Generates:
- ROC and PR curves
- Reliability (calibration) curves
- Risk score distributions
- Decision curve analysis (net benefit)
- Quantitative metrics table (AUC, AP, Brier, ECE, sensitivity, specificity)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """Compute ECE with equal-width bins."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi) if i < bins - 1 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece)


def net_benefit(y_true: np.ndarray, y_prob: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Decision-curve net benefit for treat-if-risk-above-threshold."""
    y_true = np.asarray(y_true).astype(int)
    nb = []
    n = len(y_true)

    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        w = t / (1 - t)
        nb.append((tp / n) - (fp / n) * w)

    return np.array(nb, dtype=np.float64)


def sensitivity_specificity(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Youden-J threshold from ROC curve."""
    fpr, tpr, thr = roc_curve(y_true, y_prob)
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(thr[idx])


def aggregate_patient_features(hourly: pd.DataFrame, outcomes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Create patient-level feature matrix from hourly records."""
    id_col = "patientunitstayid"
    target_col = "mortality"

    base_exclude = {id_col, "hour"}
    numeric_cols = [c for c in hourly.columns if c not in base_exclude and pd.api.types.is_numeric_dtype(hourly[c])]

    # Focus on clinically important + sufficiently observed features.
    missing_rates = hourly[numeric_cols].isna().mean()
    selected = [c for c in numeric_cols if missing_rates[c] <= 0.35]

    stats = ["mean", "std", "min", "max", "median"]
    agg = hourly.groupby(id_col)[selected].agg(stats)
    agg.columns = [f"{f}_{s}" for f, s in agg.columns]

    # Add temporal coverage feature.
    coverage = hourly.groupby(id_col).size().rename("hourly_rows")
    agg = agg.join(coverage, how="left")

    # Add static outcome-side features if present.
    add_cols = [c for c in ["age", "icu_los_days"] if c in outcomes.columns]
    outcomes_unique = outcomes[[id_col, target_col] + add_cols].drop_duplicates(id_col).copy()

    # eICU age can include strings like '> 89'; coerce to numeric for modeling.
    for c in add_cols:
        outcomes_unique[c] = pd.to_numeric(outcomes_unique[c], errors="coerce")

    df = agg.reset_index().merge(outcomes_unique, on=id_col, how="inner")
    y = df[target_col].astype(int).values
    feature_df = df.drop(columns=[id_col, target_col])
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce")
    X = feature_df.values
    feature_names = list(feature_df.columns)

    return X, y, feature_names


def evaluate_model(y_true: np.ndarray, y_prob: np.ndarray, name: str) -> Dict[str, float]:
    thr = find_best_threshold(y_true, y_prob)
    ss = sensitivity_specificity(y_true, y_prob, thr)

    return {
        "model": name,
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece_10": expected_calibration_error(y_true, y_prob, bins=10),
        "best_threshold": float(thr),
        "sensitivity_at_best_threshold": ss["sensitivity"],
        "specificity_at_best_threshold": ss["specificity"],
    }


def run(output_dir: Path) -> None:
    ensure_dir(output_dir)
    fig_dir = output_dir / "figures"
    ensure_dir(fig_dir)

    hourly = pd.read_csv("data/processed/eicu_hourly_all_features.csv")
    outcomes = pd.read_csv("data/processed/eicu_outcomes.csv")

    X, y, feature_names = aggregate_patient_features(hourly, outcomes)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    base_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    base_pipe.fit(X_train, y_train)
    p_base = base_pipe.predict_proba(X_test)[:, 1]

    sig_cal = CalibratedClassifierCV(base_pipe, cv=5, method="sigmoid")
    sig_cal.fit(X_train, y_train)
    p_sig = sig_cal.predict_proba(X_test)[:, 1]

    iso_cal = CalibratedClassifierCV(base_pipe, cv=5, method="isotonic")
    iso_cal.fit(X_train, y_train)
    p_iso = iso_cal.predict_proba(X_test)[:, 1]

    rows = [
        evaluate_model(y_test, p_base, "logreg_raw"),
        evaluate_model(y_test, p_sig, "logreg_sigmoid_calibrated"),
        evaluate_model(y_test, p_iso, "logreg_isotonic_calibrated"),
    ]
    metrics_df = pd.DataFrame(rows).sort_values("brier")
    metrics_df.to_csv(output_dir / "calibration_and_discrimination_metrics.csv", index=False)

    # 1) ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probs, color in [
        ("Raw", p_base, "#4C72B0"),
        ("Sigmoid Cal", p_sig, "#55A868"),
        ("Isotonic Cal", p_iso, "#C44E52"),
    ]:
        fpr, tpr, _ = roc_curve(y_test, probs)
        auc = roc_auc_score(y_test, probs)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "roc_curves_calibration_comparison.png", dpi=160)
    plt.close(fig)

    # 2) PR curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probs, color in [
        ("Raw", p_base, "#4C72B0"),
        ("Sigmoid Cal", p_sig, "#55A868"),
        ("Isotonic Cal", p_iso, "#C44E52"),
    ]:
        pr, rc, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        ax.plot(rc, pr, label=f"{name} (AP={ap:.3f})", color=color)
    prevalence = y_test.mean()
    ax.hlines(prevalence, 0, 1, colors="k", linestyles="--", label=f"Prevalence={prevalence:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "pr_curves_calibration_comparison.png", dpi=160)
    plt.close(fig)

    # 3) Reliability diagram
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, probs, color in [
        ("Raw", p_base, "#4C72B0"),
        ("Sigmoid Cal", p_sig, "#55A868"),
        ("Isotonic Cal", p_iso, "#C44E52"),
    ]:
        frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=name, color=color)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6)
    ax.set_xlabel("Mean Predicted Risk")
    ax.set_ylabel("Observed Event Rate")
    ax.set_title("Reliability Diagram")
    ax.legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "reliability_diagram.png", dpi=160)
    plt.close(fig)

    # 4) Risk score distributions
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(p_base[y_test == 0], bins=40, alpha=0.45, density=True, label="Survivor", color="#4C72B0")
    ax.hist(p_base[y_test == 1], bins=40, alpha=0.45, density=True, label="Non-survivor", color="#C44E52")
    ax.set_xlabel("Predicted Mortality Risk")
    ax.set_ylabel("Density")
    ax.set_title("Risk Score Distribution (Raw Logistic)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "risk_score_distribution_raw.png", dpi=160)
    plt.close(fig)

    # 5) Decision curve analysis
    thresholds = np.linspace(0.05, 0.8, 60)
    nb_raw = net_benefit(y_test, p_base, thresholds)
    nb_sig = net_benefit(y_test, p_sig, thresholds)
    nb_iso = net_benefit(y_test, p_iso, thresholds)

    prevalence = y_test.mean()
    nb_all = prevalence - (1 - prevalence) * (thresholds / (1 - thresholds))
    nb_none = np.zeros_like(thresholds)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, nb_raw, label="Raw", color="#4C72B0")
    ax.plot(thresholds, nb_sig, label="Sigmoid Cal", color="#55A868")
    ax.plot(thresholds, nb_iso, label="Isotonic Cal", color="#C44E52")
    ax.plot(thresholds, nb_all, "k--", label="Treat All", alpha=0.7)
    ax.plot(thresholds, nb_none, "k:", label="Treat None", alpha=0.8)
    ax.set_xlabel("Risk Threshold")
    ax.set_ylabel("Net Benefit")
    ax.set_title("Decision Curve Analysis")
    ax.legend()
    plt.tight_layout()
    fig.savefig(fig_dir / "decision_curve_analysis.png", dpi=160)
    plt.close(fig)

    summary = {
        "n_patients": int(len(y)),
        "event_rate": float(y.mean()),
        "n_features": int(X.shape[1]),
        "feature_preview": feature_names[:20],
        "best_brier_model": str(metrics_df.iloc[0]["model"]),
        "metrics": metrics_df.to_dict(orient="records"),
    }

    with open(output_dir / "model_readiness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Model readiness evaluation complete")
    print("Output:", output_dir)
    print("Best (by Brier):", summary["best_brier_model"])


if __name__ == "__main__":
    run(Path("results") / "model_readiness")
