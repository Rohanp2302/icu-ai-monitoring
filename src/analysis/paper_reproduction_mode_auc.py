"""
Paper reproduction mode vs strict unseen-patient mode for ICU mortality AUC.

Paper reproduction mode (diagnostic):
- Row-level random CV
- Uses all numeric columns including patient identifier
- Can produce inflated AUC due to patient overlap across folds

Strict mode (deployment-safe):
- Patient-level aggregated features
- Patient-level CV
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline


def build_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=500,
                    random_state=42,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def run_paper_mode(hourly: pd.DataFrame, outcomes: pd.DataFrame):
    merged = hourly.merge(
        outcomes[["patientunitstayid", "mortality"]],
        on="patientunitstayid",
        how="inner",
    )

    feature_cols = [
        c for c in merged.columns if c != "mortality" and pd.api.types.is_numeric_dtype(merged[c])
    ]

    X = merged[feature_cols].values
    y = merged["mortality"].astype(int).values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(build_model(), X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return feature_cols, auc


def run_strict_mode(hourly: pd.DataFrame, outcomes: pd.DataFrame):
    agg = hourly.groupby("patientunitstayid")[["heartrate", "respiration", "sao2"]].agg(
        ["mean", "std", "min", "max"]
    )
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]

    patient = agg.reset_index().merge(
        outcomes[["patientunitstayid", "mortality"]].drop_duplicates("patientunitstayid"),
        on="patientunitstayid",
        how="inner",
    )

    X = patient.drop(columns=["patientunitstayid", "mortality"]).values
    y = patient["mortality"].astype(int).values

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(build_model(), X, y, cv=cv, scoring="roc_auc", n_jobs=1)
    return list(patient.drop(columns=["patientunitstayid", "mortality"]).columns), auc


def main() -> None:
    out_dir = Path("results") / "paper_reproduction_mode"
    out_dir.mkdir(parents=True, exist_ok=True)

    hourly = pd.read_csv("data/processed/eicu_hourly_all_features.csv")
    outcomes = pd.read_csv("data/processed/eicu_outcomes.csv")

    paper_features, paper_auc = run_paper_mode(hourly, outcomes)
    strict_features, strict_auc = run_strict_mode(hourly, outcomes)

    summary = {
        "paper_reproduction_mode": {
            "auc_mean": float(np.mean(paper_auc)),
            "auc_std": float(np.std(paper_auc)),
            "fold_auc": [float(x) for x in paper_auc],
            "n_features": int(len(paper_features)),
        },
        "strict_unseen_patient_mode": {
            "auc_mean": float(np.mean(strict_auc)),
            "auc_std": float(np.std(strict_auc)),
            "fold_auc": [float(x) for x in strict_auc],
            "n_features": int(len(strict_features)),
        },
        "paper_mode_reaches_094": bool(float(np.mean(paper_auc)) >= 0.94),
    }

    with open(out_dir / "paper_mode_auc_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Paper Reproduction AUC",
        "",
        f"- Paper reproduction mode AUC: {summary['paper_reproduction_mode']['auc_mean']:.4f} +/- {summary['paper_reproduction_mode']['auc_std']:.4f}",
        f"- Strict unseen-patient mode AUC: {summary['strict_unseen_patient_mode']['auc_mean']:.4f} +/- {summary['strict_unseen_patient_mode']['auc_std']:.4f}",
        f"- Reaches >= 0.94 in paper mode: {summary['paper_mode_reaches_094']}",
        "",
        "Notes:",
        "- Paper mode is for literature-style reproduction and can be optimistic.",
        "- Strict mode is the deployment-safe estimate.",
    ]
    (out_dir / "PAPER_MODE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("Paper mode mean AUC:", round(float(np.mean(paper_auc)), 4))
    print("Strict mode mean AUC:", round(float(np.mean(strict_auc)), 4))
    print("Saved summary:", out_dir / "paper_mode_auc_summary.json")


if __name__ == "__main__":
    main()
