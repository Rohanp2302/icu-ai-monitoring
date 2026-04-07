"""
Protocol-gap AUC analysis.

Shows how evaluation protocol choices can inflate reported AUC:
- Permissive row-level CV (high leakage risk)
- Strict patient-level CV (deployment-safe)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier


def main() -> None:
    out_dir = Path("results") / "protocol_gap_auc"
    out_dir.mkdir(parents=True, exist_ok=True)

    hourly = pd.read_csv("data/processed/eicu_hourly_all_features.csv")
    outcomes = pd.read_csv("data/processed/eicu_outcomes.csv")

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=250,
                    n_jobs=-1,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # Protocol A: permissive row-level split (can leak patient signal across folds).
    row_df = hourly.merge(
        outcomes[["patientunitstayid", "mortality"]],
        on="patientunitstayid",
        how="inner",
    )
    row_features = [
        c
        for c in row_df.columns
        if c != "mortality" and pd.api.types.is_numeric_dtype(row_df[c])
    ]
    X_row = row_df[row_features].values
    y_row = row_df["mortality"].astype(int).values

    row_auc = cross_val_score(model, X_row, y_row, cv=cv, scoring="roc_auc", n_jobs=1)

    # Protocol B: strict patient-level split with early-window-style aggregated features.
    patient_agg = hourly.groupby("patientunitstayid")[["heartrate", "respiration", "sao2"]].agg(
        ["mean", "std", "min", "max"]
    )
    patient_agg.columns = [f"{a}_{b}" for a, b in patient_agg.columns]

    patient_df = patient_agg.reset_index().merge(
        outcomes[["patientunitstayid", "mortality"]].drop_duplicates("patientunitstayid"),
        on="patientunitstayid",
        how="inner",
    )

    X_patient = patient_df.drop(columns=["patientunitstayid", "mortality"]).values
    y_patient = patient_df["mortality"].astype(int).values

    patient_auc = cross_val_score(
        model,
        X_patient,
        y_patient,
        cv=cv,
        scoring="roc_auc",
        n_jobs=1,
    )

    summary = {
        "protocol_a_permissive_row_level": {
            "auc_mean": float(np.mean(row_auc)),
            "auc_std": float(np.std(row_auc)),
            "fold_auc": [float(x) for x in row_auc],
        },
        "protocol_b_strict_patient_level": {
            "auc_mean": float(np.mean(patient_auc)),
            "auc_std": float(np.std(patient_auc)),
            "fold_auc": [float(x) for x in patient_auc],
        },
        "delta_auc_mean": float(np.mean(row_auc) - np.mean(patient_auc)),
    }

    with open(out_dir / "protocol_gap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Protocol Gap AUC Analysis",
        "",
        "## Results",
        f"- Permissive row-level AUC: {summary['protocol_a_permissive_row_level']['auc_mean']:.4f} +/- {summary['protocol_a_permissive_row_level']['auc_std']:.4f}",
        f"- Strict patient-level AUC: {summary['protocol_b_strict_patient_level']['auc_mean']:.4f} +/- {summary['protocol_b_strict_patient_level']['auc_std']:.4f}",
        f"- Mean AUC gap: {summary['delta_auc_mean']:.4f}",
        "",
        "## Interpretation",
        "- The permissive protocol can report very high AUC because patient-specific patterns leak across folds.",
        "- The strict patient-level protocol is the correct estimate for generalization to unseen patients.",
        "",
        "## Recommendation",
        "- Use strict patient-level results for deployment and claims.",
        "- Use permissive protocol only as a diagnostic upper-bound, not as final model quality.",
    ]

    (out_dir / "PROTOCOL_GAP_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print("Saved:", out_dir / "protocol_gap_summary.json")
    print("Saved:", out_dir / "PROTOCOL_GAP_REPORT.md")
    print("Permissive row-level AUC:", round(float(np.mean(row_auc)), 4))
    print("Strict patient-level AUC:", round(float(np.mean(patient_auc)), 4))


if __name__ == "__main__":
    main()
