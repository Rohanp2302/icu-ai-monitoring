"""
Dual-track AUC benchmark for ICU mortality.

Track A (Strict):
- Early-window features only (first 24h)
- Excludes clear post-outcome leakage fields

Track B (Upper-bound Diagnostic):
- Uses full available timeline and leakage-prone metadata
- Intended as a ceiling diagnostic, not deployment-safe metric
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    hourly = pd.read_csv("data/processed/eicu_hourly_all_features.csv")
    outcomes = pd.read_csv("data/processed/eicu_outcomes.csv")
    return hourly, outcomes


def parse_age(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip()
    if s.startswith(">"):
        s = s[1:].strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def build_patient_features(hourly: pd.DataFrame, first_24h_only: bool) -> pd.DataFrame:
    id_col = "patientunitstayid"
    work = hourly.copy()

    if first_24h_only and "hour" in work.columns:
        work = work[(work["hour"] >= 0) & (work["hour"] < 24)].copy()

    num_cols = [
        c
        for c in work.columns
        if c not in [id_col, "hour"] and pd.api.types.is_numeric_dtype(work[c])
    ]

    agg = work.groupby(id_col)[num_cols].agg(["mean", "std", "min", "max", "median"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()

    return agg


def prepare_track_dataset(track: str, hourly: pd.DataFrame, outcomes: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    id_col = "patientunitstayid"
    target_col = "mortality"

    if track == "strict":
        feat_df = build_patient_features(hourly, first_24h_only=True)

        keep_cols = [id_col, target_col, "age", "gender", "ethnicity", "unittype", "apacheadmissiondx"]
        keep_cols = [c for c in keep_cols if c in outcomes.columns]
        meta = outcomes[keep_cols].drop_duplicates(id_col).copy()

        if "age" in meta.columns:
            meta["age"] = meta["age"].apply(parse_age)

        df = feat_df.merge(meta, on=id_col, how="inner")

    elif track == "upper_bound":
        feat_df = build_patient_features(hourly, first_24h_only=False)

        meta = outcomes.drop_duplicates(id_col).copy()
        if "age" in meta.columns:
            meta["age"] = meta["age"].apply(parse_age)

        df = feat_df.merge(meta, on=id_col, how="inner")
    else:
        raise ValueError("track must be strict or upper_bound")

    y = pd.to_numeric(df[target_col], errors="coerce").fillna(0).astype(int).values
    X = df.drop(columns=[id_col, target_col])

    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
    )


def get_models() -> Dict[str, object]:
    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "rf": RandomForestClassifier(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=900,
            max_depth=None,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    # Optional gradient boosters
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=1800,
            learning_rate=0.02,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="auc",
            reg_alpha=0.5,
            reg_lambda=2.0,
            min_child_weight=3,
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        pass

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = LGBMClassifier(
            n_estimators=1800,
            learning_rate=0.02,
            num_leaves=63,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        pass

    return models


def evaluate_track(track_name: str, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pre = build_preprocessor(X)
    models = get_models()

    rows: List[Dict] = []
    for name, model in models.items():
        pipe = Pipeline(steps=[("pre", pre), ("model", model)])

        scores = cross_validate(
            pipe,
            X,
            y,
            cv=cv,
            scoring={"auc": "roc_auc"},
            n_jobs=1,
            return_train_score=False,
        )

        rows.append(
            {
                "track": track_name,
                "model": name,
                "auc_mean": float(np.mean(scores["test_auc"])),
                "auc_std": float(np.std(scores["test_auc"])),
                "fold_auc": [float(v) for v in scores["test_auc"]],
            }
        )

    return pd.DataFrame(rows).sort_values("auc_mean", ascending=False)


def main() -> None:
    out_dir = Path("results") / "auc_dual_track"
    out_dir.mkdir(parents=True, exist_ok=True)

    hourly, outcomes = load_data()

    X_strict, y_strict = prepare_track_dataset("strict", hourly, outcomes)
    X_ub, y_ub = prepare_track_dataset("upper_bound", hourly, outcomes)

    strict_df = evaluate_track("strict", X_strict, y_strict)
    ub_df = evaluate_track("upper_bound", X_ub, y_ub)

    all_df = pd.concat([strict_df, ub_df], ignore_index=True)
    all_df.to_csv(out_dir / "auc_dual_track_results.csv", index=False)

    strict_best = strict_df.iloc[0].to_dict()
    ub_best = ub_df.iloc[0].to_dict()

    summary = {
        "strict_best": {
            "model": strict_best["model"],
            "auc_mean": float(strict_best["auc_mean"]),
            "auc_std": float(strict_best["auc_std"]),
        },
        "upper_bound_best": {
            "model": ub_best["model"],
            "auc_mean": float(ub_best["auc_mean"]),
            "auc_std": float(ub_best["auc_std"]),
        },
        "strict_reaches_093": bool(float(strict_best["auc_mean"]) >= 0.93),
        "upper_bound_reaches_093": bool(float(ub_best["auc_mean"]) >= 0.93),
    }

    with open(out_dir / "auc_dual_track_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    report_lines = [
        "# Dual-Track AUC Benchmark",
        "",
        "## Strict Track (deployment-safe)",
        f"- Best model: {summary['strict_best']['model']}",
        f"- Mean AUC: {summary['strict_best']['auc_mean']:.4f}",
        f"- AUC std: {summary['strict_best']['auc_std']:.4f}",
        f"- Reaches 0.93: {summary['strict_reaches_093']}",
        "",
        "## Upper-Bound Track (diagnostic, leakage-prone)",
        f"- Best model: {summary['upper_bound_best']['model']}",
        f"- Mean AUC: {summary['upper_bound_best']['auc_mean']:.4f}",
        f"- AUC std: {summary['upper_bound_best']['auc_std']:.4f}",
        f"- Reaches 0.93: {summary['upper_bound_reaches_093']}",
        "",
        "## Notes",
        "- Strict track excludes obvious post-outcome leakage fields and uses first-24h temporal windows.",
        "- Upper-bound track is for diagnostic ceiling analysis only.",
    ]

    (out_dir / "DUAL_TRACK_AUC_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")

    print("Saved:", out_dir / "auc_dual_track_results.csv")
    print("Saved:", out_dir / "auc_dual_track_summary.json")
    print("Saved:", out_dir / "DUAL_TRACK_AUC_REPORT.md")
    print("Strict best AUC:", f"{summary['strict_best']['auc_mean']:.4f}")
    print("Upper-bound best AUC:", f"{summary['upper_bound_best']['auc_mean']:.4f}")


if __name__ == "__main__":
    main()
