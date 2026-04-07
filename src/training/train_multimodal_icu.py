"""
Train multimodal ICU model (temporal + static) on processed eICU files.

This script builds:
- Temporal tensor: patient-level 24h windows from hourly vitals
- Static tensor: demographics and admission-level context
- Multi-task targets: mortality, risk class, LOS

Then runs k-fold training with the project MultiTaskICUModel.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.multitask_model import MultiTaskICUModel, MultiTaskLoss
from src.training.kfold_trainer import KFoldTrainer


def parse_age_to_numeric(age_value) -> float:
    if pd.isna(age_value):
        return np.nan
    text = str(age_value).strip()
    if text.startswith(">"):
        return float(text.replace(">", "").strip())
    try:
        return float(text)
    except ValueError:
        return np.nan


def build_patient_level_tensors(
    hourly_path: Path,
    outcomes_path: Path,
    seq_len: int = 24,
    temporal_features: List[str] | None = None,
    missing_threshold: float = 0.8,
    max_temporal_features: int = 6,
    add_mask_channels: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray], List[str], List[str]]:
    hourly = pd.read_csv(hourly_path)
    outcomes = pd.read_csv(outcomes_path)

    id_col = "patientunitstayid"
    hour_col = "hour"

    core_temporal = [
        "heartrate",
        "respiration",
        "sao2",
        "temperature",
        "systemicsystolic",
        "systemicdiastolic",
    ]

    if temporal_features is None:
        numeric_candidates = [
            c
            for c in hourly.columns
            if c not in [id_col, hour_col] and pd.api.types.is_numeric_dtype(hourly[c])
        ]
        miss = hourly[numeric_candidates].isna().mean().sort_values()
        selected = [c for c in miss.index if miss[c] <= missing_threshold]

        # Always include core ICU vitals when present.
        for feat in reversed(core_temporal):
            if feat in hourly.columns and feat not in selected:
                selected.insert(0, feat)

        seen = set()
        deduped = []
        for feat in selected:
            if feat not in seen:
                deduped.append(feat)
                seen.add(feat)

        temporal_features = deduped[:max_temporal_features]

    temporal_features = [c for c in temporal_features if c in hourly.columns]
    if len(temporal_features) < 3:
        raise ValueError("Not enough temporal features available in hourly data")

    base_temporal_features = list(temporal_features)
    output_temporal_features = list(base_temporal_features)
    if add_mask_channels:
        output_temporal_features += [f"{f}_mask" for f in base_temporal_features]

    outcomes = outcomes.copy()
    outcomes["age_num"] = outcomes["age"].apply(parse_age_to_numeric)
    outcomes["icu_los_days"] = pd.to_numeric(outcomes.get("icu_los_days"), errors="coerce")

    # Static features
    static_df = outcomes[[id_col, "age_num", "gender", "icu_los_days", "ethnicity"]].drop_duplicates(id_col)

    gender_map = {"Male": 1.0, "Female": 0.0}
    static_df["gender_num"] = static_df["gender"].map(gender_map).fillna(0.5)

    top_eth = static_df["ethnicity"].fillna("Unknown").value_counts().head(5).index.tolist()
    static_df["ethnicity_bucket"] = static_df["ethnicity"].where(static_df["ethnicity"].isin(top_eth), "Other").fillna("Unknown")
    static_df = pd.get_dummies(static_df, columns=["ethnicity_bucket"], prefix="eth", dummy_na=False)

    static_cols = [c for c in static_df.columns if c not in [id_col, "gender", "ethnicity"]]
    for c in static_cols:
        static_df[c] = pd.to_numeric(static_df[c], errors="coerce")
    static_df[static_cols] = static_df[static_cols].fillna(static_df[static_cols].median())

    static_map = static_df.set_index(id_col)

    # Labels
    label_df = outcomes[[id_col, "mortality", "icu_los_days"]].drop_duplicates(id_col).copy()
    label_df["mortality"] = pd.to_numeric(label_df["mortality"], errors="coerce").fillna(0).astype(int)
    label_df["icu_los_days"] = pd.to_numeric(label_df["icu_los_days"], errors="coerce").fillna(label_df["icu_los_days"].median())

    los_rank = label_df["icu_los_days"].rank(method="first")
    label_df["risk"] = pd.qcut(los_rank, q=4, labels=False).astype(int)
    label_df.loc[label_df["mortality"] == 1, "risk"] = np.maximum(
        label_df.loc[label_df["mortality"] == 1, "risk"].values,
        2,
    )

    labels_map = label_df.set_index(id_col)

    patient_ids = hourly[id_col].dropna().unique()

    x_temporal, x_static = [], []
    y_mortality, y_risk, y_los = [], [], []

    for pid in patient_ids:
        if pid not in static_map.index or pid not in labels_map.index:
            continue

        patient_rows = hourly[hourly[id_col] == pid].sort_values(hour_col)

        raw_values = patient_rows[base_temporal_features].to_numpy(dtype=np.float32)
        observed_mask = np.isfinite(raw_values).astype(np.float32)
        values = patient_rows[base_temporal_features].ffill().bfill().fillna(0.0).to_numpy(dtype=np.float32)

        if add_mask_channels:
            values = np.concatenate([values, observed_mask], axis=1)

        if values.shape[0] == 0:
            continue

        if values.shape[0] >= seq_len:
            seq = values[-seq_len:]
        else:
            pad_n = seq_len - values.shape[0]
            pad_val = values[-1:] if values.shape[0] > 0 else np.zeros((1, values.shape[1]), dtype=np.float32)
            seq = np.vstack([np.repeat(pad_val, pad_n, axis=0), values])

        static_vec = static_map.loc[pid, static_cols].to_numpy(dtype=np.float32)

        mortality = int(labels_map.loc[pid, "mortality"])
        risk = int(labels_map.loc[pid, "risk"])
        total_los = float(labels_map.loc[pid, "icu_los_days"])
        remaining_los = max(total_los * 0.5, 0.0)

        x_temporal.append(seq)
        x_static.append(static_vec)
        y_mortality.append(mortality)
        y_risk.append(risk)
        y_los.append([total_los, remaining_los])

    X_temporal = np.array(x_temporal, dtype=np.float32)
    X_static = np.array(x_static, dtype=np.float32)

    # Global z-score normalization (kept simple for this training utility).
    n_base = len(base_temporal_features)
    t_mean = X_temporal[:, :, :n_base].mean(axis=(0, 1), keepdims=True)
    t_std = X_temporal[:, :, :n_base].std(axis=(0, 1), keepdims=True) + 1e-6
    X_temporal[:, :, :n_base] = (X_temporal[:, :, :n_base] - t_mean) / t_std

    s_mean = X_static.mean(axis=0, keepdims=True)
    s_std = X_static.std(axis=0, keepdims=True) + 1e-6
    X_static = (X_static - s_mean) / s_std

    y_dict = {
        "mortality": np.array(y_mortality, dtype=np.float32),
        "risk": np.array(y_risk, dtype=np.int64),
        "los": np.array(y_los, dtype=np.float32),
    }

    return X_temporal, X_static, y_dict, output_temporal_features, static_cols


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multimodal ICU model")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--hourly", type=str, default="data/processed/eicu_hourly_all_features.csv")
    parser.add_argument("--outcomes", type=str, default="data/processed/eicu_outcomes.csv")
    parser.add_argument("--missing-threshold", type=float, default=0.8)
    parser.add_argument("--max-temporal-features", type=int, default=6)
    parser.add_argument("--mask-channels", action="store_true")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--modality-dropout", type=float, default=0.1)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--focal-alpha", type=float, default=0.75)
    args = parser.parse_args()

    X_temporal, X_static, y_dict, temporal_features, static_cols = build_patient_level_tensors(
        Path(args.hourly),
        Path(args.outcomes),
        seq_len=24,
        missing_threshold=args.missing_threshold,
        max_temporal_features=args.max_temporal_features,
        add_mask_channels=args.mask_channels,
    )

    print("Prepared multimodal tensors")
    print("X_temporal:", X_temporal.shape)
    print("X_static:", X_static.shape)
    print("Temporal features:", temporal_features)
    print("Static features:", static_cols)
    print("Mortality prevalence:", float(y_dict["mortality"].mean()))

    model_factory = lambda: MultiTaskICUModel(
        input_dim=X_temporal.shape[2],
        static_dim=X_static.shape[1],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        modality_dropout_prob=args.modality_dropout,
    )

    loss_factory = lambda device: MultiTaskLoss(
        device=device,
        use_focal_for_mortality=True,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
    )

    trainer = KFoldTrainer(
        model_class=model_factory,
        loss_fn_class=loss_factory,
        device=args.device,
        n_splits=args.n_splits,
        checkpoint_dir="checkpoints/multimodal",
        log_dir="logs/multimodal",
    )

    trainer.run_kfold(
        X=X_temporal,
        X_static=X_static,
        y_dict=y_dict,
        epochs=args.epochs,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
