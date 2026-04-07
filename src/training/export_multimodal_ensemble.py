"""
Export fold best-checkpoints as a deployable multimodal ensemble package.

Creates:
- model_*.pt files
- ensemble_metadata.json
- calibration_thresholds.csv (copied from latest k-fold summary)
- ensemble_config.json (feature and dimension info)
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ensemble import ModelEnsemble
from src.models.multitask_model import MultiTaskICUModel
from src.training.train_multimodal_icu import build_patient_level_tensors


def latest_file(pattern: str) -> Path | None:
    files = sorted(Path(".").glob(pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Export multimodal ensemble from fold checkpoints")
    parser.add_argument("--hourly", type=str, default="data/processed/eicu_hourly_all_features.csv")
    parser.add_argument("--outcomes", type=str, default="data/processed/eicu_outcomes.csv")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints/multimodal")
    parser.add_argument("--logs-dir", type=str, default="logs/multimodal")
    parser.add_argument("--output-dir", type=str, default="results/multimodal_ensemble")
    parser.add_argument("--seq-len", type=int, default=24)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build tensors once to infer input dimensions and feature names.
    X_temporal, X_static, _, temporal_features, static_features = build_patient_level_tensors(
        Path(args.hourly), Path(args.outcomes), seq_len=args.seq_len
    )

    model_factory = lambda: MultiTaskICUModel(
        input_dim=X_temporal.shape[2],
        static_dim=X_static.shape[1],
    )

    ensemble = ModelEnsemble(model_factory, device="cpu")

    ckpt_dir = Path(args.checkpoints_dir)
    ckpts = sorted(ckpt_dir.glob("fold_*_best_model.pt"), key=lambda p: p.name)
    if not ckpts:
        raise FileNotFoundError(f"No fold checkpoints found in {ckpt_dir}")

    for ckpt in ckpts:
        ensemble.add_model(str(ckpt), model_name=ckpt.stem)

    ensemble.save_ensemble(str(output_dir))

    # Copy latest k-fold summary for calibrated thresholds.
    latest_summary = latest_file(f"{args.logs_dir}/kfold_summary_*.csv")
    if latest_summary is not None:
        summary_df = pd.read_csv(latest_summary)
        cols = [
            c
            for c in [
                "fold",
                "mortality_threshold",
                "mortality_auc",
                "mortality_f1",
                "mortality_sensitivity",
                "mortality_specificity",
            ]
            if c in summary_df.columns
        ]
        if cols:
            summary_df[cols].to_csv(output_dir / "calibration_thresholds.csv", index=False)
        shutil.copy2(latest_summary, output_dir / "kfold_summary_source.csv")

    config = {
        "input_dim": int(X_temporal.shape[2]),
        "static_dim": int(X_static.shape[1]),
        "seq_len": int(X_temporal.shape[1]),
        "n_models": int(len(ckpts)),
        "temporal_features": temporal_features,
        "static_features": static_features,
    }
    with open(output_dir / "ensemble_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Ensemble export complete")
    print("Output directory:", output_dir)
    print("Models exported:", len(ckpts))


if __name__ == "__main__":
    main()
