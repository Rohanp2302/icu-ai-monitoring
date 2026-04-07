"""
Complete ICU Deep Learning Pipeline
Trains model with multiple optimizers and generates comprehensive report
"""

import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import torch

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import model and comparator
try:
    from src.models.dl_icu_model import DeepICUModel, ICUDataset, load_and_prepare_data
    from src.models.optimizer_comparator import OptimizerComparator, save_comparison_report
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


def main():
    logger.info("="*70)
    logger.info("ICU MORTALITY PREDICTION - DEEP LEARNING PIPELINE")
    logger.info("="*70)

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # Load data
    data_path = 'data/processed/eicu_hourly_all_features.csv'

    if not Path(data_path).exists():
        logger.error(f"Data file not found: {data_path}")
        logger.info("Please run: python src/models/dl_data_extractor.py")
        return

    logger.info(f"Loading data from {data_path}...")
    X, y, scaler = load_and_prepare_data(data_path)

    logger.info(f"\nDataset Summary:")
    logger.info(f"  Samples: {len(X):,}")
    logger.info(f"  Sequence length: {X.shape[1]}")
    logger.info(f"  Features: {X.shape[2]}")
    logger.info(f"  Mortality rate: {y.mean():.1%}")
    logger.info(f"  Class distribution: Negative={np.sum(y==0):,}, Positive={np.sum(y==1):,}")

    # Define optimizers to compare
    optimizer_configs = [
        {'name': 'adam', 'lr': 0.001},
        {'name': 'adamw', 'lr': 0.001},
        {'name': 'sgd_momentum', 'lr': 0.01},
        {'name': 'sgd_nesterov', 'lr': 0.01},
        {'name': 'rmsprop', 'lr': 0.001},
    ]

    # Run comparison
    logger.info(f"\nStarting {len(optimizer_configs)}-Optimizer Comparison with 5-Fold Cross-Validation...")

    comparator = OptimizerComparator(DeepICUModel, device=device)

    summary, detailed_results = comparator.compare_optimizers(
        X, y,
        optimizer_configs=optimizer_configs,
        n_folds=5,
        epochs_per_fold=30
    )

    # Save reports
    report = save_comparison_report(summary)

    logger.info("\n" + "="*70)
    logger.info("PIPELINE COMPLETE")
    logger.info("="*70)

    return summary, report


if __name__ == '__main__':
    main()
