"""
Phase 2 Verification Script

Test feature engineering and augmentation pipeline:
1. Load raw data
2. Apply feature engineering
3. Load splits
4. Apply augmentation
5. Verify output shapes and data quality
"""

import numpy as np
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering import FeatureEngineer
from src.augmentation import DataAugmenter
from src.icu_data_loader import load_datasets


def verify_phase2():
    """Run Phase 2 verification"""
    print("=" * 80)
    print("PHASE 2 VERIFICATION")
    print("=" * 80)

    # Step 1: Load raw data
    print("\n[1/5] Loading raw datasets...")
    try:
        X_eicu, X_physio, stats = load_datasets()
        print(f"  ✓ X_eicu: {X_eicu.shape}")
        print(f"  ✓ X_physio: {X_physio.shape}")
    except FileNotFoundError as e:
        print(f"  ✗ Error loading datasets: {e}")
        return False

    # Step 2: Feature Engineering
    print("\n[2/5] Running feature engineering...")
    try:
        fe = FeatureEngineer('data/therapeutic_targets.json')

        X_eicu_eng, entropy_eicu = fe.engineer_features(X_eicu)
        X_physio_eng, entropy_physio = fe.engineer_features(X_physio)

        print(f"  ✓ eICU engineered: {X_eicu_eng.shape} (features: {X_eicu_eng.shape[2]})")
        print(f"  ✓ PhysioNet engineered: {X_physio_eng.shape} (features: {X_physio_eng.shape[2]})")
        print(f"  ✓ eICU entropy: {entropy_eicu.shape}")
        print(f"  ✓ PhysioNet entropy: {entropy_physio.shape}")

        # Save engineered features
        fe.save_engineered_features(X_eicu_eng, entropy_eicu, prefix='eicu_')
        fe.save_engineered_features(X_physio_eng, entropy_physio, prefix='physio_')

    except Exception as e:
        print(f"  ✗ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 3: Load splits
    print("\n[3/5] Loading split indices...")
    try:
        split_data = np.load('data/split_indices.npz')
        split_indices = {
            'train_indices': split_data['train'],
            'val_indices': split_data['val'],
            'test_indices': split_data['test']
        }
        print(f"  ✓ Train: {len(split_indices['train_indices']):,}")
        print(f"  ✓ Val: {len(split_indices['val_indices']):,}")
        print(f"  ✓ Test: {len(split_indices['test_indices']):,}")
    except FileNotFoundError as e:
        print(f"  ✗ Error loading splits: {e}")
        return False

    # Step 4: Augmentation
    print("\n[4/5] Running data augmentation...")
    try:
        augmenter = DataAugmenter(random_state=42)

        X_eicu_aug, stats_eicu = augmenter.create_augmentation_pipeline(
            X_eicu_eng, split_indices, prefix='eicu_'
        )
        X_physio_aug, stats_physio = augmenter.create_augmentation_pipeline(
            X_physio_eng, split_indices, prefix='physio_'
        )

        print(f"  ✓ eICU augmented: {X_eicu_aug.shape}")
        print(f"  ✓ PhysioNet augmented: {X_physio_aug.shape}")

    except Exception as e:
        print(f"  ✗ Augmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Step 5: Verify data quality
    print("\n[5/5] Verifying data quality...")
    try:
        # Check for NaN patterns
        eicu_valid_pct = 100 * np.sum(~np.isnan(X_eicu_aug)) / X_eicu_aug.size
        physio_valid_pct = 100 * np.sum(~np.isnan(X_physio_aug)) / X_physio_aug.size

        print(f"  ✓ eICU valid data: {eicu_valid_pct:.1f}%")
        print(f"  ✓ PhysioNet valid data: {physio_valid_pct:.1f}%")

        # Check engineering metadata
        with open('data/eicu_feature_engineering_metadata.json', 'r') as f:
            eng_meta = json.load(f)
        print(f"\n  ✓ Feature Engineering Metadata:")
        print(f"    - Features per timestep: {eng_meta['features_per_timestep']}")
        print(f"    - Feature groups: {eng_meta['feature_groups']}")

        # Check augmentation stats
        with open('data/eicu_augmentation_stats.json', 'r') as f:
            aug_stats = json.load(f)
        print(f"\n  ✓ Augmentation Statistics (eICU):")
        for aug_type, count in aug_stats['augmentation_distribution'].items():
            pct = 100 * count / aug_stats['augmented_training_samples']
            print(f"    - {aug_type:25} {count:>8,} ({pct:>5.1f}%)")

    except Exception as e:
        print(f"  ✗ Data quality verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n" + "=" * 80)
    print("PHASE 2 VERIFICATION COMPLETE ✓")
    print("=" * 80)
    print("\nSummary:")
    print(f"  - Feature Engineering: {X_eicu_eng.shape[2]} features per timestep")
    print(f"  - eICU samples: {len(X_eicu_eng):,}")
    print(f"  - PhysioNet samples: {len(X_physio_eng):,}")
    print(f"  - Training samples augmented: {aug_stats['augmented_training_samples']:,}")
    print(f"  - Data quality preserved: {eicu_valid_pct:.1f}% + {physio_valid_pct:.1f}% valid")
    print(f"\n✓ Ready for Phase 3: Multi-Task Deep Learning Model")

    return True


if __name__ == "__main__":
    success = verify_phase2()
    sys.exit(0 if success else 1)
