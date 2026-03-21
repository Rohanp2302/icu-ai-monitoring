"""
Phase 2: Data Augmentation

Four augmentation strategies for robust training:

1. Temporal Perturbations (20%): Shift measurements by ±2 hours
2. Feature Masking (20%): Randomly mask 10-30% of features
3. Synthetic Trajectory Generation (10%): Generate realistic ICU trajectories
4. Clinical Perturbations (15%): Simulate medication effects

Applied randomly during training: 50% of batches augmented, 30% unaugmented
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


class DataAugmenter:
    """Implement augmentation strategies for ICU data"""

    def __init__(self, random_state=42, augmentation_prob=0.5):
        """
        Args:
            random_state: Random seed
            augmentation_prob: Probability of applying augmentation (vs keeping original)
        """
        self.random_state = random_state
        self.augmentation_prob = augmentation_prob
        np.random.seed(random_state)

        # Clinical perturbation parameters
        # Simulate medication effects (e.g., vasopressors, sedatives)
        self.clinical_effects = {
            'vasopressor': {'hr_increase': 0.05, 'map_increase': 15, 'prob': 0.3},
            'sedative': {'hr_decrease': 0.1, 'rr_decrease': 0.2, 'prob': 0.2},
            'antibiotics': {'rr_increase': 0.03, 'temp_change': 0.5, 'prob': 0.4},
        }

    def temporal_perturbation(self, X, max_shift=2, shift_prob=0.5):
        """
        Shift measurements by ±max_shift hours (simulate measurement timing variability).

        Args:
            X: (T, F) single sample timeseries
            max_shift: Maximum hours to shift (default ±2)
            shift_prob: Probability of applying perturbation

        Returns:
            X_perturbed: (T, F) with shifted measurements
        """
        if np.random.rand() > shift_prob:
            return X.copy()

        T, F = X.shape
        shift = np.random.randint(-max_shift, max_shift + 1)

        if shift == 0:
            return X.copy()

        X_perturbed = np.full_like(X, np.nan)

        if shift > 0:
            # Shift forward (first `shift` rows become NaN)
            X_perturbed[shift:] = X[:-shift]
        else:
            # Shift backward (last `|shift|` rows become NaN)
            X_perturbed[:shift] = X[-shift:]

        return X_perturbed

    def feature_masking(self, X, mask_prob_min=0.1, mask_prob_max=0.3, mask_prob_apply=0.5):
        """
        Randomly mask 10-30% of features in each sample (simulate missing measurements).

        Args:
            X: (T, F) single sample
            mask_prob_min: Minimum fraction of features to mask
            mask_prob_max: Maximum fraction of features to mask
            mask_prob_apply: Probability of applying masking

        Returns:
            X_masked: (T, F) with random features set to NaN
        """
        if np.random.rand() > mask_prob_apply:
            return X.copy()

        T, F = X.shape
        X_masked = X.copy()

        # Randomly choose mask probability
        mask_prob = np.random.uniform(mask_prob_min, mask_prob_max)

        # For each feature, mask with probability mask_prob
        for f in range(F):
            if np.random.rand() < mask_prob:
                # Mask all timesteps for this feature
                X_masked[:, f] = np.nan

        return X_masked

    def synthetic_trajectory(self, X, outcome_severity=None, trajectory_prob=0.5):
        """
        Generate synthetic ICU trajectory by interpolating from existing patterns.

        This creates realistic variations by:
        1. Sampling trajectory patterns from similar patients
        2. Interpolating smoothly between key points
        3. Adding small realistic noise

        Args:
            X: (T, F) single sample reference trajectory
            outcome_severity: Optional severity level (0-1) to condition generation
            trajectory_prob: Probability of applying

        Returns:
            X_synthetic: (T, F) with smoothly interpolated trajectory
        """
        if np.random.rand() > trajectory_prob:
            return X.copy()

        T, F = X.shape
        X_synthetic = np.full_like(X, np.nan)

        for f in range(F):
            signal = X[:, f]
            valid_idx = np.where(~np.isnan(signal))[0]

            if len(valid_idx) < 2:
                X_synthetic[:, f] = signal.copy()
                continue

            # Create interpolation function
            try:
                f_interp = interp1d(
                    valid_idx, signal[valid_idx],
                    kind='cubic', bounds_error=False, fill_value=np.nan
                )

                # Interpolate all timesteps
                X_synthetic[:, f] = f_interp(np.arange(T))

                # Add small noise to interpolated values
                noise = np.random.normal(0, 0.1, T)
                valid_synth = ~np.isnan(X_synthetic[:, f])
                X_synthetic[valid_synth, f] += noise[valid_synth]

            except Exception:
                X_synthetic[:, f] = signal.copy()

        return X_synthetic

    def clinical_perturbation(self, X, feature_names=None, pert_prob=0.5):
        """
        Simulate clinical medication effects on vitals.

        Example effects:
        - Vasopressors: increase HR by 5%, increase BP
        - Sedatives: decrease HR by 10%, decrease RR
        - Antibiotics: increase RR by 3%, change temperature

        Args:
            X: (T, F) single sample where F should be [HR, RR, SaO2, ...]
            feature_names: List of feature names (default: ['heartrate', 'respiration', 'sao2'])
            pert_prob: Probability of applying perturbation

        Returns:
            X_perturbed: (T, F) with clinical effects applied
        """
        if np.random.rand() > pert_prob:
            return X.copy()

        if feature_names is None:
            feature_names = ['heartrate', 'respiration', 'sao2']

        X_perturbed = X.copy()
        T, F = X.shape

        # Randomly apply medication effect
        effect_type = np.random.choice(
            list(self.clinical_effects.keys()),
            p=[0.3, 0.2, 0.5]  # Probabilities for vasopressor, sedative, antibiotics
        )

        effect_params = self.clinical_effects[effect_type]

        # Generate random time window when effect starts (first 50% of timesteps)
        start_time = np.random.randint(0, T // 2)
        effect_duration = np.random.randint(T // 4, T)
        end_time = min(start_time + effect_duration, T)

        # Create effect amplitude curve (sigmoid rise, plateau, sigmoid fall)
        effect_curve = np.zeros(T)
        effect_curve[start_time:end_time] = np.linspace(0, 1, end_time - start_time)

        # Apply effects to corresponding features
        if effect_type == 'vasopressor':
            if 'heartrate' in feature_names:
                hr_idx = feature_names.index('heartrate')
                hr_increase = effect_params['hr_increase']
                valid = ~np.isnan(X_perturbed[:, hr_idx])
                X_perturbed[valid, hr_idx] *= (1 + hr_increase * effect_curve[valid])

        elif effect_type == 'sedative':
            if 'heartrate' in feature_names:
                hr_idx = feature_names.index('heartrate')
                hr_decrease = effect_params['hr_decrease']
                valid = ~np.isnan(X_perturbed[:, hr_idx])
                X_perturbed[valid, hr_idx] *= (1 - hr_decrease * effect_curve[valid])

            if 'respiration' in feature_names:
                rr_idx = feature_names.index('respiration')
                rr_decrease = effect_params['rr_decrease']
                valid = ~np.isnan(X_perturbed[:, rr_idx])
                X_perturbed[valid, rr_idx] *= (1 - rr_decrease * effect_curve[valid])

        elif effect_type == 'antibiotics':
            if 'respiration' in feature_names:
                rr_idx = feature_names.index('respiration')
                rr_increase = effect_params['rr_increase']
                valid = ~np.isnan(X_perturbed[:, rr_idx])
                X_perturbed[valid, rr_idx] *= (1 + rr_increase * effect_curve[valid])

        return X_perturbed

    def augment_batch(self, X_batch, augmentation_strategy='random'):
        """
        Apply augmentation to a batch of samples.

        Args:
            X_batch: (B, T, F) batch of samples
            augmentation_strategy: 'random', 'temporal', 'masking', 'synthetic', 'clinical', or 'none'

        Returns:
            X_augmented: (B, T, F) augmented batch
        """
        B, T, F = X_batch.shape
        X_augmented = X_batch.copy()

        augmentation_types = {
            'temporal': (self.temporal_perturbation, 0.2),
            'masking': (self.feature_masking, 0.2),
            'synthetic': (self.synthetic_trajectory, 0.1),
            'clinical': (self.clinical_perturbation, 0.15),
            'none': (lambda x: x.copy(), 0.35)
        }

        for b in range(B):
            if augmentation_strategy == 'random':
                # Randomly select augmentation type weighted by frequency
                aug_type = np.random.choice(
                    list(augmentation_types.keys()),
                    p=list(v[1] for v in augmentation_types.values())
                )
            else:
                aug_type = augmentation_strategy

            aug_func = augmentation_types[aug_type][0]
            X_augmented[b] = aug_func(X_batch[b])

        return X_augmented

    def create_augmentation_pipeline(self, X, split_indices, output_dir='data', prefix=''):
        """
        Create augmented versions of dataset with proper train/val/test handling.

        Only apply augmentation to training set, keep val/test unchanged.

        Args:
            X: (N, T, F) full dataset
            split_indices: Dict with 'train_indices', 'val_indices', 'test_indices'
            output_dir: Directory to save augmented data
            prefix: Filename prefix (e.g., 'eicu_')

        Returns:
            X_augmented: (N, T, F) with augmented training samples
            augmentation_info: Dict with augmentation statistics
        """
        print(f"[Augmentation Pipeline] Creating augmented versions...")

        N, T, F = X.shape
        X_augmented = X.copy()

        train_idx = split_indices['train_indices']
        val_idx = split_indices['val_indices']
        test_idx = split_indices['test_indices']

        counts = {'temporal': 0, 'masking': 0, 'synthetic': 0, 'clinical': 0, 'none': 0}

        # Augment training set only
        print(f"  Augmenting {len(train_idx):,} training samples...")
        for i, idx in enumerate(train_idx):
            # Randomly select augmentation type
            aug_type = np.random.choice(
                ['temporal', 'masking', 'synthetic', 'clinical', 'none'],
                p=[0.2, 0.2, 0.1, 0.15, 0.35]
            )
            counts[aug_type] += 1

            if aug_type == 'temporal':
                X_augmented[idx] = self.temporal_perturbation(X[idx])
            elif aug_type == 'masking':
                X_augmented[idx] = self.feature_masking(X[idx])
            elif aug_type == 'synthetic':
                X_augmented[idx] = self.synthetic_trajectory(X[idx])
            elif aug_type == 'clinical':
                X_augmented[idx] = self.clinical_perturbation(X[idx])
            # else: 'none' - keep original

            if (i + 1) % 10000 == 0:
                print(f"    Processed {i+1:,} samples...")

        # Validation and test sets remain unchanged
        print(f"  Validation set ({len(val_idx):,} samples): no augmentation")
        print(f"  Test set ({len(test_idx):,} samples): no augmentation")

        # Save augmented data
        print(f"\n[Saving] Augmented dataset...")
        np.save(f'{output_dir}/{prefix}X_augmented.npy', X_augmented)

        augmentation_stats = {
            'total_samples': N,
            'augmented_training_samples': len(train_idx),
            'augmentation_distribution': {
                'temporal_perturbation': counts['temporal'],
                'feature_masking': counts['masking'],
                'synthetic_trajectory': counts['synthetic'],
                'clinical_perturbation': counts['clinical'],
                'no_augmentation': counts['none']
            },
            'val_test_note': 'Validation and test sets preserved without augmentation'
        }

        import json
        with open(f'{output_dir}/{prefix}augmentation_stats.json', 'w') as f:
            json.dump(augmentation_stats, f, indent=2)

        print(f"  Saved: {output_dir}/{prefix}X_augmented.npy")
        print(f"  Saved: {output_dir}/{prefix}augmentation_stats.json")
        print(f"\nAugmentation breakdown (train set):")
        for aug_type, count in counts.items():
            pct = 100 * count / len(train_idx)
            print(f"  {aug_type:25} {count:>8,} ({pct:>5.1f}%)")

        return X_augmented, augmentation_stats


def main():
    """Run data augmentation pipeline"""
    print("=" * 80)
    print("PHASE 2: DATA AUGMENTATION")
    print("=" * 80)

    # Load engineered features
    print("\n[1] Loading engineered features...")
    try:
        X_eicu_eng = np.load('data/eicu_X_engineered.npy')
        X_physio_eng = np.load('data/physio_X_engineered.npy')
        print(f"  eICU engineered:   {X_eicu_eng.shape}")
        print(f"  PhysioNet engineered: {X_physio_eng.shape}")
    except FileNotFoundError:
        print("  Note: Engineered features not found. Run feature_engineering.py first.")
        print("  For now, using raw data from original tensors...")
        from src.icu_data_loader import load_datasets
        X_eicu_eng, X_physio_eng, _ = load_datasets()

    # Load split indices
    print("\n[2] Loading split indices...")
    split_data = np.load('data/split_indices.npz')
    split_indices = {
        'train_indices': split_data['train'],
        'val_indices': split_data['val'],
        'test_indices': split_data['test']
    }
    print(f"  Train: {len(split_indices['train_indices']):,}")
    print(f"  Val: {len(split_indices['val_indices']):,}")
    print(f"  Test: {len(split_indices['test_indices']):,}")

    # Initialize augmenter
    augmenter = DataAugmenter(random_state=42)

    # Augment eICU
    print("\n[3] Augmenting eICU dataset...")
    X_eicu_aug, stats_eicu = augmenter.create_augmentation_pipeline(
        X_eicu_eng, split_indices, prefix='eicu_'
    )

    # Augment PhysioNet
    print("\n[4] Augmenting PhysioNet dataset...")
    X_physio_aug, stats_physio = augmenter.create_augmentation_pipeline(
        X_physio_eng, split_indices, prefix='physio_'
    )

    print("\n" + "=" * 80)
    print("DATA AUGMENTATION COMPLETE")
    print("=" * 80)
    print(f"eICU augmented:  {X_eicu_aug.shape}")
    print(f"PhysioNet augmented: {X_physio_aug.shape}")
    print(f"\nNext: Build multi-task deep learning model architecture")


if __name__ == "__main__":
    main()
