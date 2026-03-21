"""
Phase 2: Feature Engineering

Create derived features from 24-hour vital sign windows:
- Trend features (slopes, first/second derivatives)
- Statistical features (min, max, mean, std, percentiles)
- Volatility features (coefficient of variation, rolling std)
- Therapeutic deviation (distance from target ranges)
- Temporal patterns (autocorrelation, entropy)

Input: (N, 24, 3) tensors [heartrate, respiration, sao2]
Output: (N, 24, F) where F=50+ features per timestep
        + (N, static_features) demographic/comorbidity info
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from scipy.signal import savgol_filter
import json
import warnings
warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Create derived features from vital sign timeseries"""

    def __init__(self, therapeutic_targets_path='data/therapeutic_targets.json'):
        """
        Args:
            therapeutic_targets_path: Path to therapeutic target ranges
        """
        # Load therapeutic targets
        with open(therapeutic_targets_path, 'r') as f:
            self.therapeutic_targets = json.load(f)

        # Map feature names to indices
        self.feature_names = ['heartrate', 'respiration', 'sao2']
        self.feature_idx = {name: idx for idx, name in enumerate(self.feature_names)}

        # Therapeutic ranges
        self.targets = {
            'heartrate': self.therapeutic_targets['heartrate']['target_range'],    # [60, 100]
            'respiration': self.therapeutic_targets['respiration']['target_range'], # [12, 20]
            'sao2': self.therapeutic_targets['sao2']['target_range']              # [92, 100]
        }

    def compute_temporal_features(self, X):
        """
        Compute temporal features per window.

        Args:
            X: (N, 24, 3) array

        Returns:
            X_temporal: (N, 24, F_temporal) where F_temporal includes:
                - Original 3 features
                - 1st derivative (rate of change)
                - 2nd derivative (acceleration)
                - Smoothed values (Savitzky-Golay filter)
        """
        N, T, F = X.shape
        features = []

        # Original features
        features.append(X)  # (N, 24, 3)

        # First derivative (rate of change)
        dX = np.full_like(X, np.nan)
        dX[:, 1:, :] = np.diff(X, axis=1)
        features.append(dX)  # (N, 24, 3)

        # Second derivative (acceleration)
        ddX = np.full_like(X, np.nan)
        ddX[:, 2:, :] = np.diff(dX[:, 1:, :], axis=1)
        features.append(ddX)  # (N, 24, 3)

        # Smoothed values (Savitzky-Golay, window=5, order=2)
        X_smooth = np.full_like(X, np.nan)
        for i in range(N):
            for f in range(F):
                valid_idx = ~np.isnan(X[i, :, f])
                if np.sum(valid_idx) >= 5:
                    try:
                        X_smooth[i, valid_idx, f] = savgol_filter(
                            X[i, valid_idx, f], window_length=5, polyorder=2
                        )
                    except:
                        X_smooth[i, valid_idx, f] = X[i, valid_idx, f]
        features.append(X_smooth)  # (N, 24, 3)

        # Concatenate: (N, 24, 3+3+3+3) = (N, 24, 12)
        X_temporal = np.concatenate(features, axis=2)
        return X_temporal

    def compute_statistical_features(self, X):
        """
        Compute cumulative statistical features up to each timestep.
        For each timestep t, compute stats over [0:t+1].

        Args:
            X: (N, 24, 3) array

        Returns:
            X_stats: (N, 24, F_stats) where F_stats includes:
                - Mean over [0:t]
                - Std over [0:t]
                - Min over [0:t]
                - Max over [0:t]
                - Percentiles (25, 50, 75) over [0:t]
                - Range over [0:t]
        """
        N, T, F = X.shape
        features = []

        for t in range(T):
            window = X[:, :t+1, :]  # (N, t+1, 3)

            # Mean
            mean_t = np.nanmean(window, axis=1, keepdims=True)  # (N, 1, 3)

            # Std
            std_t = np.nanstd(window, axis=1, keepdims=True)  # (N, 1, 3)

            # Min
            min_t = np.nanmin(window, axis=1, keepdims=True)  # (N, 1, 3)

            # Max
            max_t = np.nanmax(window, axis=1, keepdims=True)  # (N, 1, 3)

            # Percentiles (25, 50, 75)
            p25_t = np.full((N, 1, F), np.nan)
            p50_t = np.full((N, 1, F), np.nan)
            p75_t = np.full((N, 1, F), np.nan)

            for i in range(N):
                for f in range(F):
                    valid = window[i, :, f]
                    valid = valid[~np.isnan(valid)]
                    if len(valid) > 0:
                        p25_t[i, 0, f] = np.percentile(valid, 25)
                        p50_t[i, 0, f] = np.percentile(valid, 50)
                        p75_t[i, 0, f] = np.percentile(valid, 75)

            # Range
            range_t = max_t - min_t

            # Concatenate at this timestep
            timestep_features = np.concatenate(
                [mean_t, std_t, min_t, max_t, p25_t, p50_t, p75_t, range_t],
                axis=2
            )  # (N, 1, 8*3=24)
            features.append(timestep_features)

        # Stack across timesteps: (N, 24, 24)
        X_stats = np.concatenate(features, axis=1)
        return X_stats

    def compute_therapeutic_deviation(self, X):
        """
        Compute deviation from therapeutic target ranges.

        Args:
            X: (N, 24, 3) array

        Returns:
            deviations: (N, 24, 3) array where each value is:
                - 0 if within target range
                - Distance to range boundary if outside
        """
        N, T, F = X.shape
        deviations = np.full_like(X, np.nan)

        for f, fname in enumerate(self.feature_names):
            target_min, target_max = self.targets[fname]

            for i in range(N):
                for t in range(T):
                    val = X[i, t, f]
                    if not np.isnan(val):
                        if val < target_min:
                            deviations[i, t, f] = target_min - val  # How far below
                        elif val > target_max:
                            deviations[i, t, f] = val - target_max  # How far above
                        else:
                            deviations[i, t, f] = 0.0  # Within range

        return deviations

    def compute_volatility_features(self, X, window_size=5):
        """
        Compute rolling volatility features.

        Args:
            X: (N, 24, 3) array
            window_size: Rolling window size

        Returns:
            volatility: (N, 24, 3) rolling coefficient of variation
        """
        N, T, F = X.shape
        volatility = np.full_like(X, np.nan)

        for i in range(N):
            for f in range(F):
                for t in range(T):
                    # Current window: [max(0, t-window_size+1) : t+1]
                    start = max(0, t - window_size + 1)
                    window = X[i, start:t+1, f]

                    valid = window[~np.isnan(window)]
                    if len(valid) > 1:
                        mean_val = np.mean(valid)
                        std_val = np.std(valid)
                        if mean_val != 0:
                            volatility[i, t, f] = std_val / mean_val  # CV
                        else:
                            volatility[i, t, f] = 0.0

        return volatility

    def compute_entropy_features(self, X):
        """
        Compute Shannon entropy per window as measure of complexity.

        Args:
            X: (N, 24, 3) array

        Returns:
            entropy: (N, 3) Shannon entropy for each feature
        """
        N, T, F = X.shape
        entropy = np.full((N, F), np.nan)

        for i in range(N):
            for f in range(F):
                signal = X[i, :, f]
                valid = signal[~np.isnan(signal)]

                if len(valid) > 1:
                    # Discretize into 10 bins
                    hist, _ = np.histogram(valid, bins=10, density=True)
                    hist = hist[hist > 0]  # Remove zeros
                    entropy[i, f] = -np.sum(hist * np.log2(hist + 1e-10))

        return entropy

    def engineer_features(self, X):
        """
        Create all derived features from raw vital sign tensors.

        Args:
            X: (N, 24, 3) raw vital signs

        Returns:
            X_engineered: (N, 24, F_total) where F_total includes:
                - Original features (3)
                - Temporal: 1st/2nd derivatives, smoothed (12)
                - Statistical: mean, std, min, max, percentiles, range (24)
                - Therapeutic deviation (3)
                - Volatility (3)
                Total: 3 + 12 + 24 + 3 + 3 = 45 per-timestep features

            entropy_features: (N, 3) global entropy per feature
        """
        print("[Feature Engineering] Creating derived features...")

        # Temporal features
        print("  - Computing temporal features (derivative, acceleration)...")
        X_temporal = self.compute_temporal_features(X)

        # Statistical features
        print("  - Computing statistical features (mean, std, percentiles)...")
        X_stats = self.compute_statistical_features(X)

        # Therapeutic deviation
        print("  - Computing therapeutic target deviations...")
        X_therapeutic = self.compute_therapeutic_deviation(X)

        # Volatility
        print("  - Computing volatility (rolling CV)...")
        X_volatility = self.compute_volatility_features(X)

        # Concatenate all features
        # (N, 24, 12) + (N, 24, 24) + (N, 24, 3) + (N, 24, 3) = (N, 24, 42)
        X_engineered = np.concatenate(
            [X_temporal, X_stats, X_therapeutic, X_volatility],
            axis=2
        )

        # Entropy features (static per sample)
        print("  - Computing entropy features...")
        entropy = self.compute_entropy_features(X)

        print(f"  [OK] Feature engineering complete: {X_engineered.shape}")
        print(f"    Per-timestep features: {X_engineered.shape[2]}")
        print(f"    Global entropy features: {entropy.shape}")

        return X_engineered, entropy

    def save_engineered_features(self, X_engineered, entropy, output_dir='data', prefix=''):
        """Save engineered features to disk"""
        print(f"\n[Saving] Engineered features...")

        np.save(f'{output_dir}/{prefix}X_engineered.npy', X_engineered)
        np.save(f'{output_dir}/{prefix}entropy_features.npy', entropy)

        metadata = {
            'X_engineered_shape': X_engineered.shape,
            'entropy_shape': entropy.shape,
            'features_per_timestep': X_engineered.shape[2],
            'timesteps': X_engineered.shape[1],
            'feature_groups': {
                'original': 3,
                'temporal': 12,
                'statistical': 24,
                'therapeutic_deviation': 3,
                'volatility': 3
            }
        }

        import json
        with open(f'{output_dir}/{prefix}feature_engineering_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved: {output_dir}/{prefix}X_engineered.npy")
        print(f"  Saved: {output_dir}/{prefix}entropy_features.npy")
        print(f"  Saved: {output_dir}/{prefix}feature_engineering_metadata.json")


def main():
    """Run feature engineering on eICU and PhysioNet datasets"""
    print("=" * 80)
    print("PHASE 2: FEATURE ENGINEERING")
    print("=" * 80)

    # Load data
    print("\n[1] Loading datasets...")
    from src.icu_data_loader import load_datasets

    X_eicu, X_physio, stats = load_datasets()
    print(f"  X_eicu:   {X_eicu.shape}")
    print(f"  X_physio: {X_physio.shape}")

    # Initialize feature engineer
    fe = FeatureEngineer()

    # Process eICU
    print("\n[2] Processing eICU dataset...")
    X_eicu_eng, entropy_eicu = fe.engineer_features(X_eicu)
    fe.save_engineered_features(X_eicu_eng, entropy_eicu, prefix='eicu_')

    # Process PhysioNet
    print("\n[3] Processing PhysioNet dataset...")
    X_physio_eng, entropy_physio = fe.engineer_features(X_physio)
    fe.save_engineered_features(X_physio_eng, entropy_physio, prefix='physio_')

    print("\n" + "=" * 80)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 80)
    print(f"eICU engineered:  {X_eicu_eng.shape}")
    print(f"PhysioNet engineered: {X_physio_eng.shape}")
    print(f"\nNext: Implement data augmentation and create augmentation pipeline")


if __name__ == "__main__":
    main()
