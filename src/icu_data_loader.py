"""
Usage guide for processed ICU datasets
X_eicu_24h.npy and X_physio_24h.npy
"""

import numpy as np

def load_datasets():
    """
    Load preprocessed ICU datasets.

    Returns:
        X_eicu: numpy array (109837, 24, 3)
            - 109837 patient 24-hour windows
            - 24 timesteps (hours)
            - 3 features: [heartrate, respiration, sao2]
        X_physio: numpy array (116627, 24, 3)
            - 116627 patient 24-hour windows
            - 24 timesteps (hours)
            - 3 features: [heartrate, respiration, sao2]
        stats: dict with 'means' and 'stds' for zscore normalization
    """
    X_eicu = np.load('X_eicu_24h.npy')
    X_physio = np.load('X_physio_24h.npy')
    stats = np.load('normalization_stats.npy', allow_pickle=True).item()

    return X_eicu, X_physio, stats


def handle_missing_values(X, method='forward_fill'):
    """
    Handle NaN values in the tensors.

    Args:
        X: numpy array (N, T, F) with NaN values
        method: 'forward_fill', 'mean', 'interpolate', or 'drop'

    Returns:
        X_filled: array with handled NaNs
    """
    X_filled = X.copy()

    if method == 'forward_fill':
        # Forward fill along time axis
        for i in range(X_filled.shape[0]):
            for f in range(X_filled.shape[2]):
                mask = np.isnan(X_filled[i, :, f])
                idx = np.where(~mask, np.arange(mask.size), 0)
                np.maximum.accumulate(idx, out=idx)
                X_filled[i, :, f][mask] = X_filled[i, idx[mask], f]

    elif method == 'mean':
        # Fill with feature mean (already normalized to ~0)
        X_filled[np.isnan(X_filled)] = 0.0

    elif method == 'interpolate':
        # Linear interpolation
        for i in range(X_filled.shape[0]):
            for f in range(X_filled.shape[2]):
                valid_idx = np.where(~np.isnan(X_filled[i, :, f]))[0]
                if len(valid_idx) > 1:
                    X_filled[i, :, f] = np.interp(
                        np.arange(X_filled.shape[1]),
                        valid_idx,
                        X_filled[i, valid_idx, f],
                        left=X_filled[i, valid_idx[0], f],
                        right=X_filled[i, valid_idx[-1], f]
                    )

    elif method == 'drop':
        # Remove samples with any NaN
        mask = ~np.isnan(X_filled).any(axis=(1, 2))
        X_filled = X_filled[mask]

    return X_filled


# Example usage
if __name__ == "__main__":
    X_eicu, X_physio, stats = load_datasets()

    print("Dataset Info:")
    print(f"  eICU:     {X_eicu.shape} - {100 * np.sum(~np.isnan(X_eicu)) / X_eicu.size:.1f}% valid")
    print(f"  PhysioNet: {X_physio.shape} - {100 * np.sum(~np.isnan(X_physio)) / X_physio.size:.1f}% valid")
    print(f"\nNormalization (z-score):")
    print(f"  heartrate   - mean: {stats['means'][0]:.2f}, std: {stats['stds'][0]:.2f}")
    print(f"  respiration - mean: {stats['means'][1]:.2f}, std: {stats['stds'][1]:.2f}")
    print(f"  sao2        - mean: {stats['means'][2]:.2f}, std: {stats['stds'][2]:.2f}")

    # Example: Handle missing values
    X_eicu_filled = handle_missing_values(X_eicu, method='forward_fill')
    print(f"\nAfter forward-fill: {X_eicu_filled.shape}")
