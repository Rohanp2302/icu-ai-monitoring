"""
Optimized processing of eICU and PhysioNet 2012 ICU datasets
Uses numpy-based operations for efficiency
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def process_eicu_sparse(csv_path, target_features=None):
    """
    Process eICU using optimized dtypes and direct pandas operations.
    """
    if target_features is None:
        target_features = ['heartrate', 'respiration', 'sao2']

    print(f"Loading eICU vitalPeriodic from {csv_path}")

    # Read with optimized dtypes
    dtypes = {
        'patientunitstayid': 'uint32',
        'observationoffset': 'int32',
        'heartrate': 'float32',
        'respiration': 'float32',
        'sao2': 'float32'
    }

    df = pd.read_csv(csv_path, usecols=list(dtypes.keys()), dtype=dtypes)

    print(f"  Loaded shape: {df.shape}")
    print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")

    # Convert to hours
    df['hour'] = (df['observationoffset'] / 60).round(0).astype('int16')
    df = df.drop('observationoffset', axis=1)

    # Aggregate by patient + hour
    print("  Aggregating by patient-hour...")
    df_hourly = df.groupby(['patientunitstayid', 'hour'], as_index=False)[target_features].mean()

    print(f"  Aggregated shape: {df_hourly.shape}")
    print(f"  Unique patients: {df_hourly['patientunitstayid'].nunique()}")
    print(f"  Hour range: {df_hourly['hour'].min()} to {df_hourly['hour'].max()}")

    del df  # Free memory

    return df_hourly


def create_patient_tensors(df, target_features, num_timesteps=24, min_valid_ratio=0.5):
    """
    Convert to (N, T, F) tensors efficiently
    """
    num_features = len(target_features)
    samples = []
    patient_ids = []
    hour_ranges = []

    patients = df['patientunitstayid'].unique()
    print(f"Processing {len(patients)} patients into {num_timesteps}-hour windows...")

    for patient_id in patients:
        patient_data = df[df['patientunitstayid'] == patient_id].copy()
        patient_data = patient_data.sort_values('hour').reset_index(drop=True)

        hours_arr = patient_data['hour'].values
        values = patient_data[target_features].values

        # Sliding window: find all 24-hour windows
        for i in range(len(hours_arr)):
            start_hour = hours_arr[i]
            end_hour = start_hour + num_timesteps - 1

            # Create output tensor
            window_tensor = np.full((num_timesteps, num_features), np.nan, dtype='float32')

            # Fill in available data
            for j in range(i, len(hours_arr)):
                hour = hours_arr[j]
                if hour > end_hour:
                    break
                hour_idx = hour - start_hour
                if 0 <= hour_idx < num_timesteps:
                    window_tensor[hour_idx] = values[j]

            # Check if we have enough valid data
            valid_count = np.sum(~np.isnan(window_tensor))
            min_required = (num_timesteps * num_features) * min_valid_ratio

            if valid_count >= min_required:
                samples.append(window_tensor)
                patient_ids.append(patient_id)
                hour_ranges.append((start_hour, end_hour))

    X = np.array(samples, dtype='float32')
    print(f"  Created {X.shape[0]} samples")
    print(f"  Tensor shape: {X.shape}")
    print(f"  Valid data: {100 * np.sum(~np.isnan(X)) / X.size:.1f}%")

    return X, np.array(patient_ids, dtype='uint32'), hour_ranges


def normalize_tensors_joint(X_list, method='zscore'):
    """
    Normalize multiple tensors on combined statistics.
    """
    num_features = X_list[0].shape[-1]

    # Collect all valid data for fitting
    print("Fitting normalization on combined data...")
    all_valid = []
    for X in X_list:
        X_flat = X.reshape(-1, num_features)
        valid_mask = ~np.isnan(X_flat).any(axis=1)
        all_valid.append(X_flat[valid_mask])

    all_valid = np.vstack(all_valid)
    print(f"  Fit on {all_valid.shape[0]} valid records")

    scaler = StandardScaler()
    scaler.fit(all_valid)

    print(f"  Means: {scaler.mean_.round(2)}")
    print(f"  Stds: {scaler.scale_.round(2)}")

    # Transform each tensor
    X_normalized = []
    for X in X_list:
        shape = X.shape
        X_flat = X.reshape(-1, num_features)

        # Preserve NaN values during normalization
        X_out = np.full_like(X_flat, np.nan, dtype='float32')
        valid_mask = ~np.isnan(X_flat).any(axis=1)
        X_out[valid_mask] = scaler.transform(X_flat[valid_mask])

        X_normalized.append(X_out.reshape(shape))

    stats = {
        'means': scaler.mean_.astype('float32'),
        'stds': scaler.scale_.astype('float32')
    }

    return X_normalized, scaler, stats


def main():
    print("=" * 70)
    print("OPTIMIZED ICU DATASET PROCESSING")
    print("=" * 70)

    # Process eICU
    print("\n" + "=" * 70)
    print("PROCESSING eICU (vitalPeriodic)")
    print("=" * 70)
    df_eicu = process_eicu_sparse('data/raw/eicu/vitalPeriodic.csv')

    # Load PhysioNet
    print("\n" + "=" * 70)
    print("LOADING PhysioNet 2012 (hourly)")
    print("=" * 70)
    print("Loading processed_icu_hourly_v2.csv")
    dtype_physio = {
        'patientunitstayid': 'uint32',
        'hour': 'int16',
        'sao2': 'float32',
        'heartrate': 'float32',
        'respiration': 'float32'
    }
    df_physio = pd.read_csv(
        'data/processed_icu_hourly_v2.csv',
        usecols=list(dtype_physio.keys()),
        dtype=dtype_physio
    )
    print(f"  Shape: {df_physio.shape}")
    print(f"  Unique patients: {df_physio['patientunitstayid'].nunique()}")

    target_features = ['heartrate', 'respiration', 'sao2']

    # Create tensors
    print("\n" + "=" * 70)
    print("CONVERTING TO TENSORS")
    print("=" * 70)

    print("\neICU tensors:")
    X_eicu, eicu_pts, eicu_ranges = create_patient_tensors(
        df_eicu, target_features, num_timesteps=24
    )

    print("\nPhysioNet tensors:")
    X_physio, physio_pts, physio_ranges = create_patient_tensors(
        df_physio, target_features, num_timesteps=24
    )

    # Normalize
    print("\n" + "=" * 70)
    print("JOINT NORMALIZATION")
    print("=" * 70)
    [X_eicu, X_physio], scaler, stats = normalize_tensors_joint(
        [X_eicu, X_physio], method='zscore'
    )

    # Results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"X_eicu shape:   {X_eicu.shape}")
    print(f"X_physio shape: {X_physio.shape}")
    print(f"Features: {target_features}")

    for name, X in [("eICU", X_eicu), ("PhysioNet", X_physio)]:
        valid_pct = 100 * np.sum(~np.isnan(X)) / X.size
        print(f"{name:12} - {valid_pct:5.1f}% valid")

    # Save
    print("\n" + "=" * 70)
    print("SAVING")
    print("=" * 70)
    np.save('X_eicu_24h.npy', X_eicu)
    np.save('X_physio_24h.npy', X_physio)
    np.save('normalization_stats.npy', stats, allow_pickle=True)

    print("[+] Saved X_eicu_24h.npy")
    print("[+] Saved X_physio_24h.npy")
    print("[+] Saved normalization_stats.npy")

    print("\n" + "=" * 70)
    print("NORMALIZATION STATS")
    print("=" * 70)
    for i, feat in enumerate(target_features):
        print(f"{feat:15} Mean: {stats['means'][i]:8.3f}  Std: {stats['stds'][i]:8.3f}")

    return X_eicu, X_physio, stats


if __name__ == "__main__":
    X_eicu, X_physio, stats = main()
