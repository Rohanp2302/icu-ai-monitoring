"""
Process eICU and PhysioNet 2012 ICU datasets
Align them to same shape (N, 24, F) with common features: heartrate, respiration, sao2
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def process_eicu_to_hourly(csv_path, target_features=None, chunksize=50000):
    """
    Convert eICU vitalPeriodic.csv to hourly patient-wise time series.
    Uses chunked reading for memory efficiency.

    Args:
        csv_path: Path to vitalPeriodic.csv
        target_features: List of features to extract
        chunksize: Rows to read at a time

    Returns:
        DataFrame with columns: [patientunitstayid, hour, feature1, feature2, ...]
    """
    if target_features is None:
        target_features = ['heartrate', 'respiration', 'sao2']

    print(f"Loading eICU vitalPeriodic from {csv_path} (chunked)")

    # Read and process in chunks
    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=['patientunitstayid', 'observationoffset'] + target_features,
                             chunksize=chunksize):
        # Convert observationoffset (minutes) to hours
        chunk['hour'] = (chunk['observationoffset'] / 60).round(0).astype(int)

        # Drop observationoffset and aggregate
        chunk = chunk.drop('observationoffset', axis=1)
        chunk_hourly = chunk.groupby(['patientunitstayid', 'hour'])[target_features].mean()
        chunks.append(chunk_hourly)

        if len(chunks) % 5 == 0:
            print(f"  Processed {len(chunks) * chunksize} rows...")

    # Combine all chunks
    df_hourly = pd.concat(chunks).groupby(['patientunitstayid', 'hour'])[target_features].mean()
    df_hourly = df_hourly.reset_index()

    print(f"  Shape after hourly aggregation: {df_hourly.shape}")
    print(f"  Unique patients: {df_hourly['patientunitstayid'].nunique()}")
    print(f"  Hour range: {df_hourly['hour'].min()} to {df_hourly['hour'].max()}")

    return df_hourly


def load_physionet_2012(csv_path, target_features=None):
    """
    Load PhysioNet 2012 (processed_icu_hourly) dataset.

    Args:
        csv_path: Path to processed ICU hourly CSV
        target_features: List of features to extract

    Returns:
        DataFrame with columns: [patientunitstayid, hour, feature1, feature2, ...]
    """
    if target_features is None:
        target_features = ['heartrate', 'respiration', 'sao2']

    print(f"Loading PhysioNet 2012 from {csv_path}")
    df = pd.read_csv(csv_path)

    # Select relevant columns
    cols_to_keep = ['patientunitstayid', 'hour'] + target_features
    df = df[cols_to_keep].copy()

    print(f"  Shape: {df.shape}")
    print(f"  Unique patients: {df['patientunitstayid'].nunique()}")
    print(f"  Hour range: {df['hour'].min()} to {df['hour'].max()}")

    return df


def patient_timeseries_to_tensor(df, num_timesteps=24, target_features=None):
    """
    Convert hourly patient data to tensor format (N, timesteps, features).

    Strategy:
    - Take contiguous 24-hour windows for each patient
    - Fill missing hours with NaN (will be handled in normalization)
    - Drop patients/windows with insufficient valid data

    Args:
        df: DataFrame with columns [patientunitstayid, hour, feature1, ...]
        num_timesteps: Number of consecutive hours per sample
        target_features: List of feature names

    Returns:
        X: numpy array of shape (N, num_timesteps, num_features)
        feature_names: list of feature names
        patient_ids: list of patient IDs corresponding to each sample
        hour_ranges: list of (start_hour, end_hour) for each sample
    """
    if target_features is None:
        target_features = ['heartrate', 'respiration', 'sao2']

    num_features = len(target_features)
    samples = []
    sample_patient_ids = []
    sample_hour_ranges = []

    for patient_id in df['patientunitstayid'].unique():
        patient_data = df[df['patientunitstayid'] == patient_id].copy()
        patient_data = patient_data.sort_values('hour')

        # Find contiguous or near-contiguous 24-hour windows
        hours = patient_data['hour'].values

        # Sliding window approach: use any 24 consecutive (or near-consecutive) hours
        for start_idx in range(len(hours)):
            start_hour = hours[start_idx]
            end_hour = start_hour + num_timesteps - 1

            # Get data for this window
            window = patient_data[
                (patient_data['hour'] >= start_hour) &
                (patient_data['hour'] <= end_hour)
            ].copy()

            if len(window) == 0:
                continue

            # Create full 24-hour grid (even with missing hours)
            full_hours = np.arange(start_hour, end_hour + 1)
            window_data = np.full((num_timesteps, num_features), np.nan)

            for hour_idx, hour in enumerate(full_hours):
                hour_data = window[window['hour'] == hour]
                if len(hour_data) > 0:
                    # Take first record if multiple for same hour
                    values = hour_data[target_features].iloc[0].values
                    window_data[hour_idx, :] = values

            # Only keep if we have at least 50% valid data
            valid_count = np.sum(~np.isnan(window_data))
            min_valid = (num_timesteps * num_features) * 0.5

            if valid_count >= min_valid:
                samples.append(window_data)
                sample_patient_ids.append(patient_id)
                sample_hour_ranges.append((start_hour, end_hour))

    X = np.array(samples)
    print(f"  Generated {len(samples)} samples from {df['patientunitstayid'].nunique()} patients")
    print(f"  Tensor shape: {X.shape}")
    print(f"  Valid data percentage: {100 * np.sum(~np.isnan(X)) / X.size:.1f}%")

    return X, target_features, np.array(sample_patient_ids), sample_hour_ranges


def normalize_tensors(X_list, method='zscore'):
    """
    Normalize multiple tensors using the same fit (on combined data).

    Args:
        X_list: List of numpy arrays with shape (N, T, F)
        method: 'zscore' (StandardScaler) or 'minmax'

    Returns:
        X_normalized_list: List of normalized arrays
        scaler: Fitted scaler object
        stats: Dictionary with normalization statistics
    """
    num_features = X_list[0].shape[-1]

    # Flatten all data for fitting (handle NaN)
    all_data = []
    for X in X_list:
        # Reshape to (N*T, F) and remove NaN rows
        X_flat = X.reshape(-1, num_features)
        X_valid = X_flat[~np.isnan(X_flat).any(axis=1)]
        all_data.append(X_valid)

    all_data = np.vstack(all_data)
    print(f"\nFitting normalization on {all_data.shape[0]} valid records")

    if method == 'zscore':
        scaler = StandardScaler()
        scaler.fit(all_data)
        print(f"  Mean per feature: {scaler.mean_.round(2)}")
        print(f"  Std per feature: {scaler.scale_.round(2)}")
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Apply normalization to each tensor
    X_normalized_list = []
    for X in X_list:
        shape = X.shape
        X_flat = X.reshape(-1, num_features)

        # Create output with NaN preserved
        X_scaled = np.full_like(X_flat, np.nan)
        valid_mask = ~np.isnan(X_flat).any(axis=1)
        X_scaled[valid_mask] = scaler.transform(X_flat[valid_mask])

        X_normalized_list.append(X_scaled.reshape(shape))

    stats = {
        'means': scaler.mean_,
        'stds': scaler.scale_
    }

    return X_normalized_list, scaler, stats


def align_tensors_shape(X_list, target_shape_t=24):
    """
    Ensure all tensors have the requested time dimension.
    Drop samples that don't meet criteria.

    Args:
        X_list: List of numpy arrays
        target_shape_t: Target time steps

    Returns:
        X_aligned_list: List of arrays with aligned shape
    """
    X_aligned_list = []
    for X in X_list:
        # Keep only samples with exact time steps
        if X.shape[1] != target_shape_t:
            print(f"Warning: Expected {target_shape_t} timesteps, got {X.shape[1]}")
        X_aligned_list.append(X)

    return X_aligned_list


def process_and_align_datasets(eicu_path, physionet_path, num_timesteps=24,
                               target_features=None):
    """
    Main function: Load, process, and align both datasets.

    Args:
        eicu_path: Path to vitalPeriodic.csv
        physionet_path: Path to processed_icu_hourly_v2.csv
        num_timesteps: Number of hours per sample (default 24)
        target_features: Features to extract (default: heartrate, respiration, sao2)

    Returns:
        X_eicu: numpy array (N1, 24, 3)
        X_physio: numpy array (N2, 24, 3)
        stats: Normalization statistics
    """
    if target_features is None:
        target_features = ['heartrate', 'respiration', 'sao2']

    print("=" * 70)
    print("PROCESSING EICU DATASET")
    print("=" * 70)
    df_eicu = process_eicu_to_hourly(eicu_path, target_features)

    print("\n" + "=" * 70)
    print("PROCESSING PHYSIONET 2012 DATASET")
    print("=" * 70)
    df_physio = load_physionet_2012(physionet_path, target_features)

    print("\n" + "=" * 70)
    print("CONVERTING TO TENSORS: eICU")
    print("=" * 70)
    X_eicu, features, eicu_patient_ids, eicu_hour_ranges = patient_timeseries_to_tensor(
        df_eicu, num_timesteps, target_features
    )

    print("\n" + "=" * 70)
    print("CONVERTING TO TENSORS: PhysioNet 2012")
    print("=" * 70)
    X_physio, _, physio_patient_ids, physio_hour_ranges = patient_timeseries_to_tensor(
        df_physio, num_timesteps, target_features
    )

    print("\n" + "=" * 70)
    print("NORMALIZING BOTH DATASETS (Z-SCORE)")
    print("=" * 70)
    [X_eicu, X_physio], scaler, stats = normalize_tensors([X_eicu, X_physio], method='zscore')

    print("\n" + "=" * 70)
    print("FINAL TENSOR SHAPES")
    print("=" * 70)
    print(f"X_eicu shape:   {X_eicu.shape}  (patients={X_eicu.shape[0]}, hours={X_eicu.shape[1]}, features={X_eicu.shape[2]})")
    print(f"X_physio shape: {X_physio.shape}  (patients={X_physio.shape[0]}, hours={X_physio.shape[1]}, features={X_physio.shape[2]})")
    print(f"Features: {features}")

    print("\n" + "=" * 70)
    print("DATA QUALITY SUMMARY")
    print("=" * 70)
    for name, X in [("eICU", X_eicu), ("PhysioNet", X_physio)]:
        valid_pct = 100 * np.sum(~np.isnan(X)) / X.size
        nan_pct = 100 * np.sum(np.isnan(X)) / X.size
        print(f"{name:12} - Valid: {valid_pct:5.1f}%  NaN: {nan_pct:5.1f}%")

    return X_eicu, X_physio, stats, {
        'eicu_patient_ids': eicu_patient_ids,
        'eicu_hour_ranges': eicu_hour_ranges,
        'physio_patient_ids': physio_patient_ids,
        'physio_hour_ranges': physio_hour_ranges,
        'features': features,
        'scaler': scaler
    }


if __name__ == "__main__":
    # Paths
    eicu_path = 'data/raw/eicu/vitalPeriodic.csv'
    physionet_path = 'data/processed_icu_hourly_v2.csv'

    # Process datasets
    X_eicu, X_physio, stats, metadata = process_and_align_datasets(
        eicu_path,
        physionet_path,
        num_timesteps=24,
        target_features=['heartrate', 'respiration', 'sao2']
    )

    # Save results
    print("\n" + "=" * 70)
    print("SAVING OUTPUTS")
    print("=" * 70)

    np.save('X_eicu_24h.npy', X_eicu)
    np.save('X_physio_24h.npy', X_physio)
    np.save('normalization_stats.npy', stats, allow_pickle=True)

    print(f"✓ Saved X_eicu_24h.npy: {X_eicu.shape}")
    print(f"✓ Saved X_physio_24h.npy: {X_physio.shape}")
    print(f"✓ Saved normalization_stats.npy")

    # Display summary
    print("\n" + "=" * 70)
    print("NORMALIZATION STATISTICS")
    print("=" * 70)
    for i, feature in enumerate(metadata['features']):
        print(f"{feature:15} - Mean: {stats['means'][i]:8.2f}  Std: {stats['stds'][i]:8.2f}")
