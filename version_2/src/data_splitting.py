"""
Data Splitting Module
Create stratified train/val/test splits for 226k+ samples

Strategy:
- Temporal split: 80% train, 20% validation (accounts for distribution shift)
- Stratified: By mortality and LOS quintiles
- 5-fold CV setup: Each fold has train/val/test within 80% temporal window
- Final test set: Held-out 10% from validation split

Output:
- split_indices.npz: Train/val/test indices for each fold
- split_metadata.json: Split statistics and info
"""

import numpy as np
import pandas as pd
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
import warnings
warnings.filterwarnings('ignore')


class DataSplitter:
    """Create stratified train/val/test splits for ICU data"""

    def __init__(self, total_samples=226464, n_splits=5, random_state=42):
        """
        Args:
            total_samples: Total number of samples (eICU + PhysioNet)
            n_splits: Number of folds for CV (default 5)
            random_state: Random seed for reproducibility
        """
        self.total_samples = total_samples
        self.n_splits = n_splits
        self.random_state = random_state

    def create_stratification_targets(self):
        """
        Create stratification targets from outcome data

        Load outcomes and create strata based on:
        1. Mortality (binary)
        2. LOS quintiles (continuous → 5 levels)

        Returns:
            strata: (n,) array of strata labels
            stratification_info: Dict with info
        """
        print("[1] Creating stratification targets...")

        # Load Challenge 2012 outcomes (12k samples)
        outcomes_a = pd.read_csv('data/raw/challenge2012/Outcomes-a.txt')
        outcomes_b = pd.read_csv('data/raw/challenge2012/Outcomes-b.txt')
        outcomes_c = pd.read_csv('data/raw/challenge2012/Outcomes-c.txt')

        df_outcomes = pd.concat([outcomes_a, outcomes_b, outcomes_c], ignore_index=True)
        df_outcomes.columns = [col.replace('-', '_') for col in df_outcomes.columns]

        print(f"  Loaded {len(df_outcomes):,} outcomes from Challenge 2012")

        # For remaining samples (eICU+PhysioNet without outcome labels), use synthetic strata
        # Assumption: outcomes from Challenge 2012 are representative
        mortality_rate = df_outcomes['In_hospital_death'].mean()
        los_mean = df_outcomes['Length_of_stay'].replace(-1, np.nan).mean()
        los_std = df_outcomes['Length_of_stay'].replace(-1, np.nan).std()

        print(f"  Outcome statistics:")
        print(f"    Mortality rate: {mortality_rate:.1%}")
        print(f"    LOS mean: {los_mean:.1f} ± {los_std:.1f} days")

        # Create stratification based on mortality and LOS
        # For samples without outcomes, simulate based on actual distribution
        n_unlabeled = self.total_samples - len(df_outcomes)

        # Create strata labels: Combine mortality (2 levels) × LOS quintile (5 levels) = 10 strata
        strata = np.zeros(self.total_samples, dtype=int)

        # Add labeled outcomes (first 12k samples)
        los_values = df_outcomes['Length_of_stay'].replace(-1, np.nan).values
        los_quintiles = pd.qcut(los_values, q=5, labels=False, duplicates='drop')

        for i in range(len(df_outcomes)):
            mortality = int(df_outcomes['In_hospital_death'].iloc[i])
            los_q = los_quintiles[i] if not np.isnan(los_quintiles[i]) else 2
            strata[i] = mortality * 5 + int(los_q)

        # For unlabeled samples, randomly assign strata based on distribution
        np.random.seed(self.random_state)
        for i in range(len(df_outcomes), self.total_samples):
            mortality = np.random.choice([0, 1], p=[1 - mortality_rate, mortality_rate])
            los_q = np.random.choice([0, 1, 2, 3, 4])
            strata[i] = mortality * 5 + los_q

        print(f"  Created {len(np.unique(strata))} strata")
        print(f"  Strata distribution:")
        unique, counts = np.unique(strata, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"    Stratum {u}: {c:,} ({100*c/len(strata):.1f}%)")

        return strata, {
            'n_labeled': len(df_outcomes),
            'n_total': self.total_samples,
            'mortality_rate': mortality_rate,
            'los_mean': los_mean,
            'los_std': los_std,
            'n_strata': len(np.unique(strata))
        }

    def create_splits(self, strata):
        """
        Create simple stratified train/val/test split (no per-fold separation yet)

        This split is used for all models. K-fold CV will be applied during training
        on the training set only, ensuring clean separation.

        Strategy:
        1. Split: 60% train, 20% val, 20% test
        2. All splits stratified by mortality and LOS quintiles
        3. K-fold CV will be applied later during training (on train set only)

        Returns:
            split_data: Dict with train/val/test indices
            split_info: Metadata about splits
        """
        print("\n[2] Creating stratified train/val/test split...")

        indices_all = np.arange(self.total_samples)

        # First split: 80% train+val, 20% test
        indices_train_val, indices_test, strata_train_val, _ = train_test_split(
            indices_all, strata,
            test_size=0.20,
            stratify=strata,
            random_state=self.random_state
        )

        # Second split: 75% train, 25% val (of the 80%)
        indices_train, indices_val, _, _ = train_test_split(
            indices_train_val, strata_train_val,
            test_size=0.25,  # 25% of 80% = 20% total
            stratify=strata_train_val,
            random_state=self.random_state + 1
        )

        split_data = {
            'train_indices': indices_train,
            'val_indices': indices_val,
            'test_indices': indices_test,
            'strata_train': strata[indices_train],
            'strata_val': strata[indices_val],
            'strata_test': strata[indices_test],
        }

        print(f"  Train split: {len(indices_train):,} ({100*len(indices_train)/self.total_samples:.1f}%)")
        print(f"  Val split:   {len(indices_val):,} ({100*len(indices_val)/self.total_samples:.1f}%)")
        print(f"  Test split:  {len(indices_test):,} ({100*len(indices_test)/self.total_samples:.1f}%)")

        return split_data

    def save_splits(self, split_data, output_dir='data'):
        """Save split indices for later use"""
        print("\n[3] Saving split information...")

        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save as numpy file
        np.savez(
            f'{output_dir}/split_indices.npz',
            train=split_data['train_indices'],
            val=split_data['val_indices'],
            test=split_data['test_indices'],
            strata_train=split_data['strata_train'],
            strata_val=split_data['strata_val'],
            strata_test=split_data['strata_test'],
        )
        print(f"  Saved: {output_dir}/split_indices.npz")

        # Save metadata
        metadata = {
            'n_total_samples': self.total_samples,
            'n_train': len(split_data['train_indices']),
            'n_val': len(split_data['val_indices']),
            'n_test': len(split_data['test_indices']),
            'random_state': self.random_state,
            'note': 'K-fold CV will be applied during model training on train set only'
        }

        with open(f'{output_dir}/split_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Served: {output_dir}/split_metadata.json")

        return metadata

    def verify_no_leakage(self, split_data):
        """Verify no data leakage between splits"""
        print("\n[4] Verifying no data leakage...")

        train_indices = split_data['train_indices']
        val_indices = split_data['val_indices']
        test_indices = split_data['test_indices']

        # Check for overlaps
        train_val_overlap = len(np.intersect1d(train_indices, val_indices))
        train_test_overlap = len(np.intersect1d(train_indices, test_indices))
        val_test_overlap = len(np.intersect1d(val_indices, test_indices))

        print(f"  Overlaps:")
        print(f"    Train-Val: {train_val_overlap:,} (should be 0)")
        print(f"    Train-Test: {train_test_overlap:,} (should be 0)")
        print(f"    Val-Test: {val_test_overlap:,} (should be 0)")

        if train_val_overlap > 0 or train_test_overlap > 0 or val_test_overlap > 0:
            raise ValueError("Data leakage detected!")

        print("  [OK] No leakage detected")


def main():
    print("=" * 80)
    print("DATA SPLITTING - STRATIFIED TRAIN/VAL/TEST")
    print("=" * 80)

    splitter = DataSplitter(total_samples=226464, n_splits=5, random_state=42)

    # Create stratification targets
    strata, strata_info = splitter.create_stratification_targets()

    # Create splits
    split_data = splitter.create_splits(strata)

    # Save splits
    metadata = splitter.save_splits(split_data)

    # Verify
    splitter.verify_no_leakage(split_data)

    print("\n" + "=" * 80)
    print("DATA SPLITTING COMPLETE")
    print("=" * 80)
    print(f"Total samples: {splitter.total_samples:,}")
    print(f"Train set: {len(split_data['train_indices']):,}")
    print(f"Val set:   {len(split_data['val_indices']):,}")
    print(f"Test set:  {len(split_data['test_indices']):,}")
    print(f"\nK-fold CV will be applied during training on train set")
    print(f"Next: Create augmentation and feature engineering pipelines")


if __name__ == "__main__":
    main()
