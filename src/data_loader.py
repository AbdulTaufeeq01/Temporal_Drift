"""
Data Loading and Time-Window Splitting Module

Handles loading datasets, creating chronological windows for temporal analysis,
and splitting into reference and test sets.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Load and preprocess datasets for temporal drift analysis."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the data loader.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        np.random.seed(random_state)
        
    def load_synthetic_data(
        self,
        n_samples_per_window: int,
        n_features: int,
        n_windows: int,
        drift_magnitude: float = 0.3,
        drift_type: str = "gradual",
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Generate synthetic dataset with temporal drift.
        
        Args:
            n_samples_per_window: Samples per time window
            n_features: Number of features
            n_windows: Number of time windows
            drift_magnitude: Magnitude of drift (0.0-1.0)
            drift_type: "gradual" or "abrupt"
            
        Returns:
            DataFrame with data, array of window indices
        """
        logger.info(f"Generating synthetic data: {n_windows} windows, "
                   f"{n_samples_per_window} samples each, drift={drift_magnitude}")
        
        data_list = []
        window_indices = []
        target_values = []
        
        # Base distribution parameters
        base_mean = np.zeros(n_features)
        base_cov = np.eye(n_features)
        
        for window_idx in range(n_windows):
            # Calculate drift for this window
            if drift_type == "gradual":
                drift_scale = drift_magnitude * (window_idx / n_windows)
            elif drift_type == "abrupt":
                drift_scale = drift_magnitude if window_idx >= n_windows // 2 else 0
            else:
                drift_scale = 0
            
            # Generate features with drift
            drifted_mean = base_mean + (drift_scale * np.random.randn(n_features))
            X = np.random.multivariate_normal(
                drifted_mean, base_cov, n_samples_per_window
            )
            
            # Generate binary target based on first few features
            y = (X[:, 0] + X[:, 1] + drift_scale > 0).astype(int)
            
            data_list.append(X)
            window_indices.extend([window_idx] * n_samples_per_window)
            target_values.extend(y)
        
        # Combine all data
        X_full = np.vstack(data_list)
        feature_names = [f"feature_{i}" for i in range(n_features)]
        df = pd.DataFrame(X_full, columns=feature_names)
        df['window'] = window_indices
        df['target'] = target_values
        
        logger.info(f"Generated synthetic data: shape={df.shape}")
        return df, np.array(window_indices)
    
    def load_diabetes_dataset(self) -> pd.DataFrame:
        """Load diabetes readmission dataset from UCI."""
        try:
            from sklearn.datasets import fetch_openml
            df = fetch_openml(
                name='diabetes_130-us_hospitals_for_years_1999-2008',
                version=1,
                as_frame=True,
                parser='auto'
            ).frame
            logger.info(f"Loaded diabetes dataset: shape={df.shape}")
            return df
        except Exception as e:
            logger.error(f"Failed to load diabetes dataset: {e}")
            raise
    
    def split_into_windows(
        self,
        df: pd.DataFrame,
        n_windows: int,
        time_column: Optional[str] = None,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Split data into chronological windows.
        
        Args:
            df: Input dataframe
            n_windows: Number of time windows
            time_column: Column name for time ordering (if None, uses index order)
            
        Returns:
            List of (X, y) tuples for each window
        """
        if 'target' not in df.columns:
            raise ValueError("DataFrame must contain 'target' column")
        
        if time_column is None:
            # Use sequential order
            window_size = len(df) // n_windows
            windows = []
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = (i + 1) * window_size if i < n_windows - 1 else len(df)
                window_data = df.iloc[start_idx:end_idx]
                windows.append((window_data, i))
        else:
            # Sort by time column and split
            df_sorted = df.sort_values(time_column).reset_index(drop=True)
            window_size = len(df_sorted) // n_windows
            windows = []
            for i in range(n_windows):
                start_idx = i * window_size
                end_idx = (i + 1) * window_size if i < n_windows - 1 else len(df_sorted)
                window_data = df_sorted.iloc[start_idx:end_idx]
                windows.append((window_data, i))
        
        logger.info(f"Split data into {n_windows} windows")
        return windows
    
    def get_ref_and_test_splits(
        self,
        windows: List[Tuple[pd.DataFrame, int]],
        ref_window_idx: int = 0,
    ) -> List[Dict[str, np.ndarray]]:
        """
        Create reference and test splits for each window.
        
        Args:
            windows: List of (window_data, window_idx) tuples
            ref_window_idx: Index of reference window for drift detection
            
        Returns:
            List of dicts with keys: X_ref, y_ref, X_test, y_test, window_idx
        """
        ref_data, _ = windows[ref_window_idx]
        ref_features = [c for c in ref_data.columns if c != 'target']
        
        X_ref = ref_data[ref_features].values
        y_ref = ref_data['target'].values
        
        splits = []
        for test_data, window_idx in windows:
            X_test = test_data[ref_features].values
            y_test = test_data['target'].values
            
            splits.append({
                'X_ref': X_ref,
                'y_ref': y_ref,
                'X_test': X_test,
                'y_test': y_test,
                'window_idx': window_idx,
                'feature_names': ref_features,
            })
        
        logger.info(f"Created {len(splits)} ref-test splits")
        return splits
    
    def standardize_splits(
        self,
        splits: List[Dict[str, np.ndarray]],
    ) -> List[Dict[str, np.ndarray]]:
        """
        Standardize features using reference window.
        
        Args:
            splits: List of ref-test split dicts
            
        Returns:
            List of splits with standardized values
        """
        X_ref = splits[0]['X_ref']
        scaler = StandardScaler()
        scaler.fit(X_ref)
        
        standardized_splits = []
        for split in splits:
            split_copy = split.copy()
            split_copy['X_ref'] = scaler.transform(split['X_ref'])
            split_copy['X_test'] = scaler.transform(split['X_test'])
            standardized_splits.append(split_copy)
        
        logger.info("Standardized all splits using reference window scaler")
        return standardized_splits


class SyntheticDriftDataset:
    """Generate synthetic datasets with controlled drift patterns."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)
    
    def generate(
        self,
        n_samples_per_window: int,
        n_features: int,
        n_windows: int,
        drift_magnitude: float = 0.3,
        drift_type: str = "gradual",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic drift data.
        
        Returns:
            X: Feature matrix (n_samples_total, n_features)
            y: Target vector (n_samples_total,)
            window_ids: Window assignment for each sample
        """
        loader = DataLoader(random_state=self.random_state)
        df, window_ids = loader.load_synthetic_data(
            n_samples_per_window=n_samples_per_window,
            n_features=n_features,
            n_windows=n_windows,
            drift_magnitude=drift_magnitude,
            drift_type=drift_type,
        )
        
        feature_cols = [c for c in df.columns if c not in ['window', 'target']]
        X = df[feature_cols].values
        y = df['target'].values
        
        return X, y, window_ids
