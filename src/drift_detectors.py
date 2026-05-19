"""
Drift Detection Module

Implements multiple statistical divergence measures for detecting data drift:
- KL Divergence (Kullback-Leibler)
- Population Stability Index (PSI)
- Kolmogorov-Smirnov Test (KS Test)
- Jensen-Shannon Divergence
- Wasserstein Distance
"""

import numpy as np
import pandas as pd
from scipy.stats import entropy, ks_2samp, wasserstein_distance
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class DriftDetector:
    """Statistical drift detection for feature distributions."""
    
    def __init__(self, n_bins: int = 10, epsilon: float = 1e-10):
        """
        Initialize drift detector.
        
        Args:
            n_bins: Number of bins for discretizing continuous features
            epsilon: Small value to avoid log(0) in divergence calculations
        """
        self.n_bins = n_bins
        self.epsilon = epsilon
    
    def discretize_feature(
        self,
        X_ref: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Discretize continuous features into bins.
        
        Args:
            X_ref: Reference feature values (1D array)
            X_test: Test feature values (1D array)
            
        Returns:
            Binned reference and test arrays
        """
        # Create bins based on reference data
        bins = np.linspace(X_ref.min(), X_ref.max(), self.n_bins + 1)
        
        X_ref_binned = np.digitize(X_ref, bins)
        X_test_binned = np.digitize(X_test, bins)
        
        return X_ref_binned, X_test_binned
    
    def kl_divergence(
        self,
        X_ref: np.ndarray,
        X_test: np.ndarray,
    ) -> float:
        """
        Calculate KL(P_ref || P_test) divergence.
        
        Lower values indicate less drift. Uses discretization for continuous features.
        
        Args:
            X_ref: Reference feature values
            X_test: Test feature values
            
        Returns:
            KL divergence value (non-negative)
        """
        # Discretize features
        X_ref_bin, X_test_bin = self.discretize_feature(X_ref, X_test)
        
        # Calculate probability distributions
        ref_counts = np.bincount(X_ref_bin, minlength=self.n_bins + 2)
        test_counts = np.bincount(X_test_bin, minlength=self.n_bins + 2)
        
        P = (ref_counts + self.epsilon) / (ref_counts.sum() + self.epsilon * len(ref_counts))
        Q = (test_counts + self.epsilon) / (test_counts.sum() + self.epsilon * len(test_counts))
        
        # KL divergence using scipy.stats.entropy
        # entropy(pk, qk) returns sum(pk * log(pk/qk))
        kl = entropy(P, Q)
        
        return float(np.nan_to_num(kl, nan=0.0, posinf=1.0))
    
    def psi(
        self,
        X_ref: np.ndarray,
        X_test: np.ndarray,
    ) -> float:
        """
        Calculate Population Stability Index (PSI).
        
        PSI < 0.1: No drift
        PSI 0.1-0.25: Moderate drift
        PSI > 0.25: Significant drift
        
        Args:
            X_ref: Reference feature values
            X_test: Test feature values
            
        Returns:
            PSI value (non-negative)
        """
        # Discretize features
        X_ref_bin, X_test_bin = self.discretize_feature(X_ref, X_test)
        
        # Calculate probability distributions
        ref_counts = np.bincount(X_ref_bin, minlength=self.n_bins + 2)
        test_counts = np.bincount(X_test_bin, minlength=self.n_bins + 2)
        
        P = (ref_counts + self.epsilon) / (ref_counts.sum() + self.epsilon * len(ref_counts))
        Q = (test_counts + self.epsilon) / (test_counts.sum() + self.epsilon * len(test_counts))
        
        # PSI = sum((P - Q) * ln(P/Q))
        psi_value = np.sum((P - Q) * np.log(P / Q))
        
        return float(np.nan_to_num(psi_value, nan=0.0, posinf=1.0))
    
    def ks_test(
        self,
        X_ref: np.ndarray,
        X_test: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for distribution difference.
        
        Args:
            X_ref: Reference feature values
            X_test: Test feature values
            
        Returns:
            (KS statistic, p-value)
        """
        statistic, p_value = ks_2samp(X_ref, X_test)
        return float(statistic), float(p_value)
    
    def js_divergence(
        self,
        X_ref: np.ndarray,
        X_test: np.ndarray,
    ) -> float:
        """
        Jensen-Shannon divergence (symmetric KL divergence).
        
        Args:
            X_ref: Reference feature values
            X_test: Test feature values
            
        Returns:
            JS divergence value
        """
        # Discretize features
        X_ref_bin, X_test_bin = self.discretize_feature(X_ref, X_test)
        
        # Calculate probability distributions
        ref_counts = np.bincount(X_ref_bin, minlength=self.n_bins + 2)
        test_counts = np.bincount(X_test_bin, minlength=self.n_bins + 2)
        
        P = (ref_counts + self.epsilon) / (ref_counts.sum() + self.epsilon * len(ref_counts))
        Q = (test_counts + self.epsilon) / (test_counts.sum() + self.epsilon * len(test_counts))
        
        # JS = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5 * (P + Q)
        M = 0.5 * (P + Q)
        js = 0.5 * entropy(P, M) + 0.5 * entropy(Q, M)
        
        return float(np.nan_to_num(js, nan=0.0, posinf=1.0))
    
    def wasserstein(
        self,
        X_ref: np.ndarray,
        X_test: np.ndarray,
    ) -> float:
        """
        Wasserstein distance between two distributions.
        
        Args:
            X_ref: Reference feature values
            X_test: Test feature values
            
        Returns:
            Wasserstein distance
        """
        distance = wasserstein_distance(X_ref, X_test)
        return float(distance)
    
    def detect_drift_all_metrics(
        self,
        X_ref: np.ndarray,
        X_test: np.ndarray,
        metrics: Optional[list] = None,
    ) -> Dict[str, float]:
        """
        Calculate all drift metrics for a feature.
        
        Args:
            X_ref: Reference feature values
            X_test: Test feature values
            metrics: List of metrics to compute (default: all)
            
        Returns:
            Dictionary with metric names and values
        """
        if metrics is None:
            metrics = ['psi', 'kl', 'ks_test', 'js', 'wasserstein']
        
        results = {}
        
        if 'psi' in metrics:
            results['psi'] = self.psi(X_ref, X_test)
        
        if 'kl' in metrics:
            results['kl'] = self.kl_divergence(X_ref, X_test)
        
        if 'ks_test' in metrics:
            ks_stat, ks_pval = self.ks_test(X_ref, X_test)
            results['ks_stat'] = ks_stat
            results['ks_pval'] = ks_pval
        
        if 'js' in metrics:
            results['js'] = self.js_divergence(X_ref, X_test)
        
        if 'wasserstein' in metrics:
            results['wasserstein'] = self.wasserstein(X_ref, X_test)
        
        return results


class MultiWindowDriftAnalysis:
    """Analyze drift across multiple time windows."""
    
    def __init__(self, n_bins: int = 10):
        self.detector = DriftDetector(n_bins=n_bins)
    
    def compute_drift_matrix(
        self,
        splits: list,
        metrics: Optional[list] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute drift metrics across all features and windows.
        
        Args:
            splits: List of split dicts with X_ref, X_test, feature_names
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of DataFrames, one per metric (features x windows)
        """
        if metrics is None:
            metrics = ['psi', 'kl', 'ks_stat', 'js', 'wasserstein']
        
        feature_names = splits[0]['feature_names']
        n_features = len(feature_names)
        n_windows = len(splits)
        
        # Initialize storage for each metric
        drift_matrices = {}
        for metric in metrics:
            drift_matrices[metric] = np.zeros((n_features, n_windows))
        
        # Compute drift for each feature and window
        for window_idx, split in enumerate(splits):
            X_ref = split['X_ref']
            X_test = split['X_test']
            
            for feat_idx, feat_name in enumerate(feature_names):
                X_ref_feat = X_ref[:, feat_idx]
                X_test_feat = X_test[:, feat_idx]
                
                drift_scores = self.detector.detect_drift_all_metrics(
                    X_ref_feat, X_test_feat, metrics=metrics
                )
                
                for metric in metrics:
                    if metric == 'ks_test':
                        # Already split into ks_stat and ks_pval
                        continue
                    if metric in drift_scores:
                        drift_matrices[metric][feat_idx, window_idx] = drift_scores[metric]
        
        # Convert to DataFrames
        drift_dfs = {}
        for metric, matrix in drift_matrices.items():
            drift_dfs[metric] = pd.DataFrame(
                matrix,
                index=feature_names,
                columns=[f"window_{i}" for i in range(n_windows)]
            )
        
        logger.info(f"Computed drift matrix for {n_features} features across {n_windows} windows")
        return drift_dfs
    
    def get_top_drifting_features(
        self,
        drift_df: pd.DataFrame,
        top_n: int = 5,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """
        Rank features by drift magnitude.
        
        Args:
            drift_df: Drift matrix (features x windows)
            top_n: Number of top features to return
            aggregation: "mean", "max", "median" across windows
            
        Returns:
            DataFrame with feature names and aggregated drift scores
        """
        if aggregation == "mean":
            scores = drift_df.mean(axis=1)
        elif aggregation == "max":
            scores = drift_df.max(axis=1)
        elif aggregation == "median":
            scores = drift_df.median(axis=1)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        ranked = scores.sort_values(ascending=False)
        top_features = pd.DataFrame({
            'feature': ranked.index[:top_n],
            'drift_score': ranked.values[:top_n],
            'rank': range(1, top_n + 1)
        })
        
        return top_features
    
    def get_drift_onset_window(
        self,
        drift_df: pd.DataFrame,
        threshold: float,
    ) -> Dict[str, Optional[int]]:
        """
        Identify first window where drift exceeds threshold for each feature.
        
        Args:
            drift_df: Drift matrix (features x windows)
            threshold: Drift threshold for detection
            
        Returns:
            Dictionary mapping feature names to onset window index
        """
        onset_windows = {}
        
        for feature in drift_df.index:
            values = drift_df.loc[feature].values
            onset_idx = np.where(values > threshold)[0]
            onset_windows[feature] = int(onset_idx[0]) if len(onset_idx) > 0 else None
        
        return onset_windows
