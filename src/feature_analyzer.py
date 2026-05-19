"""
Feature Analysis Module

Analyzes per-feature drift patterns over time and identifies
key drift characteristics for each feature.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from .drift_detectors import DriftDetector, MultiWindowDriftAnalysis
import logging

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """Analyze drift patterns for individual features."""
    
    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.detector = DriftDetector(n_bins=n_bins)
        self.multi_window = MultiWindowDriftAnalysis(n_bins=n_bins)
    
    def analyze_feature_trajectory(
        self,
        splits: List[Dict],
        feature_idx: int,
        feature_name: str,
    ) -> Dict:
        """
        Analyze drift trajectory for a single feature across all windows.
        
        Args:
            splits: List of split dicts
            feature_idx: Index of feature to analyze
            feature_name: Name of feature
            
        Returns:
            Dictionary with drift metrics across windows
        """
        psi_scores = []
        kl_scores = []
        ks_stats = []
        ks_pvals = []
        
        for split in splits:
            X_ref_feat = split['X_ref'][:, feature_idx]
            X_test_feat = split['X_test'][:, feature_idx]
            
            psi_scores.append(self.detector.psi(X_ref_feat, X_test_feat))
            kl_scores.append(self.detector.kl_divergence(X_ref_feat, X_test_feat))
            
            ks_stat, ks_pval = self.detector.ks_test(X_ref_feat, X_test_feat)
            ks_stats.append(ks_stat)
            ks_pvals.append(ks_pval)
        
        return {
            'feature_name': feature_name,
            'psi_scores': np.array(psi_scores),
            'kl_scores': np.array(kl_scores),
            'ks_stats': np.array(ks_stats),
            'ks_pvals': np.array(ks_pvals),
        }
    
    def compute_feature_statistics(
        self,
        splits: List[Dict],
        feature_idx: int,
    ) -> Dict:
        """
        Compute statistical properties of feature across windows.
        
        Args:
            splits: List of split dicts
            feature_idx: Index of feature
            
        Returns:
            Dictionary with statistical summaries
        """
        ref_means = []
        ref_stds = []
        test_means = []
        test_stds = []
        
        for split in splits:
            X_ref_feat = split['X_ref'][:, feature_idx]
            X_test_feat = split['X_test'][:, feature_idx]
            
            ref_means.append(np.mean(X_ref_feat))
            ref_stds.append(np.std(X_ref_feat))
            test_means.append(np.mean(X_test_feat))
            test_stds.append(np.std(X_test_feat))
        
        return {
            'ref_means': np.array(ref_means),
            'ref_stds': np.array(ref_stds),
            'test_means': np.array(test_means),
            'test_stds': np.array(test_stds),
            'mean_shift': np.abs(np.array(test_means) - np.array(ref_means)),
        }
    
    def analyze_all_features(
        self,
        splits: List[Dict],
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Comprehensive analysis of all features.
        
        Args:
            splits: List of split dicts
            
        Returns:
            (summary_df, detailed_trajectories_dict)
        """
        feature_names = splits[0]['feature_names']
        n_features = len(feature_names)
        
        summaries = []
        trajectories = {}
        
        for feat_idx, feat_name in enumerate(feature_names):
            trajectory = self.analyze_feature_trajectory(splits, feat_idx, feat_name)
            trajectories[feat_name] = trajectory
            
            stats = self.compute_feature_statistics(splits, feat_idx)
            
            summary = {
                'feature': feat_name,
                'avg_psi': trajectory['psi_scores'].mean(),
                'max_psi': trajectory['psi_scores'].max(),
                'std_psi': trajectory['psi_scores'].std(),
                'avg_kl': trajectory['kl_scores'].mean(),
                'max_kl': trajectory['kl_scores'].max(),
                'avg_ks_stat': trajectory['ks_stats'].mean(),
                'mean_shift': stats['mean_shift'].mean(),
                'max_mean_shift': stats['mean_shift'].max(),
            }
            summaries.append(summary)
        
        summary_df = pd.DataFrame(summaries)
        logger.info(f"Analyzed {n_features} features")
        
        return summary_df, trajectories
    
    def get_feature_risk_scores(
        self,
        summary_df: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Compute risk scores for each feature based on drift metrics.
        
        Args:
            summary_df: Feature summary dataframe
            weights: Weights for each metric (default: equal)
            
        Returns:
            DataFrame with risk scores and rankings
        """
        if weights is None:
            weights = {
                'psi': 0.4,
                'kl': 0.3,
                'ks_stat': 0.2,
                'mean_shift': 0.1,
            }
        
        # Normalize each metric to [0, 1]
        df_normalized = summary_df.copy()
        
        for metric in ['avg_psi', 'avg_kl', 'avg_ks_stat', 'mean_shift']:
            max_val = df_normalized[metric].max()
            if max_val > 0:
                df_normalized[metric] = df_normalized[metric] / max_val
        
        # Compute weighted risk score
        risk_score = (
            weights.get('psi', 0.25) * df_normalized['avg_psi'] +
            weights.get('kl', 0.25) * df_normalized['avg_kl'] +
            weights.get('ks_stat', 0.25) * df_normalized['avg_ks_stat'] +
            weights.get('mean_shift', 0.25) * df_normalized['mean_shift']
        )
        
        result = summary_df.copy()
        result['risk_score'] = risk_score
        result['risk_rank'] = result['risk_score'].rank(ascending=False).astype(int)
        result = result.sort_values('risk_rank')
        
        return result


class FeatureImportanceAnalyzer:
    """Analyze relationship between drift and model performance."""
    
    def compute_drift_performance_correlation(
        self,
        drift_df: pd.DataFrame,
        performance_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute correlation between feature drift and model performance.
        
        Args:
            drift_df: Drift scores (features x windows)
            performance_df: Performance metrics (windows, or windows x models)
            
        Returns:
            Correlation matrix
        """
        # Aggregate performance across models if needed
        if isinstance(performance_df.columns, pd.MultiIndex):
            perf_agg = performance_df.mean(axis=1)  # Average across models
        else:
            perf_agg = performance_df.iloc[:, 0]  # Take first metric
        
        correlations = []
        
        for feature in drift_df.index:
            feature_drift = drift_df.loc[feature].values
            
            # Align lengths
            min_len = min(len(feature_drift), len(perf_agg))
            feature_drift = feature_drift[:min_len]
            perf_values = perf_agg.values[:min_len]
            
            corr = np.corrcoef(feature_drift, perf_values)[0, 1]
            correlations.append({
                'feature': feature,
                'drift_perf_correlation': corr,
            })
        
        corr_df = pd.DataFrame(correlations)
        corr_df['abs_correlation'] = np.abs(corr_df['drift_perf_correlation'])
        corr_df = corr_df.sort_values('abs_correlation', ascending=False)
        
        return corr_df
