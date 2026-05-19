"""
Visualization Module

Generate plots and dashboards for drift analysis and model performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class DriftVisualizer:
    """Generate visualizations for drift analysis."""
    
    def __init__(self, output_dir: str = "reports/figures"):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"Output directory: {self.output_dir}")
    
    def plot_drift_heatmap(
        self,
        drift_df: pd.DataFrame,
        metric_name: str = "PSI",
        cmap: str = "RdYlGn_r",
        save_name: str = "drift_heatmap.png"
    ) -> None:
        """
        Plot drift heatmap (features x time windows).
        
        Args:
            drift_df: DataFrame with drift scores (features x windows)
            metric_name: Name of metric for title
            cmap: Colormap name
            save_name: Output filename
        """
        fig, ax = plt.subplots(figsize=(16, 10))
        
        sns.heatmap(
            drift_df,
            cmap=cmap,
            cbar_kws={'label': metric_name},
            ax=ax,
            annot=False,
            fmt='.3f',
        )
        
        ax.set_title(f"Drift Heatmap ({metric_name}): Features × Time Windows", fontsize=14, fontweight='bold')
        ax.set_xlabel("Time Windows", fontsize=12)
        ax.set_ylabel("Features", fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved drift heatmap: {save_name}")
    
    def plot_feature_distributions(
        self,
        X_ref: np.ndarray,
        X_current: np.ndarray,
        feature_names: List[str] = None,
        top_n_features: List[int] = None,
        save_name: str = "feature_distributions.png"
    ) -> None:
        """
        Plot reference vs current distributions for top drifted features.
        
        Args:
            X_ref: Reference feature matrix
            X_current: Current feature matrix
            feature_names: Feature names
            top_n_features: List of feature indices to plot
            save_name: Output filename
        """
        n_features = X_ref.shape[1]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(n_features)]
        
        if top_n_features is None:
            top_n_features = list(range(min(5, n_features)))
        
        n_plots = len(top_n_features)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for plot_idx, feat_idx in enumerate(top_n_features):
            ax = axes[plot_idx]
            
            ax.hist(X_ref[:, feat_idx], bins=30, alpha=0.6, label='Reference', color='blue')
            ax.hist(X_current[:, feat_idx], bins=30, alpha=0.6, label='Current', color='red')
            
            ax.set_xlabel("Value", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title(f"{feature_names[feat_idx]}", fontsize=11, fontweight='bold')
            ax.legend()
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle("Feature Distributions: Reference vs Current", fontsize=14, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved feature distributions: {save_name}")
    
    def plot_performance_over_time(
        self,
        performance_data: Dict[str, pd.DataFrame],
        metrics: List[str] = None,
        save_name: str = "performance_over_time.png"
    ) -> None:
        """
        Plot model performance over time for multiple strategies.
        
        Args:
            performance_data: Dict mapping strategy names to performance DFs
            metrics: List of metrics to plot
            save_name: Output filename
        """
        if metrics is None:
            metrics = ['auc_roc', 'f1']
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]
            
            for strategy_name, perf_df in performance_data.items():
                if metric in perf_df.columns:
                    ax.plot(
                        perf_df['window'],
                        perf_df[metric],
                        marker='o',
                        label=strategy_name,
                        linewidth=2,
                    )
            
            ax.set_xlabel("Time Window", fontsize=11)
            ax.set_ylabel(metric.upper(), fontsize=11)
            ax.set_title(f"{metric.upper()} Over Time", fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle("Model Performance Across Retraining Strategies", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved performance plot: {save_name}")
    
    def plot_drift_trajectory(
        self,
        drift_trajectories: Dict[str, np.ndarray],
        feature_names: List[str] = None,
        top_n: int = 5,
        save_name: str = "drift_trajectory.png"
    ) -> None:
        """
        Plot drift scores over time for top features.
        
        Args:
            drift_trajectories: Dict mapping feature names to drift score arrays
            feature_names: Feature names (or use keys from drift_trajectories)
            top_n: Number of top features to plot
            save_name: Output filename
        """
        if feature_names is None:
            feature_names = list(drift_trajectories.keys())[:top_n]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for feat_name in feature_names:
            if feat_name in drift_trajectories:
                ax.plot(
                    drift_trajectories[feat_name],
                    marker='o',
                    label=feat_name,
                    linewidth=2,
                )
        
        ax.set_xlabel("Time Window", fontsize=11)
        ax.set_ylabel("Drift Score (PSI)", fontsize=11)
        ax.set_title(f"Top {top_n} Drifting Features Over Time", fontsize=12, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved drift trajectory: {save_name}")
    
    def plot_performance_with_retraining_events(
        self,
        performance_df: pd.DataFrame,
        retrain_windows: List[int],
        metric: str = 'auc_roc',
        save_name: str = "performance_with_events.png"
    ) -> None:
        """
        Plot performance with retraining events marked.
        
        Args:
            performance_df: Performance DataFrame
            retrain_windows: List of window indices where retraining occurred
            metric: Metric to plot
            save_name: Output filename
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot performance line
        ax.plot(
            performance_df['window'],
            performance_df[metric],
            marker='o',
            linewidth=2,
            label=metric.upper(),
            color='blue'
        )
        
        # Mark retraining events
        for retrain_window in retrain_windows:
            if retrain_window > 0:  # Skip initial training
                ax.axvline(x=retrain_window, color='red', linestyle='--', alpha=0.7, linewidth=2)
        
        # Add legend for retraining events
        if retrain_windows:
            ax.axvline(x=retrain_windows[0], color='red', linestyle='--', alpha=0.7, linewidth=2, label='Retraining Events')
        
        ax.set_xlabel("Time Window", fontsize=11)
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(f"{metric.upper()} with Retraining Events", fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved performance with events: {save_name}")
    
    def plot_strategy_comparison(
        self,
        comparison_df: pd.DataFrame,
        metrics: List[str] = None,
        save_name: str = "strategy_comparison.png"
    ) -> None:
        """
        Compare strategies across metrics.
        
        Args:
            comparison_df: Strategy comparison DataFrame
            metrics: List of metrics to plot
            save_name: Output filename
        """
        if metrics is None:
            numeric_cols = comparison_df.select_dtypes(include=[np.number]).columns.tolist()
            metrics = [c for c in numeric_cols if c != 'n_retrains'][:3]
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
        if n_metrics == 1:
            axes = [axes]
        
        for metric_idx, metric in enumerate(metrics):
            ax = axes[metric_idx]
            
            if metric in comparison_df.columns:
                comparison_df.plot(
                    x='strategy',
                    y=metric,
                    kind='bar',
                    ax=ax,
                    legend=False,
                    color='steelblue'
                )
                
                ax.set_xlabel("Strategy", fontsize=11)
                ax.set_ylabel(metric.upper(), fontsize=11)
                ax.set_title(f"{metric.upper()} by Strategy", fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle("Strategy Comparison", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved strategy comparison: {save_name}")
    
    def create_dashboard(
        self,
        drift_dfs: Dict[str, pd.DataFrame],
        performance_data: Dict[str, pd.DataFrame],
        splits: List[Dict],
        save_name: str = "dashboard.png"
    ) -> None:
        """
        Create a comprehensive dashboard.
        
        Args:
            drift_dfs: Dict of drift matrices
            performance_data: Dict of performance DataFrames
            splits: List of data splits
            save_name: Output filename
        """
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Drift heatmap
        ax1 = fig.add_subplot(gs[0, :2])
        if 'psi' in drift_dfs:
            sns.heatmap(drift_dfs['psi'], cmap='RdYlGn_r', ax=ax1, cbar_kws={'label': 'PSI'})
            ax1.set_title("Drift Heatmap (PSI)", fontweight='bold')
        
        # 2. Performance over time (first strategy)
        ax2 = fig.add_subplot(gs[0, 2])
        if performance_data:
            first_strategy = list(performance_data.keys())[0]
            perf_df = performance_data[first_strategy]
            if 'auc_roc' in perf_df.columns:
                ax2.plot(perf_df['window'], perf_df['auc_roc'], marker='o', linewidth=2)
                ax2.set_xlabel("Window")
                ax2.set_ylabel("AUC-ROC")
                ax2.set_title("Performance Over Time", fontweight='bold')
                ax2.grid(True, alpha=0.3)
        
        # 3. Feature distributions
        ax3 = fig.add_subplot(gs[1, :])
        if splits:
            X_ref = splits[0]['X_ref']
            X_test = splits[-1]['X_test']
            for i in range(min(3, X_ref.shape[1])):
                ax3.hist(X_ref[:, i], bins=20, alpha=0.5, label=f'Ref-F{i}')
                ax3.hist(X_test[:, i], bins=20, alpha=0.5, label=f'Test-F{i}')
            ax3.set_xlabel("Value")
            ax3.set_ylabel("Frequency")
            ax3.set_title("Feature Distributions (First 3 Features)", fontweight='bold')
            ax3.legend()
        
        # 4. Statistics
        ax4 = fig.add_subplot(gs[2, :])
        ax4.axis('off')
        
        stats_text = "Analysis Summary:\n"
        if drift_dfs and 'psi' in drift_dfs:
            psi_df = drift_dfs['psi']
            stats_text += f"• Features analyzed: {len(psi_df)}\n"
            stats_text += f"• Time windows: {len(psi_df.columns)}\n"
            stats_text += f"• Max PSI: {psi_df.max().max():.3f}\n"
            stats_text += f"• Mean PSI: {psi_df.mean().mean():.3f}\n"
        
        if performance_data:
            stats_text += f"• Strategies evaluated: {len(performance_data)}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center', family='monospace')
        
        plt.savefig(self.output_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved dashboard: {save_name}")


class ReportGenerator:
    """Generate summary reports."""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
    
    def generate_text_report(
        self,
        summary: Dict,
        save_name: str = "summary.md"
    ) -> None:
        """Generate markdown summary report."""
        report = "# Temporal Drift Analysis Report\n\n"
        
        report += "## Summary Statistics\n\n"
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                report += f"- **{key}**: {value:.4f}\n"
            else:
                report += f"- **{key}**: {value}\n"
        
        with open(self.output_dir / save_name, 'w') as f:
            f.write(report)
        
        logger.info(f"Saved report: {save_name}")
        
        for i in range(top_n):
            if i < n_features:
                ax = axes[i]
                
                ax.hist(X_ref[:, i], bins=30, alpha=0.6, label='Reference', color='blue', density=True)
                ax.hist(X_current[:, i], bins=30, alpha=0.6, label='Current', color='red', density=True)
                
                ax.set_title(f"Distribution: {feature_names[i]}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Density")
                ax.legend()
        
        # Hide unused subplots
        for i in range(top_n, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()

    def plot_drift_heatmap(
        self,
        drift_df: pd.DataFrame,
        metric: str = 'psi',
        save_name: str = "drift_heatmap.png"
    ) -> None:
        """
        Plot heatmap of drift metrics across features and windows.
        X-axis: time windows, Y-axis: features, Color: metric value
        """
        # Pivot table
        heatmap_data = drift_df.pivot_table(
            index='feature',
            columns='window',
            values=metric,
            aggfunc='mean'
        )
        
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            heatmap_data,
            cmap='YlOrRd',
            annot=True,
            fmt='.3f',
            cbar_kws={'label': metric.upper()},
            ax=ax
        )
        
        ax.set_title(f"Feature Drift Heatmap ({metric.upper()})")
        ax.set_xlabel("Time Window")
        ax.set_ylabel("Feature")
        
        plt.tight_layout()
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()

    def plot_drift_over_time(
        self,
        drift_df: pd.DataFrame,
        feature_name: str,
        metric: str = 'psi',
        save_name: str = None
    ) -> None:
        """Plot drift metric over time for a single feature."""
        if save_name is None:
            save_name = f"drift_trajectory_{feature_name}.png"
        
        feature_data = drift_df[drift_df['feature'] == feature_name].sort_values('window')
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(feature_data['window'], feature_data[metric], marker='o', linewidth=2, markersize=8)
        ax.fill_between(feature_data['window'], feature_data[metric], alpha=0.3)
        
        ax.set_title(f"Drift Trajectory: {feature_name} ({metric.upper()})")
        ax.set_xlabel("Time Window")
        ax.set_ylabel(f"{metric.upper()} Score")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()

    def plot_multiple_drift_metrics(
        self,
        drift_df: pd.DataFrame,
        feature_name: str,
        metrics: List[str] = None,
        save_name: str = None
    ) -> None:
        """Plot multiple drift metrics for a feature."""
        if metrics is None:
            metrics = ['psi', 'kl', 'js', 'wasserstein']
        
        if save_name is None:
            save_name = f"drift_metrics_{feature_name}.png"
        
        feature_data = drift_df[drift_df['feature'] == feature_name].sort_values('window')
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics[:4]):
            if metric in feature_data.columns:
                ax = axes[idx]
                ax.plot(feature_data['window'], feature_data[metric], marker='o', linewidth=2, color='tab:blue')
                ax.fill_between(feature_data['window'], feature_data[metric], alpha=0.3)
                ax.set_title(f"{metric.upper()}")
                ax.set_xlabel("Time Window")
                ax.set_ylabel("Score")
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()


class PerformanceVisualizer:
    """Generate visualizations for model performance."""

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def plot_performance_over_time(
        self,
        performance_df: pd.DataFrame,
        metric: str = 'auc_roc',
        save_name: str = None
    ) -> None:
        """Plot model performance over time windows."""
        if save_name is None:
            save_name = f"performance_{metric}.png"
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(performance_df['window'], performance_df[metric], marker='o', linewidth=2.5, markersize=8, label=metric.upper())
        ax.fill_between(performance_df['window'], performance_df[metric], alpha=0.2)
        
        # Add baseline
        baseline = performance_df[metric].iloc[0]
        ax.axhline(baseline, color='red', linestyle='--', linewidth=2, label='Baseline', alpha=0.7)
        
        ax.set_title(f"Model Performance Over Time ({metric.upper()})")
        ax.set_xlabel("Time Window")
        ax.set_ylabel(metric.upper())
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()

    def plot_performance_comparison(
        self,
        performance_results: Dict[str, pd.DataFrame],
        metric: str = 'auc_roc',
        save_name: str = None
    ) -> None:
        """Compare performance across multiple strategies."""
        if save_name is None:
            save_name = f"strategy_comparison_{metric}.png"
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for strategy_name, perf_df in performance_results.items():
            ax.plot(perf_df['window'], perf_df[metric], marker='o', linewidth=2, label=strategy_name)
        
        ax.set_title(f"Strategy Comparison ({metric.upper()})")
        ax.set_xlabel("Time Window")
        ax.set_ylabel(metric.upper())
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()

    def plot_retraining_events(
        self,
        performance_df: pd.DataFrame,
        metric: str = 'auc_roc',
        save_name: str = None
    ) -> None:
        """
        Plot performance with retraining event markers.
        """
        if save_name is None:
            save_name = f"retraining_events_{metric}.png"
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        ax.plot(performance_df['window'], performance_df[metric], marker='o', linewidth=2.5, markersize=8, label='Performance')
        
        # Mark retraining events
        if 'retrained' in performance_df.columns:
            retrained_windows = performance_df[performance_df['retrained']]['window'].values
            for window in retrained_windows:
                ax.axvline(window, color='green', linestyle='--', alpha=0.5, linewidth=1.5)
            
            # Add legend entry for retraining
            ax.plot([], [], color='green', linestyle='--', linewidth=1.5, label='Retrain Event')
        
        ax.set_title(f"Performance with Retraining Events ({metric.upper()})")
        ax.set_xlabel("Time Window")
        ax.set_ylabel(metric.upper())
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()

    def plot_performance_degradation(
        self,
        performance_df: pd.DataFrame,
        metric: str = 'auc_roc',
        save_name: str = None
    ) -> None:
        """Plot performance degradation from baseline."""
        if save_name is None:
            save_name = f"degradation_{metric}.png"
        
        baseline = performance_df[metric].iloc[0]
        degradation = baseline - performance_df[metric]
        degradation_pct = (degradation / baseline * 100)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Absolute degradation
        ax1.bar(performance_df['window'], degradation, color='coral')
        ax1.set_title("Absolute Degradation")
        ax1.set_xlabel("Time Window")
        ax1.set_ylabel(f"Degradation ({metric.upper()})")
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Percentage degradation
        ax2.bar(performance_df['window'], degradation_pct, color='lightcoral')
        ax2.set_title("Percentage Degradation")
        ax2.set_xlabel("Time Window")
        ax2.set_ylabel(f"Degradation (%)")
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()

    def plot_multiple_metrics(
        self,
        performance_df: pd.DataFrame,
        metrics: List[str] = None,
        save_name: str = None
    ) -> None:
        """Plot multiple performance metrics."""
        if metrics is None:
            metrics = ['accuracy', 'f1', 'auc_roc']
        
        if save_name is None:
            save_name = "performance_metrics.png"
        
        available_metrics = [m for m in metrics if m in performance_df.columns]
        
        fig, axes = plt.subplots(1, len(available_metrics), figsize=(5 * len(available_metrics), 5))
        if len(available_metrics) == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, available_metrics):
            ax.plot(performance_df['window'], performance_df[metric], marker='o', linewidth=2, markersize=8)
            ax.fill_between(performance_df['window'], performance_df[metric], alpha=0.2)
            ax.set_title(metric.upper())
            ax.set_xlabel("Time Window")
            ax.set_ylabel("Score")
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()


class SummaryDashboard:
    """Create comprehensive summary dashboard."""

    def __init__(self, output_dir: str = "reports/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

    def create_full_dashboard(
        self,
        drift_df: pd.DataFrame,
        performance_df: pd.DataFrame,
        save_name: str = "full_dashboard.png"
    ) -> None:
        """Create 2x2 dashboard with key visualizations."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Plot 1: Drift heatmap
        ax1 = fig.add_subplot(gs[0, 0])
        heatmap_data = drift_df.pivot_table(index='feature', columns='window', values='psi', aggfunc='mean')
        sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax1, cbar_kws={'label': 'PSI'})
        ax1.set_title("Feature Drift (PSI)")
        
        # Plot 2: Performance over time
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(performance_df['window'], performance_df['auc_roc'], marker='o', linewidth=2, markersize=8)
        ax2.fill_between(performance_df['window'], performance_df['auc_roc'], alpha=0.2)
        ax2.set_title("Model Performance (AUC-ROC)")
        ax2.set_xlabel("Time Window")
        ax2.set_ylabel("AUC-ROC")
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Top drifting features
        ax3 = fig.add_subplot(gs[1, 0])
        top_features = drift_df.groupby('feature')['psi'].mean().nlargest(5)
        top_features.plot(kind='barh', ax=ax3, color='steelblue')
        ax3.set_title("Top 5 Drifting Features")
        ax3.set_xlabel("Mean PSI")
        
        # Plot 4: Performance degradation
        ax4 = fig.add_subplot(gs[1, 1])
        baseline = performance_df['auc_roc'].iloc[0]
        degradation = baseline - performance_df['auc_roc']
        ax4.bar(performance_df['window'], degradation, color='coral')
        ax4.set_title("Performance Degradation")
        ax4.set_xlabel("Time Window")
        ax4.set_ylabel("Degradation")
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        filepath = self.output_dir / save_name
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {filepath}")
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create sample data
    X_ref = np.random.normal(0, 1, (1000, 5))
    X_current = np.random.normal(0.3, 1, (1000, 5))
    
    drift_viz = DriftVisualizer()
    drift_viz.plot_feature_distributions(X_ref, X_current)
    
    print("Visualizations created successfully!")
