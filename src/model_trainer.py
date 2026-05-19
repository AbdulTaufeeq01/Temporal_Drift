"""
Model Training and Evaluation Module

Trains baseline models and tracks their performance degradation
across time windows.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and manage baseline models."""
    
    def __init__(self, model_type: str = "logistic", random_state: int = 42):
        """
        Initialize model trainer.
        
        Args:
            model_type: "logistic", "random_forest", or "xgboost"
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = self._create_model()
        self.feature_names = None
    
    def _create_model(self):
        """Create the model based on model_type."""
        if self.model_type == "logistic":
            return LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_type == "xgboost":
            return XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                verbosity=0,
                use_label_encoder=False,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model on reference data.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target vector (n_samples,)
        """
        logger.info(f"Training {self.model_type} on {X.shape[0]} samples")
        self.model.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            # For models without predict_proba
            predictions = self.predict(X)
            proba = np.zeros((len(predictions), 2))
            proba[:, 1] = predictions
            proba[:, 0] = 1 - predictions
            return proba


class ModelEvaluator:
    """Evaluate model performance across time windows."""
    
    METRICS = {
        'accuracy': accuracy_score,
        'f1': f1_score,
        'auc_roc': roc_auc_score,
        'brier_score': brier_score_loss,
    }
    
    def __init__(self, metrics: Optional[List[str]] = None):
        """
        Initialize evaluator.
        
        Args:
            metrics: List of metrics to compute (default: all)
        """
        self.metrics = metrics or list(self.METRICS.keys())
    
    def evaluate_window(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Evaluate model on a single window.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (needed for AUC and Brier)
            
        Returns:
            Dictionary with metric names and values
        """
        results = {}
        
        for metric_name in self.metrics:
            if metric_name == 'accuracy':
                results[metric_name] = accuracy_score(y_true, y_pred)
            
            elif metric_name == 'f1':
                results[metric_name] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            
            elif metric_name == 'auc_roc':
                if y_proba is not None:
                    results[metric_name] = roc_auc_score(
                        y_true, y_proba[:, 1], multi_class='ovr'
                    )
                else:
                    results[metric_name] = np.nan
            
            elif metric_name == 'brier_score':
                if y_proba is not None:
                    results[metric_name] = brier_score_loss(y_true, y_proba[:, 1])
                else:
                    results[metric_name] = np.nan
        
        return results
    
    def evaluate_all_windows(
        self,
        model: ModelTrainer,
        splits: List[Dict],
    ) -> pd.DataFrame:
        """
        Evaluate model on all time windows.
        
        Args:
            model: Trained ModelTrainer instance
            splits: List of split dicts with X_test, y_test, window_idx
            
        Returns:
            DataFrame with metrics for each window
        """
        results = []
        
        for split in splits:
            X_test = split['X_test']
            y_test = split['y_test']
            window_idx = split['window_idx']
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)
            
            # Evaluate
            metrics = self.evaluate_window(y_test, y_pred, y_proba)
            metrics['window'] = window_idx
            
            results.append(metrics)
        
        df = pd.DataFrame(results)
        logger.info(f"Evaluated model on {len(splits)} windows")
        
        return df
    
    def compute_performance_degradation(
        self,
        performance_df: pd.DataFrame,
        ref_window_idx: int = 0,
    ) -> Dict[str, np.ndarray]:
        """
        Compute performance degradation relative to reference window.
        
        Args:
            performance_df: DataFrame with metrics across windows
            ref_window_idx: Reference window (default: first window)
            
        Returns:
            Dictionary mapping metrics to degradation arrays
        """
        degradation = {}
        
        ref_row = performance_df[performance_df['window'] == ref_window_idx].iloc[0]
        
        for metric in self.metrics:
            ref_value = ref_row[metric]
            current_values = performance_df[metric].values
            
            # Degradation: (ref - current) / ref, negative means improvement
            deg = (ref_value - current_values) / (ref_value + 1e-10)
            degradation[metric] = deg
        
        return degradation
    
    def identify_critical_degradation(
        self,
        degradation_dict: Dict[str, np.ndarray],
        threshold: float = 0.05,
    ) -> List[int]:
        """
        Identify windows with significant performance degradation.
        
        Args:
            degradation_dict: Degradation dict from compute_performance_degradation
            threshold: Degradation threshold (default: 5%)
            
        Returns:
            List of window indices with significant degradation
        """
        # Use primary metric (AUC-ROC) if available, else use first metric
        metric_key = 'auc_roc' if 'auc_roc' in degradation_dict else list(degradation_dict.keys())[0]
        degradation = degradation_dict[metric_key]
        
        critical_windows = np.where(degradation > threshold)[0].tolist()
        
        return critical_windows


class PerformanceTracker:
    """Track and aggregate performance across models and strategies."""
    
    def __init__(self):
        self.results = {}  # {strategy_name: {model_name: performance_df}}
    
    def record_strategy_results(
        self,
        strategy_name: str,
        model_name: str,
        performance_df: pd.DataFrame,
    ) -> None:
        """
        Record performance results for a strategy-model combination.
        
        Args:
            strategy_name: Name of retraining strategy
            model_name: Name of model
            performance_df: DataFrame with metrics across windows
        """
        if strategy_name not in self.results:
            self.results[strategy_name] = {}
        
        self.results[strategy_name][model_name] = performance_df
    
    def get_strategy_summary(
        self,
        strategy_name: str,
        aggregation: str = "mean",
    ) -> pd.DataFrame:
        """
        Get summary statistics for a strategy across all models.
        
        Args:
            strategy_name: Name of strategy
            aggregation: "mean", "median", or "std"
            
        Returns:
            Aggregated performance summary
        """
        if strategy_name not in self.results:
            raise ValueError(f"Strategy {strategy_name} not found")
        
        model_results = self.results[strategy_name]
        
        # Combine results from all models
        combined = pd.concat(model_results.values(), ignore_index=False)
        
        if aggregation == "mean":
            summary = combined.groupby('window').mean()
        elif aggregation == "median":
            summary = combined.groupby('window').median()
        elif aggregation == "std":
            summary = combined.groupby('window').std()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
        
        return summary
    
    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare all strategies on average metrics.
        
        Returns:
            DataFrame with strategy names and average metrics
        """
        comparisons = []
        
        for strategy_name in self.results:
            model_results = self.results[strategy_name]
            
            # Combine all models
            combined = pd.concat(model_results.values(), ignore_index=False)
            
            # Compute averages
            avg_metrics = combined.mean(numeric_only=True)
            
            comparison = {
                'strategy': strategy_name,
                **avg_metrics.to_dict()
            }
            comparisons.append(comparison)
        
        df = pd.DataFrame(comparisons)
        logger.info(f"Compared {len(self.results)} strategies")
        
        return df
