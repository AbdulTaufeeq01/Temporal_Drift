"""
Retraining Strategy Module

Implement and compare different model retraining strategies:
- No retraining
- Periodic retraining
- Trigger-based retraining
- Sliding window retraining
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from .model_trainer import ModelTrainer, ModelEvaluator

logger = logging.getLogger(__name__)


class RetrainingStrategy:
    """Base class for retraining strategies."""
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize strategy.
        
        Args:
            model_type: Type of model to use
            random_state: Random seed
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.retrain_windows = []
    
    def should_retrain(self, window_idx: int, **kwargs) -> bool:
        """
        Determine if model should be retrained at this window.
        
        Override in subclasses.
        """
        return False
    
    def retrain(self, X: np.ndarray, y: np.ndarray, window_idx: int) -> None:
        """Retrain the model."""
        self.model = ModelTrainer(self.model_type, self.random_state)
        self.model.train(X, y)
        self.retrain_windows.append(window_idx)
        logger.info(f"Retrained at window {window_idx}")
    
    def get_retrain_count(self) -> int:
        """Get number of retraining events."""
        return len(self.retrain_windows)
    
    def get_retrain_windows(self) -> List[int]:
        """Get list of windows where retraining occurred."""
        return self.retrain_windows.copy()


class NoRetrainingStrategy(RetrainingStrategy):
    """Baseline: train once on reference, never retrain."""
    
    def should_retrain(self, window_idx: int, **kwargs) -> bool:
        """Never retrain."""
        return False


class PeriodicRetrainingStrategy(RetrainingStrategy):
    """Retrain every K windows."""
    
    def __init__(
        self,
        period: int,
        model_type: str = 'random_forest',
        random_state: int = 42,
    ):
        """
        Initialize periodic retraining.
        
        Args:
            period: Retrain every K windows
            model_type: Type of model
            random_state: Random seed
        """
        super().__init__(model_type, random_state)
        self.period = period
    
    def should_retrain(self, window_idx: int, **kwargs) -> bool:
        """Retrain if window_idx is divisible by period."""
        # Don't retrain at window 0 (reference)
        return window_idx > 0 and window_idx % self.period == 0


class TriggerBasedRetrainingStrategy(RetrainingStrategy):
    """Retrain when drift metric exceeds threshold."""
    
    def __init__(
        self,
        drift_metric: str = 'psi',
        threshold: float = 0.25,
        model_type: str = 'random_forest',
        random_state: int = 42,
    ):
        """
        Initialize trigger-based retraining.
        
        Args:
            drift_metric: Metric to monitor ('psi', 'kl', 'ks_stat')
            threshold: Threshold for triggering retraining
            model_type: Type of model
            random_state: Random seed
        """
        super().__init__(model_type, random_state)
        self.drift_metric = drift_metric
        self.threshold = threshold
    
    def should_retrain(self, window_idx: int, drift_value: Optional[float] = None, **kwargs) -> bool:
        """Retrain if drift exceeds threshold."""
        if drift_value is None:
            return False
        return drift_value > self.threshold


class SlidingWindowRetrainingStrategy(RetrainingStrategy):
    """Retrain using sliding window of recent data."""
    
    def __init__(
        self,
        window_size: int = 2,
        model_type: str = 'random_forest',
        random_state: int = 42,
    ):
        """
        Initialize sliding window retraining.
        
        Args:
            window_size: Keep last N windows of data for retraining
            model_type: Type of model
            random_state: Random seed
        """
        super().__init__(model_type, random_state)
        self.window_size = window_size
    
    def should_retrain(self, window_idx: int, **kwargs) -> bool:
        """Retrain at every window using sliding data."""
        # Retrain at every window > 0
        return window_idx > 0
    
    def get_training_data_indices(self, current_window_idx: int, n_windows: int) -> List[int]:
        """
        Get indices of windows to use for training.
        
        Args:
            current_window_idx: Current window index
            n_windows: Total number of windows
            
        Returns:
            List of window indices to include in training
        """
        start_idx = max(0, current_window_idx - self.window_size + 1)
        return list(range(start_idx, current_window_idx + 1))


class StrategyExecutor:
    """Execute retraining strategies and track performance."""
    
    def __init__(self, evaluator: Optional[ModelEvaluator] = None):
        """
        Initialize executor.
        
        Args:
            evaluator: ModelEvaluator instance
        """
        self.evaluator = evaluator or ModelEvaluator()
    
    def execute_strategy(
        self,
        strategy: RetrainingStrategy,
        splits: List[Dict],
        initial_model_class: type = ModelTrainer,
    ) -> Tuple[pd.DataFrame, List[int]]:
        """
        Execute a retraining strategy on all windows.
        
        Args:
            strategy: RetrainingStrategy instance
            splits: List of split dicts
            initial_model_class: Model class to use
            
        Returns:
            (performance_df, retrain_windows)
        """
        # Train initial model on reference window
        ref_split = splits[0]
        strategy.retrain(ref_split['X_ref'], ref_split['y_ref'], window_idx=0)
        
        results = []
        
        for split in splits:
            window_idx = split['window_idx']
            
            # Check if retraining is needed
            if strategy.should_retrain(window_idx):
                strategy.retrain(split['X_test'], split['y_test'], window_idx)
            
            # Evaluate current model
            y_pred = strategy.model.predict(split['X_test'])
            y_proba = strategy.model.predict_proba(split['X_test'])
            
            metrics = self.evaluator.evaluate_window(
                split['y_test'], y_pred, y_proba
            )
            metrics['window'] = window_idx
            
            results.append(metrics)
        
        performance_df = pd.DataFrame(results)
        retrain_windows = strategy.get_retrain_windows()
        
        return performance_df, retrain_windows
    
    def execute_multiple_strategies(
        self,
        strategies: Dict[str, RetrainingStrategy],
        splits: List[Dict],
    ) -> Dict[str, Tuple[pd.DataFrame, List[int]]]:
        """
        Execute multiple strategies and collect results.
        
        Args:
            strategies: Dict mapping strategy names to instances
            splits: List of split dicts
            
        Returns:
            Dict mapping strategy names to (performance_df, retrain_windows)
        """
        results = {}
        
        for strategy_name, strategy in strategies.items():
            logger.info(f"Executing strategy: {strategy_name}")
            perf_df, retrain_windows = self.execute_strategy(strategy, splits)
            results[strategy_name] = (perf_df, retrain_windows)
        
        return results
    
    def compare_strategies(
        self,
        strategy_results: Dict[str, Tuple[pd.DataFrame, List[int]]],
    ) -> pd.DataFrame:
        """
        Compare strategies on key metrics.
        
        Args:
            strategy_results: Dict from execute_multiple_strategies
            
        Returns:
            Comparison dataframe
        """
        comparisons = []
        
        for strategy_name, (perf_df, retrain_windows) in strategy_results.items():
            # Get average metrics
            numeric_cols = perf_df.select_dtypes(include=[np.number]).columns
            avg_metrics = perf_df[numeric_cols].mean().to_dict()
            
            comparison = {
                'strategy': strategy_name,
                'n_retrains': len(retrain_windows),
                **avg_metrics,
            }
            comparisons.append(comparison)
        
        df = pd.DataFrame(comparisons)
        logger.info(f"Compared {len(comparisons)} strategies")
        
        return df


class NoRetrainingStrategy(RestrainingStrategy):
    """Baseline: train once, never retrain."""

    def __init__(self, model_type: str = 'random_forest'):
        super().__init__(model_type)
        self.strategy_name = "No Retraining"

    def should_retrain(self, window_id: int, **kwargs) -> bool:
        return False

    def apply(
        self,
        windows: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> pd.DataFrame:
        """
        Train on reference window, evaluate on all test windows.
        """
        logger.info(f"Applying {self.strategy_name}...")
        
        X_ref, X_test, y_ref, y_test = windows[0]
        self.retrain(X_ref, y_ref, window_id=0)
        
        # Evaluate on all windows
        perf_df = ModelEvaluator.evaluate_on_windows(self.model, windows)
        perf_df['strategy'] = self.strategy_name
        perf_df['retrained'] = perf_df['window'].isin(self.retrain_history)
        
        self.performance_history = perf_df.to_dict('records')
        return perf_df


class PeriodicRetrainingStrategy(RetrainingStrategy):
    """Retrain every K windows."""

    def __init__(self, model_type: str = 'random_forest', retrain_period: int = 1):
        super().__init__(model_type)
        self.retrain_period = retrain_period
        self.strategy_name = f"Periodic (K={retrain_period})"

    def should_retrain(self, window_id: int, **kwargs) -> bool:
        if window_id == 0:
            return True
        return window_id % self.retrain_period == 0

    def apply(
        self,
        windows: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> pd.DataFrame:
        """Apply periodic retraining strategy."""
        logger.info(f"Applying {self.strategy_name}...")
        
        results = []
        
        for window_id, (X_ref, X_test, y_ref, y_test) in enumerate(windows):
            # Check if should retrain
            if self.should_retrain(window_id):
                self.retrain(X_ref, y_ref, window_id)
            
            # Evaluate
            if self.model is not None:
                y_pred = self.model.predict(X_test)
                y_proba = self.model.predict_proba(X_test)[:, 1]
                metrics = ModelEvaluator.compute_metrics(y_test, y_pred, y_proba)
                metrics['window'] = window_id
                metrics['retrained'] = window_id in self.retrain_history
                metrics['strategy'] = self.strategy_name
                results.append(metrics)
        
        perf_df = pd.DataFrame(results)
        self.performance_history = results
        return perf_df


class TriggerBasedRetrainingStrategy(RetrainingStrategy):
    """Retrain when drift metric exceeds threshold."""

    def __init__(
        self,
        model_type: str = 'random_forest',
        drift_threshold: float = 0.25,
        drift_metric: str = 'psi'
    ):
        super().__init__(model_type)
        self.drift_threshold = drift_threshold
        self.drift_metric = drift_metric
        self.strategy_name = f"Trigger-Based (threshold={drift_threshold})"

    def should_retrain(self, window_id: int, drift_score: float = None, **kwargs) -> bool:
        if window_id == 0:
            return True
        if drift_score is None:
            return False
        return drift_score > self.drift_threshold

    def apply(
        self,
        windows: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        drift_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Apply trigger-based retraining.
        
        Parameters:
        -----------
        windows : List[Tuple]
            Time windows
        drift_df : pd.DataFrame
            Drift scores per window
        """
        logger.info(f"Applying {self.strategy_name}...")
        
        if drift_df is None:
            logger.warning("No drift data provided, falling back to periodic retraining")
            return pd.DataFrame()
        
        # Compute aggregated drift per window
        drift_agg = drift_df.groupby('window')[self.drift_metric].mean()
        
        results = []
        
        for window_id, (X_ref, X_test, y_ref, y_test) in enumerate(windows):
            # Get drift score for this window
            drift_score = drift_agg.get(window_id, 0)
            
            # Check if should retrain
            if self.should_retrain(window_id, drift_score=drift_score):
                self.retrain(X_ref, y_ref, window_id)
            
            # Evaluate
            if self.model is not None:
                y_pred = self.model.predict(X_test)
                y_proba = self.model.predict_proba(X_test)[:, 1]
                metrics = ModelEvaluator.compute_metrics(y_test, y_pred, y_proba)
                metrics['window'] = window_id
                metrics['drift_score'] = drift_score
                metrics['retrained'] = window_id in self.retrain_history
                metrics['strategy'] = self.strategy_name
                results.append(metrics)
        
        perf_df = pd.DataFrame(results)
        self.performance_history = results
        return perf_df


class SlidingWindowRetrainingStrategy(RetrainingStrategy):
    """Retrain on the last N windows of accumulated data."""

    def __init__(self, model_type: str = 'random_forest', window_size: int = 1):
        super().__init__(model_type)
        self.window_size = window_size
        self.strategy_name = f"Sliding Window (N={window_size})"
        self.accumulated_data = []

    def apply(
        self,
        windows: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]
    ) -> pd.DataFrame:
        """Apply sliding window retraining."""
        logger.info(f"Applying {self.strategy_name}...")
        
        results = []
        
        for window_id, (X_ref, X_test, y_ref, y_test) in enumerate(windows):
            # Accumulate training data
            if window_id == 0:
                self.accumulated_data = [(X_ref, y_ref)]
            else:
                self.accumulated_data.append((X_ref, y_ref))
            
            # Keep only last N windows
            if len(self.accumulated_data) > self.window_size:
                self.accumulated_data = self.accumulated_data[-self.window_size:]
            
            # Retrain on accumulated data
            X_train = np.vstack([X for X, _ in self.accumulated_data])
            y_train = np.hstack([y for _, y in self.accumulated_data])
            
            self.retrain(X_train, y_train, window_id)
            
            # Evaluate
            if self.model is not None:
                y_pred = self.model.predict(X_test)
                y_proba = self.model.predict_proba(X_test)[:, 1]
                metrics = ModelEvaluator.compute_metrics(y_test, y_pred, y_proba)
                metrics['window'] = window_id
                metrics['retrained'] = True  # Always retraining in sliding window
                metrics['strategy'] = self.strategy_name
                results.append(metrics)
        
        perf_df = pd.DataFrame(results)
        self.performance_history = results
        return perf_df


class StrategyComparator:
    """Compare multiple retraining strategies."""

    def __init__(self):
        self.strategies = {}
        self.results = {}

    def add_strategy(self, strategy_name: str, strategy: RetrainingStrategy) -> None:
        """Add strategy to comparison."""
        self.strategies[strategy_name] = strategy

    def run_comparison(
        self,
        windows: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        drift_df: pd.DataFrame = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Run all strategies and collect results.
        """
        logger.info(f"Running {len(self.strategies)} strategies...")
        
        for strategy_name, strategy in self.strategies.items():
            logger.info(f"Running {strategy_name}...")
            
            if isinstance(strategy, TriggerBasedRetrainingStrategy):
                perf_df = strategy.apply(windows, drift_df=drift_df)
            else:
                perf_df = strategy.apply(windows)
            
            self.results[strategy_name] = perf_df
        
        return self.results

    def get_comparison_summary(self, metric: str = 'auc_roc') -> pd.DataFrame:
        """
        Create comparison summary across all strategies.
        """
        summary_list = []
        
        for strategy_name, perf_df in self.results.items():
            n_retrains = perf_df['retrained'].sum() if 'retrained' in perf_df.columns else 0
            
            summary = {
                'strategy': strategy_name,
                'mean_metric': perf_df[metric].mean(),
                'std_metric': perf_df[metric].std(),
                'final_metric': perf_df[metric].iloc[-1],
                'degradation': perf_df[metric].iloc[0] - perf_df[metric].iloc[-1],
                'n_retrains': n_retrains
            }
            summary_list.append(summary)
        
        comparison_df = pd.DataFrame(summary_list)
        comparison_df = comparison_df.sort_values('mean_metric', ascending=False)
        
        logger.info(f"\nComparison Summary:\n{comparison_df.to_string()}")
        return comparison_df

    def get_cost_benefit_analysis(self, metric: str = 'auc_roc') -> pd.DataFrame:
        """
        Analyze cost (retrains) vs benefit (performance improvement).
        """
        baseline_perf = self.results[list(self.results.keys())[0]][metric].iloc[0]
        
        analysis_list = []
        
        for strategy_name, perf_df in self.results.items():
            n_retrains = perf_df['retrained'].sum() if 'retrained' in perf_df.columns else 0
            mean_perf = perf_df[metric].mean()
            improvement = mean_perf - baseline_perf
            
            if n_retrains > 0:
                cost_benefit_ratio = improvement / n_retrains
            else:
                cost_benefit_ratio = np.inf
            
            analysis_list.append({
                'strategy': strategy_name,
                'retrains': n_retrains,
                'mean_performance': mean_perf,
                'improvement': improvement,
                'cost_benefit_ratio': cost_benefit_ratio
            })
        
        analysis_df = pd.DataFrame(analysis_list)
        analysis_df = analysis_df.sort_values('cost_benefit_ratio', ascending=False)
        
        logger.info(f"\nCost-Benefit Analysis:\n{analysis_df.to_string()}")
        return analysis_df

    def export_results(self, output_dir: str) -> None:
        """Export all results to CSV files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for strategy_name, perf_df in self.results.items():
            filename = f"{strategy_name.replace(' ', '_').lower()}.csv"
            filepath = os.path.join(output_dir, filename)
            perf_df.to_csv(filepath, index=False)
            logger.info(f"Exported {strategy_name} to {filepath}")


# Example usage
if __name__ == "__main__":
    from data_loader import TemporalDataLoader
    
    # Generate data
    loader = TemporalDataLoader()
    df = loader.load_synthetic_drift_data(n_samples=500, n_features=5, n_windows=8)
    windows = loader.split_into_windows(df, n_windows=8, target_column='target')
    
    # Compare strategies
    comparator = StrategyComparator()
    comparator.add_strategy("No Retrain", NoRetrainingStrategy())
    comparator.add_strategy("Periodic K=2", PeriodicRetrainingStrategy(retrain_period=2))
    comparator.add_strategy("Periodic K=3", PeriodicRetrainingStrategy(retrain_period=3))
    comparator.add_strategy("Sliding N=2", SlidingWindowRetrainingStrategy(window_size=2))
    
    results = comparator.run_comparison(windows)
    summary = comparator.get_comparison_summary(metric='auc_roc')
    print(summary)
