"""
Class Balancing Module using SMOTE

This module implements Synthetic Minority Oversampling Technique (SMOTE)
and related methods to address class imbalance in soccer match prediction,
particularly for draw outcomes which are typically underrepresented.

Key features:
1. SMOTE oversampling for minority classes
2. Borderline-SMOTE for difficult cases
3. ADASYN adaptive sampling
4. Custom soccer-specific sampling strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

class SoccerSMOTE:
    """
    Enhanced SMOTE implementation specifically designed for soccer match prediction
    with domain-specific constraints and features.
    """
    
    def __init__(self, sampling_strategy: Union[str, Dict] = 'auto', 
                 k_neighbors: int = 5, random_state: int = 42):
        """
        Initialize the SoccerSMOTE balancer.
        
        Args:
            sampling_strategy: Strategy for resampling ('auto', 'minority', 'all', or dict)
            k_neighbors: Number of nearest neighbors for SMOTE
            random_state: Random seed for reproducibility
        """
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.smote_method = None
        self.is_fitted = False
        self.class_distribution_before: Optional[Counter] = None
        self.class_distribution_after: Optional[Counter] = None
        
    def _validate_soccer_features(self, X: np.ndarray, feature_names: Optional[List[str]] = None) -> None:
        """
        Validate that features make sense for soccer context.
        
        Args:
            X: Feature matrix
            feature_names: Optional list of feature names
        """
        if feature_names is None:
            return
            
        # Check for common soccer features
        soccer_features = [
            'elo_home', 'elo_away', 'home_advantage', 'goals_for_home', 
            'goals_against_home', 'form_home', 'form_away', 'h2h_home_wins',
            'league_position_home', 'league_position_away'
        ]
        
        found_features = [f for f in soccer_features if f in feature_names]
        if len(found_features) < 3:            logger.warning(f"Only {len(found_features)} soccer-specific features found. "
                         "Consider adding more domain-specific features.")
    
    def _create_balanced_sampling_strategy(self, y: np.ndarray) -> Union[Dict[int, int], str]:
        """
        Create a sampling strategy that respects soccer match outcome distributions.
        
        Args:
            y: Target labels
            
        Returns:
            sampling_strategy: Dictionary with target counts for each class or 'auto'
        """
        class_counts = Counter(y)
        total_samples = len(y)
        
        # Typical soccer outcome distributions
        # Home win: ~45%, Draw: ~25%, Away win: ~30%
        target_proportions = {
            0: 0.45,  # Home win
            1: 0.25,  # Draw
            2: 0.30   # Away win
        }
        
        # Calculate target counts based on current dataset size
        strategy = {}
        for class_label, target_prop in target_proportions.items():
            if class_label in class_counts:
                target_count = int(total_samples * target_prop)
                current_count = class_counts[class_label]
                
                # Only oversample if current count is less than target
                if current_count < target_count:
                    strategy[class_label] = target_count
        
        return strategy if strategy else 'auto'
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray, 
                     feature_names: Optional[List[str]] = None,
                     method: str = 'smote') -> Tuple[Any, Any]:
        """
        Fit the SMOTE model and resample the data.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional list of feature names
            method: SMOTE variant ('smote', 'borderline', 'adasyn', 'smote_tomek', 'smote_enn')
            
        Returns:
            X_resampled: Resampled feature matrix
            y_resampled: Resampled target labels
        """
        self._validate_soccer_features(X, feature_names)
        
        # Store original class distribution
        self.class_distribution_before = Counter(y)
        
        # Create sampling strategy
        if isinstance(self.sampling_strategy, str) and self.sampling_strategy == 'soccer':
            sampling_strategy = self._create_balanced_sampling_strategy(y)
        else:
            sampling_strategy = self.sampling_strategy
          # Initialize SMOTE method
        if method == 'smote':
            self.smote_method = SMOTE(
                sampling_strategy=sampling_strategy,  # type: ignore
                k_neighbors=self.k_neighbors,
                random_state=self.random_state
            )
        elif method == 'borderline':
            self.smote_method = BorderlineSMOTE(
                sampling_strategy=sampling_strategy,  # type: ignore
                k_neighbors=self.k_neighbors,
                random_state=self.random_state
            )
        elif method == 'adasyn':
            self.smote_method = ADASYN(
                sampling_strategy=sampling_strategy,  # type: ignore
                n_neighbors=self.k_neighbors,
                random_state=self.random_state
            )
        elif method == 'smote_tomek':
            self.smote_method = SMOTETomek(
                sampling_strategy=sampling_strategy,  # type: ignore
                smote=SMOTE(k_neighbors=self.k_neighbors, random_state=self.random_state),
                random_state=self.random_state
            )
        elif method == 'smote_enn':
            self.smote_method = SMOTEENN(
                sampling_strategy=sampling_strategy,  # type: ignore
                smote=SMOTE(k_neighbors=self.k_neighbors, random_state=self.random_state),
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown SMOTE method: {method}")
        
        # Apply resampling
        logger.info(f"Applying {method} resampling...")
        logger.info(f"Original class distribution: {dict(self.class_distribution_before)}")
        
        try:
            result = self.smote_method.fit_resample(X, y)
            if len(result) == 2:
                X_resampled, y_resampled = result
            else:
                # Some methods might return additional info, take first two
                X_resampled, y_resampled = result[0], result[1]
            
            # Store new class distribution
            self.class_distribution_after = Counter(y_resampled)
            self.is_fitted = True
            
            logger.info(f"New class distribution: {dict(self.class_distribution_after)}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"SMOTE resampling failed: {str(e)}")
            logger.info("Returning original data without resampling")
            return X, y
    
    def get_resampling_report(self) -> Dict[str, Any]:
        """
        Generate a detailed report on the resampling process.
        
        Returns:
            report: Dictionary with resampling statistics
        """
        if not self.is_fitted or self.class_distribution_before is None or self.class_distribution_after is None:
            return {"error": "SMOTE not fitted yet"}
        
        # Calculate changes
        changes = {}
        for class_label in set(list(self.class_distribution_before.keys()) + 
                              list(self.class_distribution_after.keys())):
            before = self.class_distribution_before.get(class_label, 0)
            after = self.class_distribution_after.get(class_label, 0)
            changes[class_label] = {
                'before': before,
                'after': after,
                'change': after - before,
                'ratio_change': after / before if before > 0 else float('inf')
            }
        
        total_before = sum(self.class_distribution_before.values())
        total_after = sum(self.class_distribution_after.values())
        
        report = {
            'total_samples': {
                'before': total_before,
                'after': total_after,
                'synthetic_added': total_after - total_before
            },
            'class_changes': changes,
            'balance_improvement': self._calculate_balance_improvement(),
            'method_used': type(self.smote_method).__name__        }
        
        return report
    
    def _calculate_balance_improvement(self) -> Dict[str, Any]:
        """
        Calculate how much the class balance improved.
        
        Returns:
            balance_metrics: Dictionary with balance improvement metrics
        """
        if self.class_distribution_before is None or self.class_distribution_after is None:
            return {
                'gini_before': 0.0,
                'gini_after': 0.0,
                'gini_improvement': 0.0,
                'std_before': 0.0,
                'std_after': 0.0,
                'std_improvement': 0.0
            }
            
        def gini_coefficient(counts):
            """Calculate Gini coefficient for class distribution."""
            counts = list(counts.values())
            total = sum(counts)
            if total == 0:
                return 0
            
            proportions = [c / total for c in counts]
            proportions.sort()
            n = len(proportions)
            
            gini = 2 * sum((i + 1) * p for i, p in enumerate(proportions)) / (n * sum(proportions)) - (n + 1) / n
            return gini
        
        gini_before = gini_coefficient(self.class_distribution_before)
        gini_after = gini_coefficient(self.class_distribution_after)
        
        # Calculate standard deviation of class proportions
        total_before = sum(self.class_distribution_before.values())
        total_after = sum(self.class_distribution_after.values())
        
        props_before = [count / total_before for count in self.class_distribution_before.values()]
        props_after = [count / total_after for count in self.class_distribution_after.values()]
        
        std_before = np.std(props_before)
        std_after = np.std(props_after)
        
        return {
            'gini_before': gini_before,
            'gini_after': gini_after,
            'gini_improvement': gini_before - gini_after,
            'std_before': std_before,
            'std_after': std_after,
            'std_improvement': std_before - std_after
        }
    
    def plot_class_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Plot class distribution before and after resampling.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.is_fitted or self.class_distribution_before is None or self.class_distribution_after is None:
            logger.error("Cannot plot distribution - SMOTE not fitted")
            return
        
        class_labels = sorted(set(list(self.class_distribution_before.keys()) + 
                                 list(self.class_distribution_after.keys())))
        
        before_counts = [self.class_distribution_before.get(label, 0) for label in class_labels]
        after_counts = [self.class_distribution_after.get(label, 0) for label in class_labels]
        
        x = np.arange(len(class_labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, before_counts, width, label='Before SMOTE', alpha=0.7)
        bars2 = ax.bar(x + width/2, after_counts, width, label='After SMOTE', alpha=0.7)
        
        ax.set_xlabel('Class')
        ax.set_ylabel('Number of Samples')
        ax.set_title('Class Distribution Before and After SMOTE')
        ax.set_xticks(x)
        ax.set_xticklabels(['Home Win', 'Draw', 'Away Win'] if len(class_labels) == 3 
                          else [f'Class {i}' for i in class_labels])
        ax.legend()
        
        # Add value labels on bars
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{int(height)}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')
        
        autolabel(bars1)
        autolabel(bars2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {save_path}")
        else:
            plt.show()


class BalancedTrainingPipeline:
    """
    Complete pipeline for balanced training with SMOTE integration.
    """
    
    def __init__(self, smote_method: str = 'smote', 
                 validation_split: float = 0.2,
                 random_state: int = 42):
        """
        Initialize the balanced training pipeline.
        
        Args:
            smote_method: SMOTE variant to use
            validation_split: Fraction of data to use for validation
            random_state: Random seed
        """
        self.smote_method = smote_method
        self.validation_split = validation_split
        self.random_state = random_state
        self.soccer_smote = None
    
    def prepare_balanced_dataset(self, X: np.ndarray, y: np.ndarray,
                                feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Prepare a balanced dataset for training with proper train/validation split.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: Optional list of feature names
            
        Returns:
            dataset_splits: Dictionary with train/validation splits and metadata
        """
        from sklearn.model_selection import train_test_split
        
        # Split into train/validation before applying SMOTE
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=self.validation_split, 
            random_state=self.random_state, stratify=y
        )
        
        # Apply SMOTE only to training data
        self.soccer_smote = SoccerSMOTE(
            sampling_strategy='soccer',
            random_state=self.random_state
        )
        
        X_train_balanced, y_train_balanced = self.soccer_smote.fit_resample(
            X_train, y_train, feature_names, method=self.smote_method
        )
        
        # Get resampling report
        report = self.soccer_smote.get_resampling_report()
        
        return {
            'X_train': X_train_balanced,
            'y_train': y_train_balanced,
            'X_val': X_val,
            'y_val': y_val,
            'original_train_size': len(X_train),
            'balanced_train_size': len(X_train_balanced),
            'synthetic_samples_added': len(X_train_balanced) - len(X_train),
            'resampling_report': report
        }


if __name__ == "__main__":
    # Example usage with synthetic soccer data
    logging.basicConfig(level=logging.INFO)
    
    # Generate synthetic soccer match data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Create features (ELO ratings, form, etc.)
    X = np.random.randn(n_samples, n_features)
    
    # Create imbalanced target (fewer draws)
    y = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.15, 0.35])  # Imbalanced
    
    feature_names = [
        'elo_home', 'elo_away', 'form_home', 'form_away', 
        'goals_for_home', 'goals_against_home', 'h2h_home_wins',
        'league_position_home', 'league_position_away', 'home_advantage'
    ]
    
    print("Original class distribution:", Counter(y))
    
    # Test different SMOTE methods
    methods = ['smote', 'borderline', 'adasyn']
    
    for method in methods:
        print(f"\n--- Testing {method.upper()} ---")
        soccer_smote = SoccerSMOTE(sampling_strategy='soccer')
        X_resampled, y_resampled = soccer_smote.fit_resample(X, y, feature_names, method)
        
        report = soccer_smote.get_resampling_report()
        print(f"Synthetic samples added: {report['total_samples']['synthetic_added']}")
        print(f"Balance improvement (Gini): {report['balance_improvement']['gini_improvement']:.4f}")
        
    # Test balanced training pipeline
    print("\n--- Testing Balanced Training Pipeline ---")
    pipeline = BalancedTrainingPipeline(smote_method='smote')
    dataset_splits = pipeline.prepare_balanced_dataset(X, y, feature_names)
    
    print(f"Original training set size: {dataset_splits['original_train_size']}")
    print(f"Balanced training set size: {dataset_splits['balanced_train_size']}")
    print(f"Validation set size: {len(dataset_splits['X_val'])}")
