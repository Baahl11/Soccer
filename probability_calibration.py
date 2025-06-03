"""
Probability Calibration Module

This module implements advanced probability calibration techniques including
Platt scaling to improve the reliability of probability predictions from
the ensemble model.

Key features:
1. Platt scaling calibration
2. Isotonic regression calibration
3. Cross-validation for calibration assessment
4. Calibration curve plotting
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import joblib
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ProbabilityCalibrator:
    """
    Advanced probability calibrator using multiple techniques to improve
    the reliability of match outcome probability predictions.
    """
    
    def __init__(self, method: str = 'platt', cv_folds: int = 5):
        """
        Initialize the probability calibrator.
        
        Args:
            method: Calibration method ('platt' or 'isotonic')
            cv_folds: Number of cross-validation folds for calibration        """
        self.method = method
        self.cv_folds = cv_folds
        self.calibrators = {}
        self.is_fitted = False
        self.calibration_history = []
        
    def fit(self, predictions: np.ndarray, true_outcomes: np.ndarray, 
            outcome_names: Optional[List[str]] = None) -> 'ProbabilityCalibrator':
        """
        Fit calibration models for each outcome type.
        
        Args:
            predictions: Array of shape (n_samples, n_outcomes) with raw probabilities
            true_outcomes: Array of shape (n_samples,) with true outcome indices
            outcome_names: Names for each outcome (e.g., ['home_win', 'draw', 'away_win'])
            
        Returns:
            self: Fitted calibrator instance
        """
        if outcome_names is None:
            outcome_names = [f'outcome_{i}' for i in range(predictions.shape[1])]
            
        self.outcome_names = outcome_names
        n_outcomes = predictions.shape[1]
        
        logger.info(f"Fitting {self.method} calibration for {n_outcomes} outcomes")
        
        # Fit calibrator for each outcome using one-vs-rest approach
        for i, outcome_name in enumerate(outcome_names):
            logger.info(f"Calibrating probabilities for {outcome_name}")
            
            # Create binary labels for current outcome
            binary_labels = (true_outcomes == i).astype(int)
            outcome_probs = predictions[:, i]
              # Create dummy classifier that always predicts the probability
            class DummyClassifier(BaseEstimator, ClassifierMixin):
                def __init__(self, probabilities):
                    self.probabilities = probabilities
                    
                def fit(self, X, y):
                    return self
                    
                def predict_proba(self, X=None):
                    # Return probabilities as [1-p, p] format expected by sklearn
                    probs = self.probabilities.reshape(-1, 1)
                    return np.column_stack([1 - probs, probs])
                    
                def predict(self, X=None):
                    return (self.probabilities > 0.5).astype(int)
            
            dummy_clf = DummyClassifier(outcome_probs)
            
            # Apply calibration
            if self.method == 'platt':
                calibrator = CalibratedClassifierCV(
                    dummy_clf, 
                    method='sigmoid', 
                    cv=self.cv_folds
                )
            elif self.method == 'isotonic':
                calibrator = CalibratedClassifierCV(
                    dummy_clf, 
                    method='isotonic', 
                    cv=self.cv_folds
                )
            else:
                raise ValueError(f"Unknown calibration method: {self.method}")
            
            # Fit calibrator
            # Create dummy X data since we're using probabilities directly
            X_dummy = np.zeros((len(outcome_probs), 1))
            calibrator.fit(X_dummy, binary_labels)
            
            self.calibrators[outcome_name] = {
                'calibrator': calibrator,
                'original_probs': outcome_probs,
                'binary_labels': binary_labels
            }
            
        self.is_fitted = True
        logger.info("Probability calibration fitting completed")
        return self
        
    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply calibration to new probability predictions.
        
        Args:
            predictions: Array of shape (n_samples, n_outcomes) with raw probabilities
            
        Returns:
            calibrated_predictions: Array of calibrated probabilities
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before calibrating predictions")
            
        n_samples, n_outcomes = predictions.shape
        calibrated_probs = np.zeros_like(predictions)
        
        # Apply calibration to each outcome
        for i, outcome_name in enumerate(self.outcome_names):
            calibrator = self.calibrators[outcome_name]['calibrator']
            
            # Create dummy X data
            X_dummy = np.zeros((n_samples, 1))
            
            # Get calibrated probabilities (take the positive class probability)
            calibrated_probs[:, i] = calibrator.predict_proba(X_dummy)[:, 1]
        
        # Normalize probabilities to sum to 1
        row_sums = calibrated_probs.sum(axis=1)
        calibrated_probs = calibrated_probs / row_sums.reshape(-1, 1)
        
        return calibrated_probs
        
    def evaluate_calibration(self, predictions: np.ndarray, true_outcomes: np.ndarray,
                           n_bins: int = 10) -> Dict[str, Any]:
        """
        Evaluate calibration quality using reliability diagrams and calibration error.
        
        Args:
            predictions: Array of predicted probabilities
            true_outcomes: Array of true outcome indices
            n_bins: Number of bins for calibration curve
            
        Returns:
            calibration_metrics: Dictionary with calibration evaluation results
        """
        if not self.is_fitted:
            raise ValueError("Calibrator must be fitted before evaluation")
            
        results = {}
        
        for i, outcome_name in enumerate(self.outcome_names):
            binary_labels = (true_outcomes == i).astype(int)
            outcome_probs = predictions[:, i]
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                binary_labels, outcome_probs, n_bins=n_bins
            )
            
            # Calculate Expected Calibration Error (ECE)
            bin_boundaries = np.linspace(0, 1, n_bins + 1)
            bin_lowers = bin_boundaries[:-1]
            bin_uppers = bin_boundaries[1:]
            
            ece = 0
            for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                in_bin = (outcome_probs > bin_lower) & (outcome_probs <= bin_upper)
                prop_in_bin = in_bin.mean()
                
                if prop_in_bin > 0:
                    accuracy_in_bin = binary_labels[in_bin].mean()
                    avg_confidence_in_bin = outcome_probs[in_bin].mean()
                    ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            results[outcome_name] = {
                'fraction_of_positives': fraction_of_positives,
                'mean_predicted_value': mean_predicted_value,
                'expected_calibration_error': ece,
                'reliability_curve_data': {
                    'fraction_positives': fraction_of_positives.tolist(),
                    'mean_predicted': mean_predicted_value.tolist()
                }
            }
            
        return results
        
    def plot_calibration_curves(self, predictions: np.ndarray, true_outcomes: np.ndarray,
                               save_path: Optional[str] = None) -> None:
        """
        Plot calibration curves for all outcomes.
        
        Args:
            predictions: Array of predicted probabilities
            true_outcomes: Array of true outcome indices
            save_path: Optional path to save the plot
        """
        n_outcomes = len(self.outcome_names)
        fig, axes = plt.subplots(1, n_outcomes, figsize=(5 * n_outcomes, 5))
        
        if n_outcomes == 1:
            axes = [axes]
            
        for i, (outcome_name, ax) in enumerate(zip(self.outcome_names, axes)):
            binary_labels = (true_outcomes == i).astype(int)
            outcome_probs = predictions[:, i]
            
            # Plot calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                binary_labels, outcome_probs, n_bins=10
            )
            
            ax.plot(mean_predicted_value, fraction_of_positives, 
                   marker='o', linewidth=2, label=f'{outcome_name} calibration')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', 
                   label='Perfect calibration')
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'Calibration Curve - {outcome_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration curves saved to {save_path}")
        else:
            plt.show()
            
    def save_calibrator(self, filepath: str) -> None:
        """
        Save the fitted calibrator to disk.
        
        Args:
            filepath: Path to save the calibrator
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted calibrator")
            
        calibrator_data = {
            'method': self.method,
            'cv_folds': self.cv_folds,
            'calibrators': self.calibrators,
            'outcome_names': self.outcome_names,
            'is_fitted': self.is_fitted,
            'fit_timestamp': datetime.now().isoformat()
        }
        
        joblib.dump(calibrator_data, filepath)
        logger.info(f"Calibrator saved to {filepath}")
        
    @classmethod
    def load_calibrator(cls, filepath: str) -> 'ProbabilityCalibrator':
        """
        Load a fitted calibrator from disk.
        
        Args:
            filepath: Path to the saved calibrator
            
        Returns:
            calibrator: Loaded calibrator instance
        """
        calibrator_data = joblib.load(filepath)
        
        calibrator = cls(
            method=calibrator_data['method'],
            cv_folds=calibrator_data['cv_folds']
        )
        
        calibrator.calibrators = calibrator_data['calibrators']
        calibrator.outcome_names = calibrator_data['outcome_names']
        calibrator.is_fitted = calibrator_data['is_fitted']
        
        logger.info(f"Calibrator loaded from {filepath}")
        return calibrator


def brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Calculate Brier score for multiclass predictions.
    
    Args:
        y_true: True class labels (shape: n_samples)
        y_prob: Predicted probabilities (shape: n_samples, n_classes)
        
    Returns:
        brier_score: Brier score (lower is better)
    """
    n_samples, n_classes = y_prob.shape
    
    # Convert true labels to one-hot encoding
    y_true_onehot = np.zeros((n_samples, n_classes))
    y_true_onehot[np.arange(n_samples), y_true] = 1
    
    # Calculate Brier score
    brier_score = np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))
    
    return brier_score


def calibration_assessment(predictions: np.ndarray, true_outcomes: np.ndarray,
                          outcome_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Comprehensive calibration assessment for probability predictions.
    
    Args:
        predictions: Array of predicted probabilities
        true_outcomes: Array of true outcome indices
        outcome_names: Names for each outcome
        
    Returns:
        assessment_results: Comprehensive calibration metrics
    """
    if outcome_names is None:
        outcome_names = [f'outcome_{i}' for i in range(predictions.shape[1])]
    
    # Calculate overall Brier score
    brier_score = brier_score_multiclass(true_outcomes, predictions)
    
    # Initialize calibrator and evaluate
    calibrator = ProbabilityCalibrator(method='platt')
    calibrator.outcome_names = outcome_names
    calibrator.is_fitted = True  # Temporarily set for evaluation
    
    # Create dummy calibrators for evaluation
    calibrator.calibrators = {}
    for i, name in enumerate(outcome_names):
        calibrator.calibrators[name] = {'calibrator': None}
    
    # Get calibration metrics
    calibration_metrics = calibrator.evaluate_calibration(predictions, true_outcomes)
    
    # Compile results
    results = {
        'overall_brier_score': brier_score,
        'outcome_calibration': calibration_metrics,
        'summary': {
            'mean_calibration_error': np.mean([
                metrics['expected_calibration_error'] 
                for metrics in calibration_metrics.values()
            ]),
            'worst_calibration_error': np.max([
                metrics['expected_calibration_error'] 
                for metrics in calibration_metrics.values()
            ])
        }
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_outcomes = 3
    
    # Create somewhat uncalibrated probabilities
    raw_probs = np.random.dirichlet([2, 1, 2], n_samples)
    true_outcomes = np.random.choice(n_outcomes, n_samples, p=[0.4, 0.25, 0.35])
    
    outcome_names = ['home_win', 'draw', 'away_win']
    
    # Fit calibrator
    calibrator = ProbabilityCalibrator(method='platt')
    calibrator.fit(raw_probs, true_outcomes, outcome_names)
    
    # Apply calibration
    calibrated_probs = calibrator.calibrate(raw_probs)
    
    # Evaluate calibration
    print("Raw probabilities assessment:")
    raw_assessment = calibration_assessment(raw_probs, true_outcomes, outcome_names)
    print(f"Brier Score: {raw_assessment['overall_brier_score']:.4f}")
    print(f"Mean Calibration Error: {raw_assessment['summary']['mean_calibration_error']:.4f}")
    
    print("\nCalibrated probabilities assessment:")
    cal_assessment = calibration_assessment(calibrated_probs, true_outcomes, outcome_names)
    print(f"Brier Score: {cal_assessment['overall_brier_score']:.4f}")
    print(f"Mean Calibration Error: {cal_assessment['summary']['mean_calibration_error']:.4f}")
