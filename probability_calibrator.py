from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import logging

class ProbabilityCalibrator:
    """
    Handles probability calibration and class balancing for the 1X2 prediction system.
    Includes:
    - Probability calibration using various methods
    - Class balancing using SMOTE
    - Validation and performance monitoring
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.label_encoder = LabelEncoder()
        
    def calibrate_probabilities(self, 
                              base_model: Any,
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              method: str = 'sigmoid') -> Tuple[Any, Dict[str, float]]:
        """
        Calibrate prediction probabilities using specified method
        
        Args:
            base_model: The base classifier to calibrate
            X_train: Training features
            y_train: Training labels
            method: Calibration method ('sigmoid' or 'isotonic')
            
        Returns:
            Tuple of (calibrated_model, calibration_metrics)
        """
        try:
            # Encode labels if needed
            if y_train.dtype == object:
                y_train = self.label_encoder.fit_transform(y_train)
            
            # Create and fit calibrated classifier
            calibrated_model = CalibratedClassifierCV(
                base_model,
                method=method,
                cv=5
            )
            calibrated_model.fit(X_train, y_train)
            
            # Calculate calibration metrics
            metrics = self._calculate_calibration_metrics(
                calibrated_model, X_train, y_train
            )
            
            return calibrated_model, metrics
            
        except Exception as e:
            self.logger.error(f"Error in probability calibration: {e}")
            return base_model, {}
            
    def balance_classes(self, 
                       X: np.ndarray,
                       y: np.ndarray,
                       sampling_strategy: str = 'auto') -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance classes using SMOTE
        
        Args:
            X: Features array
            y: Labels array
            sampling_strategy: SMOTE sampling strategy
            
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        try:
            # Encode labels if needed
            if y.dtype == object:
                y = self.label_encoder.fit_transform(y)
            
            # Apply SMOTE
            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            # Calculate balancing metrics
            balance_metrics = self._calculate_balance_metrics(y, y_resampled)
            self.logger.info(f"Class balance metrics: {balance_metrics}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            self.logger.error(f"Error in class balancing: {e}")
            return X, y
            
    def _calculate_calibration_metrics(self,
                                     model: Any,
                                     X: np.ndarray,
                                     y: np.ndarray) -> Dict[str, float]:
        """Calculate metrics to assess probability calibration quality"""
        try:
            # Get predicted probabilities
            y_prob = model.predict_proba(X)
            
            # Calculate Brier score
            brier_scores = []
            for i in range(len(np.unique(y))):
                y_true_bin = (y == i).astype(int)
                brier_scores.append(np.mean((y_prob[:, i] - y_true_bin) ** 2))
            
            # Calculate confidence metrics
            confidence = np.max(y_prob, axis=1)
            accuracy = (model.predict(X) == y).astype(float)
            
            return {
                'brier_score_mean': np.mean(brier_scores),
                'confidence_mean': np.mean(confidence),
                'confidence_accuracy_correlation': np.corrcoef(confidence, accuracy)[0, 1]
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating calibration metrics: {e}")
            return {}
            
    def _calculate_balance_metrics(self,
                                 y_original: np.ndarray,
                                 y_resampled: np.ndarray) -> Dict[str, Any]:
        """Calculate metrics to assess class balance improvement"""
        try:
            original_counts = np.bincount(y_original)
            resampled_counts = np.bincount(y_resampled)
            
            original_ratios = original_counts / len(y_original)
            resampled_ratios = resampled_counts / len(y_resampled)
            
            return {
                'original_distribution': dict(enumerate(original_ratios)),
                'resampled_distribution': dict(enumerate(resampled_ratios)),
                'imbalance_reduction': {
                    'before': max(original_ratios) / min(original_ratios),
                    'after': max(resampled_ratios) / min(resampled_ratios)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating balance metrics: {e}")
            return {}
