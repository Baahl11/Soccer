"""
Platt scaling implementation for probability calibration.
Based on the paper: 'Probabilistic Outputs for Support Vector Machines and 
Comparisons to Regularized Likelihood Methods' by John C. Platt.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from sklearn.metrics import log_loss, brier_score_loss
from scipy.optimize import minimize
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class PlattCalibrator:
    def __init__(self, max_iter: int = 100, tol: float = 1e-8):
        """
        Initialize Platt calibrator.
        
        Args:
            max_iter: Maximum iterations for optimization
            tol: Convergence tolerance
        """
        self.max_iter = max_iter
        self.tol = tol
        self.A = 0.0  # Slope parameter
        self.B = 0.0  # Intercept parameter
        self.is_fitted = False
        self.training_size = 0
        self.last_update = None
        self.convergence_info = {}
        
    def fit(self, predictions: np.ndarray, targets: np.ndarray) -> None:
        """
        Fit Platt scaling parameters.
        
        Args:
            predictions: Raw model predictions (N samples)
            targets: Binary targets (0 or 1)
        """
        try:
            # Input validation
            if len(predictions) != len(targets):
                raise ValueError("predictions and targets must have same length")
                
            # Initialize parameters
            prior1 = float(np.sum(targets == 1))
            prior0 = float(np.sum(targets == 0))
            
            # Initialize Platt parameters
            self.A = 0.0
            self.B = np.log((prior0 + 1) / (prior1 + 1))
            
            # Define the objective function (negative log likelihood)
            def objective(params):
                A, B = params
                scores = 1 / (1 + np.exp(A * predictions + B))
                log_likelihood = np.sum(targets * np.log(scores) + 
                                     (1 - targets) * np.log(1 - scores))
                return -log_likelihood
                
            # Optimize parameters
            result = minimize(
                objective,
                x0=[self.A, self.B],
                method='Nelder-Mead',
                options={'maxiter': self.max_iter, 'tol': self.tol}
            )
            
            # Store optimized parameters
            if result.success:
                self.A, self.B = result.x
                self.is_fitted = True
                self.training_size = len(predictions)
                self.last_update = datetime.now()
                self.convergence_info = {
                    'success': True,
                    'iterations': result.nit,
                    'final_score': float(result.fun),
                    'message': result.message
                }
                
                # Calculate calibration metrics
                calibrated = self.calibrate(predictions)
                self.metrics = {
                    'log_loss': float(log_loss(targets, calibrated)),
                    'brier_score': float(brier_score_loss(targets, calibrated))
                }
                
                logger.info("✅ Platt calibration fitted successfully")
            else:
                logger.error(f"❌ Platt calibration optimization failed: {result.message}")
                
        except Exception as e:
            logger.error(f"Error fitting Platt calibration: {e}")
            self.is_fitted = False
            
    def calibrate(self, predictions: np.ndarray) -> np.ndarray:
        """
        Apply Platt scaling to new predictions.
        
        Args:
            predictions: Raw predictions to calibrate
            
        Returns:
            Calibrated probabilities
        """
        try:
            if not self.is_fitted:
                logger.warning("⚠️ Platt calibrator not fitted, returning raw predictions")
                return predictions
                
            # Apply sigmoid with fitted parameters
            return 1 / (1 + np.exp(self.A * predictions + self.B))
            
        except Exception as e:
            logger.error(f"Error applying Platt calibration: {e}")
            return predictions
            
    def get_calibration_info(self) -> Dict[str, Any]:
        """Get information about the calibration state"""
        return {
            'is_fitted': self.is_fitted,
            'training_samples': self.training_size,
            'last_update': self.last_update.isoformat() if self.last_update else None,
            'parameters': {
                'A': float(self.A),
                'B': float(self.B)
            },
            'convergence_info': self.convergence_info,
            'metrics': getattr(self, 'metrics', {}),
            'max_iter': self.max_iter,
            'tolerance': self.tol
        }
