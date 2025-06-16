"""
Automated model retraining system to keep predictions up to date.
"""

import numpy as np
import schedule
import time
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime, timedelta
import json
import os
from .model_validation import validate_goals_model
from .weather_model import calculate_weather_adjustment_factors
from .xg_model import EnhancedXGModel
from .specialized_ensemble import SpecializedEnsembleModel
from .bayesian_goals_model_new import BayesianGoalsModel

logger = logging.getLogger(__name__)

class ModelTrainingScheduler:
    def __init__(
        self,
        training_data_path: str, 
        models_path: str,
        performance_threshold: float = 0.7,
        validation_size: float = 0.2,
        retraining_frequency_days: int = 7,
        n_splits: int = 5,
        gap_days: int = 7
    ):
        """
        Initialize the model retraining scheduler with temporal validation.
        
        Args:
            training_data_path: Path to training data
            models_path: Path to save trained models
            performance_threshold: Minimum required performance to accept new model
            validation_size: Portion of data to use for validation
            retraining_frequency_days: How often to retrain in days
            n_splits: Number of temporal CV splits
            gap_days: Gap in days between train and test sets
        """
        self.training_data_path = training_data_path
        self.models_path = models_path
        self.performance_threshold = performance_threshold
        self.validation_size = validation_size
        self.retraining_frequency_days = retraining_frequency_days
        self.n_splits = n_splits
        self.gap_days = gap_days
        
        # Ensure directories exist
        os.makedirs(self.models_path, exist_ok=True)
        os.makedirs(os.path.join(self.models_path, 'archive'), exist_ok=True)
        
    def setup_schedule(self):
        """Setup the retraining schedule"""
        schedule.every(self.retraining_frequency_days).days.at("02:00").do(self.retrain_models)
        
    def run(self):
        """Run the scheduler"""
        self.setup_schedule()
        while True:
            schedule.run_pending()
            time.sleep(3600)  # Check every hour
            
    def retrain_models(self) -> bool:
        """
        Retrain all models and validate performance.
        
        Returns:
            bool: True if retraining was successful
        """
        try:
            logger.info("Starting model retraining process")
            
            # Load training data
            training_data = self._load_training_data()
            if not training_data:
                raise ValueError("No training data available")
                
            # Split into train/validation
            split_idx = int(len(training_data['matches']) * (1 - self.validation_size))
            train_data = {
                'matches': training_data['matches'][:split_idx],
                'weather': training_data['weather'],
                'league_data': training_data['league_data']
            }
            val_data = {
                'matches': training_data['matches'][split_idx:],
                'weather': training_data['weather'],
                'league_data': training_data['league_data']
            }
            
            # Train models
            models = self._train_all_models(train_data)
            
            # Validate new models
            validation_results = validate_goals_model(
                val_data['matches'],
                val_data['weather'],
                val_data['league_data']
            )
            
            # Check if performance is acceptable
            if self._check_performance(validation_results):
                # Archive old models
                self._archive_old_models()
                
                # Save new models
                self._save_models(models)
                
                # Save validation results
                self._save_validation_results(validation_results)
                
                logger.info("Model retraining complete and validated successfully")
                return True
            else:
                logger.warning("New models did not meet performance threshold")
                return False
                
        except Exception as e:
            logger.error(f"Error in model retraining: {e}")
            return False
            
    def _load_training_data(self) -> Optional[Dict[str, Any]]:
        """Load training data from file"""
        try:
            with open(self.training_data_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return None
            
    def _train_all_models(self, train_data: Dict[str, Any]) -> Dict[str, Any]:
        """Train all required models using temporal cross-validation"""
        from time_series_cross_validation import TimeSeriesValidator
        
        models = {}
        
        try:
            # Prepare data with dates
            from time_series_cross_validation import get_train_data_with_dates
            X, y, dates = get_train_data_with_dates()
            
            # Create temporal validator
            validator = TimeSeriesValidator(
                n_splits=self.n_splits,
                gap=self.gap_days,
                verbose=True
            )
            
            # Function to create models for cross-validation
            def create_model_factory(model_class):
                def model_factory(X_train, y_train):
                    model = model_class()
                    model.fit(X_train, y_train)
                    return model
                return model_factory
            
            # Train and validate models with temporal CV
            model_classes = {
                'xg_model': EnhancedXGModel,
                'ensemble_model': SpecializedEnsembleModel,
                'bayesian_model': BayesianGoalsModel
            }
            
            validation_results = {}
            for name, model_class in model_classes.items():
                logger.info(f"Training {name} with temporal cross-validation...")
                
                # Create model factory
                factory = create_model_factory(model_class)
                
                # Run temporal validation
                results = validator.validate(
                    X=X,
                    y=y,
                    dates=dates,
                    model_factory=factory,
                    metrics=['mae', 'rmse', 'r2'],
                    save_models=True,
                    output_dir=os.path.join(self.models_path, f'temporal_cv_{name}')
                )
                
                validation_results[name] = results
                
                # Train final model on most recent data
                recent_cutoff = dates.max() - np.timedelta64(30, 'D')
                recent_mask = dates >= recent_cutoff
                
                final_model = model_class()
                final_model.fit(X[recent_mask], y[recent_mask])
                models[name] = final_model
            
            # Generate validation plots
            validator.plot_validation_scheme(
                n_samples=len(X),
                output_dir=self.models_path
            )
            validator.plot_results(output_dir=self.models_path)
            
            # Save validation results
            self._save_validation_results(validation_results)
            
            return models
            
        except Exception as e:
            logger.error(f"Error training models with temporal CV: {e}")
            raise
            
    def _check_performance(self, validation_results: Dict[str, Any]) -> bool:
        """
        Check if model performance meets stricter requirements with enhanced metrics
        
        The model must pass ALL of the following criteria:
        1. Basic regression metrics (RMSE, R², MAE)
        2. Probability calibration metrics
        3. Range-specific accuracy
        4. Trend prediction accuracy
        5. Confidence metrics
        """
        try:
            # 1. Basic regression metrics
            regression_metrics = validation_results['regression_metrics']
            rmse = regression_metrics['rmse']
            r2 = regression_metrics['r2']
            mae = regression_metrics['mae']
            mape = regression_metrics.get('mape', float('inf'))
            
            # 2. Probability metrics
            prob_metrics = validation_results.get('probability_metrics', {})
            brier_over25 = validation_results['over25_metrics']['brier_score']
            brier_btts = validation_results['btts_metrics']['brier_score']
            calibration_error = prob_metrics.get('calibration_error', float('inf'))
            sharpness = prob_metrics.get('sharpness', 0)
            
            # 3. Range-specific metrics
            range_metrics = validation_results.get('range_metrics', {})
            low_score_acc = range_metrics.get('low_scoring_accuracy', 0)  # 0-2 goals
            high_score_acc = range_metrics.get('high_scoring_accuracy', 0)  # 3+ goals
            exact_score_acc = range_metrics.get('exact_score_accuracy', 0)
            
            # 4. Trend metrics
            trend_metrics = validation_results.get('trend_metrics', {})
            trend_accuracy = trend_metrics.get('trend_accuracy', 0)
            momentum_correlation = trend_metrics.get('momentum_correlation', 0)
            
            # Define stricter thresholds
            requirements = {
                # Basic regression thresholds
                'rmse_threshold': 1.2,  # Stricter than before (was 1.5)
                'r2_threshold': 0.45,   # Significantly higher (was 0.3)
                'mae_threshold': 0.8,
                'mape_threshold': 25.0,  # 25% error maximum
                
                # Probability thresholds
                'brier_threshold': 0.20,  # Stricter than before (was 0.25)
                'calibration_error_threshold': 0.1,
                'sharpness_threshold': 0.15,
                
                # Range-specific thresholds
                'low_score_acc_threshold': 0.65,
                'high_score_acc_threshold': 0.55,
                'exact_score_threshold': 0.25,
                
                # Trend thresholds
                'trend_accuracy_threshold': 0.60,
                'momentum_correlation_threshold': 0.3
            }
            
            # Check all requirements with detailed logging
            checks = {
                'basic_regression': (
                    rmse < requirements['rmse_threshold'] and
                    r2 > requirements['r2_threshold'] and
                    mae < requirements['mae_threshold'] and
                    mape < requirements['mape_threshold']
                ),
                'probability_metrics': (
                    brier_over25 < requirements['brier_threshold'] and
                    brier_btts < requirements['brier_threshold'] and
                    calibration_error < requirements['calibration_error_threshold'] and
                    sharpness > requirements['sharpness_threshold']
                ),
                'range_accuracy': (
                    low_score_acc > requirements['low_score_acc_threshold'] and
                    high_score_acc > requirements['high_score_acc_threshold'] and
                    exact_score_acc > requirements['exact_score_threshold']
                ),
                'trend_accuracy': (
                    trend_accuracy > requirements['trend_accuracy_threshold'] and
                    momentum_correlation > requirements['momentum_correlation_threshold']
                )
            }
            
            # Log detailed performance metrics
            logger.info("Performance check results:")
            for check_name, check_passed in checks.items():
                logger.info(f"{check_name}: {'PASSED' if check_passed else 'FAILED'}")
            
            # Model must pass ALL checks
            meets_requirements = all(checks.values())
            
            if meets_requirements:
                logger.info("Model passed ALL performance checks!")
            else:
                logger.warning("Model failed one or more performance checks")
            
            return meets_requirements
            
        except Exception as e:
            logger.error(f"Error checking performance: {e}")
            return False
            
    def _archive_old_models(self):
        """Archive existing models with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = os.path.join(self.models_path, 'archive', timestamp)
            os.makedirs(archive_dir, exist_ok=True)
            
            # Move existing models to archive
            model_files = [f for f in os.listdir(self.models_path) if f.endswith('.pkl')]
            for file in model_files:
                old_path = os.path.join(self.models_path, file)
                new_path = os.path.join(archive_dir, file)
                os.rename(old_path, new_path)
                
        except Exception as e:
            logger.error(f"Error archiving old models: {e}")
            raise
            
    def _save_models(self, models: Dict[str, Any]):
        """Save new models to disk"""
        try:
            import joblib
            
            for name, model in models.items():
                model_path = os.path.join(self.models_path, f"{name}.pkl")
                joblib.dump(model, model_path)
                
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            raise
            
    def _save_validation_results(self, results: Dict[str, Any]):
        """Save validation results with timestamp"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                self.models_path,
                'validation_results',
                f'validation_{timestamp}.json'
            )
            
            os.makedirs(os.path.dirname(results_file), exist_ok=True)
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving validation results: {e}")
            raise

if __name__ == '__main__':
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize and run scheduler
    scheduler = ModelTrainingScheduler(
        training_data_path='data/training_data.json',
        models_path='models/',
        performance_threshold=0.7,
        retraining_frequency_days=7
    )
    
    scheduler.run()
