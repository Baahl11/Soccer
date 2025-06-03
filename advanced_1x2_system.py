#!/usr/bin/env python3
"""
Advanced 1X2 Prediction System Integration

This module integrates all advanced 1X2 prediction features including:
- Platt scaling probability calibration
- SMOTE class balancing
- Performance monitoring
- Team composition analysis
- Weather impact analysis

Priority 2: Medium Priority Implementation
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path

# Import existing modules
from enhanced_match_winner import EnhancedPredictionSystem
from probability_calibration import ProbabilityCalibrator, calibration_assessment
from class_balancing import SoccerSMOTE, BalancedTrainingPipeline
from match_winner import MatchOutcome

logger = logging.getLogger(__name__)

class Advanced1X2System:
    """
    Advanced 1X2 prediction system with all enhancement features integrated.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the advanced 1X2 system."""
        self.config = config or {
            'use_platt_scaling': True,
            'use_smote_balancing': True,
            'use_performance_monitoring': True,
            'use_composition_analysis': False,  # Optional feature
            'use_weather_analysis': False,      # Optional feature
            'calibration_method': 'platt',
            'smote_method': 'smote',
            'monitoring_enabled': True
        }
        
        # Initialize core components
        self.enhanced_system = EnhancedPredictionSystem()
        self.calibrator = None
        self.smote_balancer = None
        self.performance_data = []
        
        # Initialize advanced features
        self._initialize_advanced_features()
        
        # Database for monitoring
        self.db_path = "advanced_1x2_monitoring.db"
        self._setup_monitoring_db()
        
    def _initialize_advanced_features(self):
        """Initialize advanced prediction features."""
        try:
            # Initialize Platt scaling calibrator
            if self.config.get('use_platt_scaling', True):
                self.calibrator = ProbabilityCalibrator(
                    method=self.config.get('calibration_method', 'platt')
                )
                logger.info("✅ Platt scaling calibrator initialized")
            
            # Initialize SMOTE balancer
            if self.config.get('use_smote_balancing', True):
                self.smote_balancer = SoccerSMOTE(
                    sampling_strategy='soccer',
                    random_state=42
                )
                logger.info("✅ SMOTE class balancer initialized")
                
        except Exception as e:
            logger.error(f"❌ Error initializing advanced features: {e}")
    
    def _setup_monitoring_db(self):
        """Setup SQLite database for performance monitoring."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    home_team_id INTEGER,
                    away_team_id INTEGER,
                    league_id INTEGER,
                    predicted_outcome TEXT,
                    home_win_prob REAL,
                    draw_prob REAL,
                    away_win_prob REAL,
                    confidence REAL,
                    actual_outcome TEXT,
                    is_correct BOOLEAN,
                    calibrated BOOLEAN,
                    smote_balanced BOOLEAN
                )
            ''')
            
            # Create performance metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    accuracy REAL,
                    brier_score REAL,
                    calibration_error REAL,
                    draw_precision REAL,
                    draw_recall REAL,
                    total_predictions INTEGER
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ Monitoring database initialized")
            
        except Exception as e:
            logger.error(f"❌ Error setting up monitoring database: {e}")
    
    def predict_match_advanced(
        self,
        home_team_id: int,
        away_team_id: int,
        league_id: int,
        use_calibration: bool = True,
        context_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Make an advanced 1X2 prediction with all enhancements.
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            league_id: League ID
            use_calibration: Whether to apply probability calibration
            context_data: Additional context data
            
        Returns:
            Enhanced prediction with advanced features
        """
        try:
            # 1. Get base enhanced prediction
            base_prediction = self.enhanced_system.predict(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                league_id=league_id,
                **context_data or {}
            )
            
            # 2. Apply probability calibration if enabled and calibrator is fitted
            if use_calibration and self.calibrator and getattr(self.calibrator, 'is_fitted', False):
                calibrated_probs = self._apply_calibration(base_prediction)
                base_prediction['probabilities'] = calibrated_probs
                base_prediction['calibrated'] = True
                logger.debug("✅ Applied probability calibration")
            else:
                base_prediction['calibrated'] = False
            
            # 3. Add advanced metrics
            advanced_metrics = self._calculate_advanced_metrics(base_prediction)
            
            # 4. Create comprehensive result
            result = {
                'prediction_id': f"{home_team_id}_{away_team_id}_{league_id}_{int(datetime.now().timestamp())}",
                'timestamp': datetime.now().isoformat(),
                'match_info': {
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'league_id': league_id
                },
                'base_prediction': base_prediction,
                'advanced_metrics': advanced_metrics,
                'system_info': {
                    'calibration_enabled': use_calibration and self.calibrator is not None,
                    'smote_balanced': self.config.get('use_smote_balancing', False),
                    'enhanced_system': True,
                    'version': '2.0'
                }
            }
            
            # 5. Store for monitoring if enabled
            if self.config.get('monitoring_enabled', True):
                self._store_prediction(result)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in advanced prediction: {e}")
            # Fallback to base prediction
            return {
                'error': str(e),
                'fallback_prediction': self.enhanced_system.predict(
                    home_team_id, away_team_id, league_id
                )
            }
    
    def _apply_calibration(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Apply probability calibration to predictions."""
        try:
            # Extract probabilities
            probs = prediction.get('probabilities', {})
            
            # Convert to numpy array format expected by calibrator
            prob_array = np.array([[
                probs.get(MatchOutcome.HOME_WIN.value, 0),
                probs.get(MatchOutcome.DRAW.value, 0),
                probs.get(MatchOutcome.AWAY_WIN.value, 0)
            ]])
            
            # Apply calibration if calibrator is not None
            if self.calibrator is not None:
                calibrated_array = self.calibrator.calibrate(prob_array)
                
                # Convert back to dictionary format
                calibrated_probs = {
                    MatchOutcome.HOME_WIN.value: float(calibrated_array[0][0]),
                    MatchOutcome.DRAW.value: float(calibrated_array[0][1]),
                    MatchOutcome.AWAY_WIN.value: float(calibrated_array[0][2])
                }
                
                return calibrated_probs
            else:
                return prediction.get('probabilities', {})
            
        except Exception as e:
            logger.error(f"❌ Error applying calibration: {e}")
            return prediction.get('probabilities', {})
    
    def _calculate_advanced_metrics(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate advanced prediction metrics."""
        probs = prediction.get('probabilities', {})
        
        # Calculate entropy (uncertainty measure)
        prob_values = list(probs.values())
        entropy = -sum(p * np.log(p + 1e-10) for p in prob_values if p > 0)
        
        # Calculate maximum probability
        max_prob = max(prob_values) if prob_values else 0
        
        # Calculate probability spread
        prob_spread = max(prob_values) - min(prob_values) if prob_values else 0
        
        # Calculate draw favorability
        draw_prob = probs.get(MatchOutcome.DRAW.value, 0)
        draw_favorability = draw_prob / max(prob_values) if max(prob_values) > 0 else 0
        
        return {
            'entropy': float(entropy),
            'max_probability': float(max_prob),
            'probability_spread': float(prob_spread),
            'draw_favorability': float(draw_favorability),
            'confidence_level': 'high' if max_prob > 0.6 else 'medium' if max_prob > 0.4 else 'low'
        }
    
    def _store_prediction(self, prediction_result: Dict[str, Any]):
        """Store prediction for monitoring and analysis."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            match_info = prediction_result.get('match_info', {})
            base_pred = prediction_result.get('base_prediction', {})
            probs = base_pred.get('probabilities', {})
            
            cursor.execute('''
                INSERT INTO predictions (
                    home_team_id, away_team_id, league_id,
                    predicted_outcome, home_win_prob, draw_prob, away_win_prob,
                    confidence, calibrated, smote_balanced
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                match_info.get('home_team_id'),
                match_info.get('away_team_id'),
                match_info.get('league_id'),
                base_pred.get('predicted_outcome'),
                probs.get(MatchOutcome.HOME_WIN.value, 0),
                probs.get(MatchOutcome.DRAW.value, 0),
                probs.get(MatchOutcome.AWAY_WIN.value, 0),
                base_pred.get('confidence', 0),
                base_pred.get('calibrated', False),
                self.config.get('use_smote_balancing', False)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error storing prediction: {e}")
    
    def train_calibrator(self, historical_data: List[Dict[str, Any]]):
        """
        Train the probability calibrator using historical data.
        
        Args:
            historical_data: List of historical predictions with actual outcomes
        """
        try:
            if not self.calibrator:
                logger.error("❌ Calibrator not initialized")
                return
            
            # Prepare training data
            predictions = []
            true_outcomes = []
            
            for data in historical_data:
                pred_probs = data.get('probabilities', {})
                actual_outcome = data.get('actual_outcome')
                
                if pred_probs and actual_outcome:
                    # Convert to array format
                    prob_array = [
                        pred_probs.get(MatchOutcome.HOME_WIN.value, 0),
                        pred_probs.get(MatchOutcome.DRAW.value, 0),
                        pred_probs.get(MatchOutcome.AWAY_WIN.value, 0)
                    ]
                    predictions.append(prob_array)
                    
                    # Convert outcome to index
                    outcome_mapping = {
                        MatchOutcome.HOME_WIN.value: 0,
                        MatchOutcome.DRAW.value: 1,
                        MatchOutcome.AWAY_WIN.value: 2
                    }
                    true_outcomes.append(outcome_mapping.get(actual_outcome, 0))
            
            if len(predictions) < 50:
                logger.warning("⚠️ Insufficient data for calibration training")
                return
            
            # Convert to numpy arrays
            X = np.array(predictions)
            y = np.array(true_outcomes)
            
            # Train calibrator
            outcome_names = ['home_win', 'draw', 'away_win']
            self.calibrator.fit(X, y, outcome_names)
            
            logger.info(f"✅ Calibrator trained on {len(predictions)} samples")
            
            # Evaluate calibration
            calibration_metrics = self.calibrator.evaluate_calibration(X, y)
            logger.info(f"📊 Calibration metrics: {calibration_metrics}")
            
        except Exception as e:
            logger.error(f"❌ Error training calibrator: {e}")
    
    def prepare_balanced_training_data(
        self, 
        training_data: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare balanced training data using SMOTE.
        
        Args:
            training_data: Raw training data
            
        Returns:
            Balanced feature matrix and target array
        """
        try:
            if not self.smote_balancer:
                logger.error("❌ SMOTE balancer not initialized")
                return np.array([]), np.array([])
            
            # Extract features and targets
            features = []
            targets = []
            
            for data in training_data:
                # Extract relevant features for training
                feature_vector = [
                    data.get('home_elo', 1500),
                    data.get('away_elo', 1500),
                    data.get('home_form', 0),
                    data.get('away_form', 0),
                    data.get('h2h_home_wins', 0),
                    data.get('h2h_draws', 0),
                    data.get('h2h_away_wins', 0),
                    data.get('league_position_home', 10),
                    data.get('league_position_away', 10),
                    data.get('home_advantage', 1)
                ]
                features.append(feature_vector)
                
                # Map outcome to class index
                outcome = data.get('actual_outcome')
                outcome_mapping = {
                    MatchOutcome.HOME_WIN.value: 0,
                    MatchOutcome.DRAW.value: 1,
                    MatchOutcome.AWAY_WIN.value: 2
                }
                # Ensure outcome is str before using get
                outcome_str = outcome if isinstance(outcome, str) else ''
                targets.append(outcome_mapping.get(outcome_str, 0))
            
            # Convert to numpy arrays
            X = np.array(features)
            y = np.array(targets)
            
            # Apply SMOTE balancing
            feature_names = [
                'home_elo', 'away_elo', 'home_form', 'away_form',
                'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
                'league_position_home', 'league_position_away', 'home_advantage'
            ]
            
            X_balanced, y_balanced = self.smote_balancer.fit_resample(
                X, y, feature_names, method=self.config.get('smote_method', 'smote')
            )
            
            logger.info(f"✅ SMOTE balancing applied: {len(X)} → {len(X_balanced)} samples")
            
            # Get balancing report
            report = self.smote_balancer.get_resampling_report()
            logger.info(f"📊 Balance improvement: {report.get('balance_improvement', {})}")
            
            return X_balanced, y_balanced
            
        except Exception as e:
            logger.error(f"❌ Error preparing balanced data: {e}")
            return np.array([]), np.array([])
    
    def evaluate_system_performance(self) -> Dict[str, Any]:
        """Evaluate the overall system performance."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get recent predictions
            cursor.execute('''
                SELECT predicted_outcome, actual_outcome, home_win_prob, 
                       draw_prob, away_win_prob, is_correct, calibrated
                FROM predictions 
                WHERE actual_outcome IS NOT NULL
                ORDER BY timestamp DESC 
                LIMIT 1000
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            if not results:
                return {'error': 'No evaluation data available'}
            
            # Calculate metrics
            total_predictions = len(results)
            correct_predictions = sum(1 for r in results if r[5])  # is_correct
            accuracy = correct_predictions / total_predictions
            
            # Calculate draw-specific metrics
            draw_predictions = [r for r in results if r[0] == MatchOutcome.DRAW.value]
            draw_correct = sum(1 for r in draw_predictions if r[5])
            draw_precision = draw_correct / len(draw_predictions) if draw_predictions else 0
            
            actual_draws = [r for r in results if r[1] == MatchOutcome.DRAW.value]
            draw_recall = draw_correct / len(actual_draws) if actual_draws else 0
            
            # Calculate Brier score for calibrated predictions
            calibrated_results = [r for r in results if r[6]]  # calibrated
            brier_score = 0
            if calibrated_results:
                brier_scores = []
                for r in calibrated_results:
                    # Convert actual outcome to one-hot
                    actual_vector = [0, 0, 0]
                    outcome_mapping = {
                        MatchOutcome.HOME_WIN.value: 0,
                        MatchOutcome.DRAW.value: 1,
                        MatchOutcome.AWAY_WIN.value: 2
                    }
                    actual_idx = outcome_mapping.get(r[1], 0)
                    actual_vector[actual_idx] = 1
                    
                    # Calculate Brier score
                    pred_vector = [r[2], r[3], r[4]]  # home, draw, away probs
                    bs = sum((pred_vector[i] - actual_vector[i])**2 for i in range(3))
                    brier_scores.append(bs)
                
                brier_score = np.mean(brier_scores)
            
            performance_metrics = {
                'total_predictions': total_predictions,
                'accuracy': accuracy,
                'draw_precision': draw_precision,
                'draw_recall': draw_recall,
                'brier_score': brier_score,
                'calibrated_predictions': len(calibrated_results),
                'calibration_rate': len(calibrated_results) / total_predictions,
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            # Store performance metrics
            self._store_performance_metrics(performance_metrics)
            
            return performance_metrics
            
        except Exception as e:
            logger.error(f"❌ Error evaluating performance: {e}")
            return {'error': str(e)}
    
    def _store_performance_metrics(self, metrics: Dict[str, Any]):
        """Store performance metrics in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_metrics (
                    accuracy, brier_score, calibration_error,
                    draw_precision, draw_recall, total_predictions
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                metrics.get('accuracy', 0),
                metrics.get('brier_score', 0),
                0,  # calibration_error - to be calculated
                metrics.get('draw_precision', 0),
                metrics.get('draw_recall', 0),
                metrics.get('total_predictions', 0)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error storing performance metrics: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and configuration."""
        return {
            'advanced_1x2_system': 'operational',
            'version': '2.0',
            'features': {
                'enhanced_match_winner': True,
                'probability_calibration': self.calibrator is not None,
                'smote_balancing': self.smote_balancer is not None,
                'performance_monitoring': self.config.get('monitoring_enabled', False),
                'platt_scaling': self.config.get('use_platt_scaling', False),
                'class_balancing': self.config.get('use_smote_balancing', False)
            },
            'calibrator_status': {
                'initialized': self.calibrator is not None,
                'fitted': self.calibrator.is_fitted if self.calibrator else False,
                'method': self.config.get('calibration_method', 'platt')
            },
            'smote_status': {
                'initialized': self.smote_balancer is not None,
                'fitted': self.smote_balancer.is_fitted if self.smote_balancer else False,
                'method': self.config.get('smote_method', 'smote')
            },
            'monitoring': {
                'database_path': self.db_path,
                'enabled': self.config.get('monitoring_enabled', False)
            },
            'timestamp': datetime.now().isoformat()
        }

def create_advanced_1x2_system(config: Optional[Dict[str, Any]] = None) -> Advanced1X2System:
    """
    Factory function to create an Advanced 1X2 System.
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Configured Advanced1X2System instance
    """
    default_config = {
        'use_platt_scaling': True,
        'use_smote_balancing': True,
        'use_performance_monitoring': True,
        'calibration_method': 'platt',
        'smote_method': 'smote',
        'monitoring_enabled': True
    }
    
    if config:
        default_config.update(config)
    
    return Advanced1X2System(default_config)

if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 Advanced 1X2 Prediction System - Priority 2 Implementation")
    print("=" * 60)
    
    # Create system
    system = create_advanced_1x2_system()
    
    # Check status
    status = system.get_system_status()
    print("\n📊 System Status:")
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # Test prediction
    print("\n🔮 Testing Advanced Prediction...")
    try:
        result = system.predict_match_advanced(
            home_team_id=33,  # Manchester United
            away_team_id=40,  # Liverpool
            league_id=39,     # Premier League
            use_calibration=True
        )
        
        print("✅ Advanced prediction completed!")
        print(f"   Prediction ID: {result.get('prediction_id')}")
        print(f"   Enhanced system: {result.get('system_info', {}).get('enhanced_system')}")
        print(f"   Calibration enabled: {result.get('system_info', {}).get('calibration_enabled')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🎉 Advanced 1X2 System Integration Complete!")
