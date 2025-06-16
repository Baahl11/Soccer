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
from dataclasses import asdict
from prediction_response_enricher import PredictionResponseEnricher

# Import existing modules
from enhanced_match_winner import EnhancedPredictionSystem
from platt_calibration import PlattCalibrator
from class_balancing import SoccerSMOTE, BalancedTrainingPipeline
from match_winner import MatchOutcome
from performance_monitoring import PerformanceMonitor, PerformanceMetric
from realtime_monitoring import RealTimeMonitor

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
        self.calibrators = {
            MatchOutcome.HOME_WIN.value: PlattCalibrator(),
            MatchOutcome.DRAW.value: PlattCalibrator(), 
            MatchOutcome.AWAY_WIN.value: PlattCalibrator()
        }
        self.smote_balancer = None
        self.performance_data = []
        
        # Initialize performance monitoring
        if self.config.get('monitoring_enabled', True):
            self.perf_monitor = PerformanceMonitor(
                db_path='soccer_performance.db',
                monitoring_window=1000
            )
            self.rt_monitor = RealTimeMonitor(
                db_path='soccer_realtime.db',
                metrics_window=1000
            )
            # Add retraining callback
            self.perf_monitor.add_retraining_callback(self._retrain_model)
        else:
            self.perf_monitor = None
            self.rt_monitor = None
        
        # Initialize advanced features
        self._initialize_advanced_features()
        
        # Database for monitoring
        self.db_path = "advanced_1x2_monitoring.db"
        self._setup_monitoring_db()
        
        # Initialize response enricher
        self.response_enricher = PredictionResponseEnricher()
        
    def _initialize_advanced_features(self):
        """Initialize advanced prediction features."""
        try:
            # Initialize Platt scaling calibrator
            if self.config.get('use_platt_scaling', True):
                for outcome in [MatchOutcome.HOME_WIN.value, MatchOutcome.DRAW.value, MatchOutcome.AWAY_WIN.value]:
                    self.calibrators[outcome] = PlattCalibrator()
                
                logger.info("✅ Platt scaling calibrators initialized")
            
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
            start_time = datetime.now()
            
            # 1. Get base enhanced prediction
            base_prediction = self.enhanced_system.predict(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                league_id=league_id,
                **context_data or {}
            )
            
            # 2. Apply probability calibration if enabled and calibrator is fitted
            if use_calibration and all(calibrator.is_fitted for calibrator in self.calibrators.values()):
                calibrated_probs = self._apply_calibration(base_prediction)
                base_prediction['probabilities'] = calibrated_probs
                base_prediction['calibrated'] = True
                logger.debug("✅ Applied probability calibration")
            else:
                base_prediction['calibrated'] = False
            
            # 3. Add advanced metrics
            advanced_metrics = self._calculate_advanced_metrics(base_prediction)
            
            # 4. Create comprehensive result
            prediction_id = f"{home_team_id}_{away_team_id}_{league_id}_{int(datetime.now().timestamp())}"
            
            # Create final result
            result = {
                'prediction_id': prediction_id,
                'timestamp': datetime.now().isoformat(),
                'match_info': {
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'league_id': league_id
                },
                'base_prediction': base_prediction,
                'advanced_metrics': advanced_metrics,
                'system_info': {
                    'calibration_enabled': use_calibration and self.calibrators is not None,
                    'smote_balanced': self.config.get('use_smote_balancing', False),
                    'enhanced_system': True,
                    'version': '2.0'
                }
            }
            
            # 5. Monitor performance
            if self.config.get('monitoring_enabled', True):
                response_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Log to real-time monitor
                self.rt_monitor.record_prediction(
                    prediction_id=prediction_id,
                    prediction={
                        'probabilities': base_prediction['probabilities'],
                        'most_likely_outcome': base_prediction.get('predicted_outcome'),
                        'confidence': base_prediction.get('confidence', 0.0)
                    },
                    response_time=response_time
                )
                
                # Log to performance monitor
                confidence_score = base_prediction.get('confidence', 0.0)
                
                self.perf_monitor.log_prediction(
                    match_id=prediction_id,
                    home_team=f"team_{home_team_id}",
                    away_team=f"team_{away_team_id}",
                    predicted_probs=np.array([
                        base_prediction['probabilities'].get(MatchOutcome.HOME_WIN.value, 0),
                        base_prediction['probabilities'].get(MatchOutcome.DRAW.value, 0),
                        base_prediction['probabilities'].get(MatchOutcome.AWAY_WIN.value, 0)
                    ]),
                    model_version='advanced_1x2',
                    confidence_score=confidence_score
                )
                
                # Add monitoring metrics to result
                current_metrics = self.rt_monitor.get_current_metrics()
                result['monitoring'] = {
                    'realtime_metrics': asdict(current_metrics),
                    'response_time_ms': response_time
                }
            
            # 6. Store for monitoring if enabled
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
            
            # Apply calibration separately for each outcome using PlattCalibrator
            calibrated_probs = {}
            for outcome, calibrator in self.calibrators.items():
                if calibrator.is_fitted:
                    # Extract probability for current outcome
                    prob = np.array([probs.get(outcome, 0.0)])
                    # Calibrate probability
                    calibrated = calibrator.calibrate(prob)
                    # Store calibrated probability
                    calibrated_probs[outcome] = float(calibrated[0])
                else:
                    # If calibrator not fitted, use original probability
                    calibrated_probs[outcome] = probs.get(outcome, 0.0)
            
            # Normalize probabilities to sum to 1
            total = sum(calibrated_probs.values())
            if total > 0:
                for outcome in calibrated_probs:
                    calibrated_probs[outcome] /= total
            
            return calibrated_probs
            
        except Exception as e:
            logger.error(f"❌ Error applying calibration: {e}")
            return prediction.get('probabilities', {})
    
    def _calculate_advanced_metrics(self, prediction: Dict[str, Any]):
        """Calculate advanced metrics for the prediction."""
        try:
            # Placeholder for advanced metrics calculation
            metrics = {
                'confidence': prediction.get('confidence', 0.0),
                'probability_range': max(prediction['probabilities'].values()) - min(prediction['probabilities'].values()),
                'average_probability': np.mean(list(prediction['probabilities'].values())),
                'home_advantage': self._calculate_home_advantage(prediction['match_info']['home_team_id']),
                'travel_impact': self._calculate_travel_impact(prediction['match_info']['away_team_id']),
                'weather_impact': self._calculate_weather_impact(prediction['match_info'].get('weather', {}))
            }
            
            # Add more metrics as needed
            
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error calculating advanced metrics: {e}")
            return {}
    
    def _calculate_home_advantage(self, home_team_id: int) -> float:
        """Estimate home advantage for the given team."""
        # Placeholder implementation
        return 0.1  # 10% home advantage
    
    def _calculate_travel_impact(self, away_team_id: int) -> float:
        """Estimate travel impact for the away team."""
        # Placeholder implementation
        return -0.05  # 5% travel disadvantage
    
    def _calculate_weather_impact(self, weather_data: Dict[str, Any]) -> float:
        """Estimate weather impact on the match."""
        # Placeholder implementation
        return 0.0  # No weather impact by default
    
    def _store_prediction(self, prediction: Dict[str, Any]):
        """Store the prediction result in the monitoring database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert prediction data
            cursor.execute('''
                INSERT INTO predictions (
                    home_team_id, away_team_id, league_id,
                    predicted_outcome, home_win_prob, draw_prob, away_win_prob,
                    confidence, actual_outcome, is_correct, calibrated, smote_balanced
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction['match_info']['home_team_id'],
                prediction['match_info']['away_team_id'],
                prediction['match_info']['league_id'],
                prediction['base_prediction'].get('predicted_outcome'),
                prediction['base_prediction'].get('home_win_prob', 0.0),
                prediction['base_prediction'].get('draw_prob', 0.0),
                prediction['base_prediction'].get('away_win_prob', 0.0),
                prediction['base_prediction'].get('confidence', 0.0),
                prediction.get('actual_outcome'),
                prediction.get('is_correct', False),
                prediction['calibrated'],
                self.config.get('use_smote_balancing', False)
            ))
            
            conn.commit()
            conn.close()
            logger.info("✅ Prediction stored in monitoring database")
            
        except Exception as e:
            logger.error(f"❌ Error storing prediction: {e}")
    
    def train_calibrator(self, historical_data: List[Dict[str, Any]]):
        """
        Train the probability calibrator using historical data.
        
        Args:
            historical_data: List of historical predictions with actual outcomes
        """
        try:
            if not self.calibrators:
                logger.error("❌ Calibrators not initialized")
                return
            
            # Prepare training data for each outcome
            outcome_data = {
                MatchOutcome.HOME_WIN.value: {"predictions": [], "targets": []},
                MatchOutcome.DRAW.value: {"predictions": [], "targets": []},
                MatchOutcome.AWAY_WIN.value: {"predictions": [], "targets": []}
            }
            
            for data in historical_data:
                pred_probs = data.get('probabilities', {})
                actual_outcome = data.get('actual_outcome')
                
                if pred_probs and actual_outcome:
                    # Process each outcome
                    for outcome in outcome_data:
                        # Get probability for current outcome
                        prob = pred_probs.get(outcome, 0.0)
                        outcome_data[outcome]["predictions"].append(prob)
                        # Set target as 1 if this was the actual outcome, 0 otherwise
                        outcome_data[outcome]["targets"].append(1 if actual_outcome == outcome else 0)
            
            # Train calibrator for each outcome
            min_samples = 50
            for outcome, data in outcome_data.items():
                if len(data["predictions"]) < min_samples:
                    logger.warning(f"⚠️ Insufficient data for calibration training for {outcome}")
                    continue
                    
                # Convert to numpy arrays
                X = np.array(data["predictions"])
                y = np.array(data["targets"])
                
                # Train calibrator
                calibrator = self.calibrators[outcome]
                calibrator.fit(X, y)
                
                logger.info(f"✅ Calibrator trained for {outcome} on {len(X)} samples")
                
                # Add calibration metrics
                if calibrator.is_fitted:
                    calibrated = calibrator.calibrate(X)
                    metrics = {
                        'mean_prediction': float(np.mean(X)),
                        'mean_calibrated': float(np.mean(calibrated)),
                        'positive_rate': float(np.mean(y))
                    }
                    logger.debug(f"📊 {outcome} calibration metrics: {metrics}")
                
        except Exception as e:
            logger.error(f"❌ Error training calibrator: {e}")
            logger.debug(f"Detailed error: {str(e)}", exc_info=True)
    
    def monitor_performance(self, actual_outcome: str, predicted_probs: Dict[str, float]):
        """
        Monitor and log the performance of the predictions.
        
        Args:
            actual_outcome: Actual outcome of the match
            predicted_probs: Predicted probabilities for each outcome
        """
        try:
            # Extract performance metrics
            accuracy = 1.0 if predicted_probs.get(actual_outcome, 0.0) > 0.5 else 0.0
            brier_score = (1 - predicted_probs.get(actual_outcome, 0.0)) ** 2
            calibration_error = abs(predicted_probs.get(actual_outcome, 0.0) - 0.5)
            
            # Log performance metrics
            logger.info(f"📈 Performance - Accuracy: {accuracy}, Brier Score: {brier_score}, Calibration Error: {calibration_error}")
            
            # Store in database
            self._store_performance_metrics(accuracy, brier_score, calibration_error)
        
        except Exception as e:
            logger.error(f"❌ Error monitoring performance: {e}")
    
    def _store_performance_metrics(self, accuracy: float, brier_score: float, calibration_error: float):
        """Store the performance metrics in the monitoring database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert performance metrics data
            cursor.execute('''
                INSERT INTO performance_metrics (
                    accuracy, brier_score, calibration_error,
                    draw_precision, draw_recall, total_predictions
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                accuracy,
                brier_score,
                calibration_error,
                0.0,  # Draw precision (not calculated)
                0.0,  # Draw recall (not calculated)
                1  # Total predictions (assuming 1 for each call)
            ))
            
            conn.commit()
            conn.close()
            logger.info("✅ Performance metrics stored in monitoring database")
            
        except Exception as e:
            logger.error(f"❌ Error storing performance metrics: {e}")
    
    def analyze_team_composition(self, team_id: int, match_context: Dict[str, Any]):
        """
        Analyze the team composition and its impact on the match outcome.
        
        Args:
            team_id: ID of the team to analyze
            match_context: Context of the match (e.g., opponent, location)
        
        Returns:
            Analysis report on the team composition
        """
        try:
            # Placeholder for team composition analysis
            report = {
                'team_id': team_id,
                'match_context': match_context,
                'key_players': self._identify_key_players(team_id),
                'tactical_advantage': self._assess_tactical_advantage(team_id, match_context)
            }
            
            return report
        
        except Exception as e:
            logger.error(f"❌ Error analyzing team composition: {e}")
            return {}
    
    def _identify_key_players(self, team_id: int) -> List[int]:
        """Identify key players in the team based on historical data."""
        # Placeholder implementation
        return [1, 2, 3]  # Example player IDs
    
    def _assess_tactical_advantage(self, team_id: int, match_context: Dict[str, Any]) -> str:
        """Assess the tactical advantage of the team in the given match context."""
        # Placeholder implementation
        return "Neutral"  # Neutral, Advantage, or Disadvantage
    
    def analyze_weather_impact(self, weather_data: Dict[str, Any]):
        """
        Analyze the impact of weather conditions on the match outcome.
        
        Args:
            weather_data: Weather data for the match
        
        Returns:
            Impact assessment report
        """
        try:
            # Placeholder for weather impact analysis
            report = {
                'temperature_impact': self._assess_temperature_impact(weather_data.get('temperature')),
                'humidity_impact': self._assess_humidity_impact(weather_data.get('humidity')),
                'wind_speed_impact': self._assess_wind_speed_impact(weather_data.get('wind_speed'))
            }
            
            return report
        
        except Exception as e:
            logger.error(f"❌ Error analyzing weather impact: {e}")
            return {}
    
    def _assess_temperature_impact(self, temperature: Optional[float]) -> str:
        """Assess the impact of temperature on the match outcome."""
        # Placeholder implementation
        if temperature is None:
            return "No data"
        elif temperature > 30:
            return "Negative"
        elif temperature < 10:
            return "Positive"
        else:
            return "Neutral"
    
    def _assess_humidity_impact(self, humidity: Optional[float]) -> str:
        """Assess the impact of humidity on the match outcome."""
        # Placeholder implementation
        if humidity is None:
            return "No data"
        elif humidity > 70:
            return "Negative"
        elif humidity < 30:
            return "Positive"
        else:
            return "Neutral"
    
    def _assess_wind_speed_impact(self, wind_speed: Optional[float]) -> str:
        """Assess the impact of wind speed on the match outcome."""
        # Placeholder implementation
        if wind_speed is None:
            return "No data"
        elif wind_speed > 20:
            return "Negative"
        elif wind_speed < 5:
            return "Positive"
        else:
            return "Neutral"
    
    def _retrain_model(self, alert: Any) -> None:
        """Callback for retraining model when performance degrades"""
        try:
            logger.warning(f"Initiating model retraining due to alert: {alert.message}")
            
            # Get latest training data
            historical_data = self._get_historical_data(days=180)  # Last 6 months
            
            # First retrain calibrators
            if self.calibrators:
                self.train_calibrator(historical_data)
            
            # Then retrain SMOTE if enabled
            if self.smote_balancer and self.config.get('use_smote_balancing'):
                X_balanced, y_balanced = self.prepare_balanced_training_data(historical_data)
                
                if len(X_balanced) > 0:
                    logger.info(f"Retraining with {len(X_balanced)} balanced samples")
                else:
                    logger.error("No balanced data available for retraining")
            
            logger.info("✅ Model retraining completed")
            
        except Exception as e:
            logger.error(f"❌ Error during model retraining: {e}")
        
    def _get_historical_data(self, days: int = 180) -> List[Dict[str, Any]]:
        """Get historical prediction data for retraining"""
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            cursor = conn.cursor()
            cursor.execute('''
                SELECT home_team_id, away_team_id, league_id, 
                       predicted_outcome, home_win_prob, draw_prob, away_win_prob, 
                       actual_outcome, confidence 
                FROM predictions
                WHERE timestamp >= ? AND actual_outcome IS NOT NULL
            ''', (cutoff_date,))
            
            results = cursor.fetchall()
            conn.close()
            
            historical_data = []
            for row in results:
                historical_data.append({
                    'home_team_id': row[0],
                    'away_team_id': row[1],
                    'league_id': row[2],
                    'probabilities': {
                        MatchOutcome.HOME_WIN.value: row[4],
                        MatchOutcome.DRAW.value: row[5],
                        MatchOutcome.AWAY_WIN.value: row[6]
                    },
                    'actual_outcome': row[7],
                    'confidence': row[8]
                })
            
            return historical_data
            
        except Exception as e:
            logger.error(f"Error getting historical data: {e}")
            return []
    
    def update_match_result(self, prediction_id: str, actual_outcome: str, match_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Update monitoring metrics with actual match result.
        
        Args:
            prediction_id: ID of the prediction
            actual_outcome: Actual match outcome (e.g., 'home_win', 'draw', 'away_win')
            match_data: Additional match data
        """
        try:
            if not self.config.get('monitoring_enabled', True):
                return
            
            # Update real-time monitor
            self.rt_monitor.update_actual_result(
                prediction_id=prediction_id,
                actual_result={
                    'outcome': actual_outcome,
                    'match_data': match_data
                }
            )
            
            # Get prediction from database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                SELECT home_team_id, away_team_id, league_id,
                       predicted_outcome, home_win_prob, draw_prob, away_win_prob,
                       confidence, calibrated
                FROM predictions 
                WHERE prediction_id = ?
            ''', (prediction_id,))
            
            row = cursor.fetchone()
            
            if row:
                # Update prediction record
                cursor.execute('''
                    UPDATE predictions
                    SET actual_outcome = ?, is_correct = ?
                    WHERE prediction_id = ?
                ''', (
                    actual_outcome,
                    row[3] == actual_outcome,  # predicted_outcome == actual_outcome
                    prediction_id
                ))
                conn.commit()
                
                # Log metrics
                if self.perf_monitor:
                    # Calculate prediction accuracy
                    is_correct = (row[3] == actual_outcome)
                    confidence = row[7]
                    calibrated = row[8]
                    
                    # Log accuracy metric
                    self.perf_monitor.log_metric(PerformanceMetric(
                        timestamp=datetime.now(),
                        metric_name='accuracy',
                        metric_value=float(is_correct),
                        model_version='advanced_1x2',
                        data_source='production',
                        additional_info={
                            'prediction_id': prediction_id,
                            'calibrated': calibrated,
                            'confidence': confidence
                        }
                    ))
                    
                    # Calculate and log Brier score
                    actual_probs = [0.0, 0.0, 0.0]  # [home, draw, away]
                    outcome_idx = {
                        MatchOutcome.HOME_WIN.value: 0,
                        MatchOutcome.DRAW.value: 1,
                        MatchOutcome.AWAY_WIN.value: 2
                    }.get(actual_outcome, 0)
                    actual_probs[outcome_idx] = 1.0
                    
                    pred_probs = [row[4], row[5], row[6]]  # [home_win_prob, draw_prob, away_win_prob]
                    brier_score = sum((pred_probs[i] - actual_probs[i])**2 for i in range(3)) / 3
                    
                    self.perf_monitor.log_metric(PerformanceMetric(
                        timestamp=datetime.now(),
                        metric_name='brier_score',
                        metric_value=float(brier_score),
                        model_version='advanced_1x2',
                        data_source='production',
                        additional_info={
                            'prediction_id': prediction_id,
                            'calibrated': calibrated
                        }
                    ))
                    
                    # Log prediction confidence calibration
                    if confidence is not None:
                        calibration_error = abs(confidence - float(is_correct))
                        self.perf_monitor.log_metric(PerformanceMetric(
                            timestamp=datetime.now(),
                            metric_name='confidence_calibration',
                            metric_value=1.0 - calibration_error,
                            model_version='advanced_1x2',
                            data_source='production',
                            additional_info={
                                'prediction_id': prediction_id,
                                'raw_error': calibration_error
                            }
                        ))
            
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error updating match result: {e}")
            logger.debug(f"Detailed error: {str(e)}", exc_info=True)
    
    def predict_match(
        self, 
        home_team_id: int, 
        away_team_id: int, 
        match_id: int = None,
        weather_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Make a complete match prediction with all enhancements
        
        Args:
            home_team_id: ID of home team
            away_team_id: ID of away team
            match_id: Optional match ID
            weather_data: Optional weather data
            
        Returns:
            Complete prediction with all analysis
        """
        try:
            # Get base prediction from ensemble
            base_prediction = self._get_base_prediction(home_team_id, away_team_id)
            
            # Prepare team data
            team_data = {
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'match_id': match_id,
                'home_team_name': self._get_team_name(home_team_id),
                'away_team_name': self._get_team_name(away_team_id)
            }
            
            # Enrich prediction with all additional analysis
            enriched_prediction = self.response_enricher.enrich_prediction(
                base_prediction, team_data, weather_data
            )
            
            return enriched_prediction
            
        except Exception as e:
            logger.error(f"Error in advanced prediction: {e}")
            return self._get_fallback_prediction()
    
    def predict_match_formatted(
        self, 
        home_team_id: int, 
        away_team_id: int, 
        match_id: int = None,
        weather_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Make a prediction and format it for dashboard presentation
        """
        try:
            # Get enriched prediction
            prediction = self.predict_match(home_team_id, away_team_id, match_id, weather_data)
            
            # Prepare team data
            team_data = {
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_team_name': self._get_team_name(home_team_id),
                'away_team_name': self._get_team_name(away_team_id)
            }
            
            # Format for presentation
            return self.response_enricher.format_for_presentation(prediction, team_data)
            
        except Exception as e:
            logger.error(f"Error in formatted prediction: {e}")
            return {"error": str(e)}
    
    def _get_base_prediction(self, home_team_id: int, away_team_id: int) -> Dict[str, Any]:
        """Get base prediction from ensemble model"""
        # TODO: Implement actual ensemble prediction
        # For now return dummy data
        return {
            'prob_1': 0.45,
            'prob_X': 0.25,
            'prob_2': 0.30,
            'confidence': 0.7
        }
    
    def _get_team_name(self, team_id: int) -> str:
        """Get team name from ID"""
        # TODO: Implement actual team name lookup
        return f"Team {team_id}"
    
    def _get_fallback_prediction(self) -> Dict[str, Any]:
        """Get fallback prediction for error cases"""
        return {
            "prediction": {
                "home_win_probability": 0.45,
                "draw_probability": 0.25,
                "away_win_probability": 0.30,
                "confidence": 0.5,
                "calibrated_probabilities": {
                    "home_win": 0.45,
                    "draw": 0.25,
                    "away_win": 0.30
                }
            },
            "system_info": {
                "version": "2.0.0",
                "timestamp": datetime.now().isoformat(),
                "enhanced_system": False,
                "calibration_enabled": False,
                "contextual_analysis_enabled": False
            }
        }
