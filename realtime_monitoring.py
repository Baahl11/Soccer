"""
Real-time monitoring system for prediction quality and model performance.
"""

import logging
import sqlite3
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime, timedelta
import numpy as np
from dataclasses import dataclass
from threading import Lock
import json
import os

logger = logging.getLogger(__name__)

@dataclass
class MonitoringMetrics:
    """Metrics tracked by the monitoring system"""
    timestamp: str
    prediction_count: int
    avg_confidence: float
    calibration_score: float
    error_rate: float
    alerts_triggered: int
    response_time_ms: float
    
class RealTimeMonitor:
    def __init__(
        self,
        db_path: str = 'data/monitoring.db',
        alert_thresholds: Optional[Dict[str, float]] = None,
        metrics_window: int = 1000
    ):
        """
        Initialize real-time monitoring system.
        
        Args:
            db_path: Path to SQLite database for metrics
            alert_thresholds: Custom alert thresholds
            metrics_window: Number of predictions to keep in rolling window
        """
        self.db_path = db_path
        self.metrics_window = metrics_window
        self.lock = Lock()
        self.recent_predictions: List[Dict[str, Any]] = []
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'min_confidence': 0.4,
            'max_error_rate': 0.15,
            'max_response_time': 1000,  # ms
            'calibration_threshold': 0.25,
            'anomaly_z_score': 3.0
        }
        
        self._setup_database()
        
    def _setup_database(self):
        """Setup SQLite database for metrics storage"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Create metrics table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS metrics (
                        timestamp TEXT PRIMARY KEY,
                        prediction_count INTEGER,
                        avg_confidence REAL,
                        calibration_score REAL,
                        error_rate REAL,
                        alerts_triggered INTEGER,
                        response_time_ms REAL,
                        metrics_json TEXT
                    )
                ''')
                
                # Create predictions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS predictions (
                        id TEXT PRIMARY KEY,
                        timestamp TEXT,
                        prediction_json TEXT,
                        actual_result_json TEXT,
                        performance_metrics_json TEXT
                    )
                ''')
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error setting up monitoring database: {e}")
            
    def record_prediction(
        self,
        prediction_id: str,
        prediction: Dict[str, Any],
        response_time: float
    ) -> None:
        """Record a new prediction for monitoring"""
        try:
            timestamp = datetime.now().isoformat()
            
            with self.lock:
                # Add to recent predictions
                self.recent_predictions.append({
                    'id': prediction_id,
                    'timestamp': timestamp,
                    'prediction': prediction,
                    'response_time': response_time
                })
                
                # Maintain window size
                if len(self.recent_predictions) > self.metrics_window:
                    self.recent_predictions.pop(0)
                
                # Store in database
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        'INSERT INTO predictions (id, timestamp, prediction_json) VALUES (?, ?, ?)',
                        (prediction_id, timestamp, json.dumps(prediction))
                    )
                    conn.commit()
                
            # Check for alerts
            self._check_alerts(prediction, response_time)
            
        except Exception as e:
            logger.error(f"Error recording prediction: {e}")
            
    def update_actual_result(
        self,
        prediction_id: str,
        actual_result: Dict[str, Any]
    ) -> None:
        """Update prediction with actual result"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get original prediction
                cursor.execute(
                    'SELECT prediction_json FROM predictions WHERE id = ?',
                    (prediction_id,)
                )
                row = cursor.fetchone()
                
                if row:
                    prediction = json.loads(row[0])
                    
                    # Calculate performance metrics
                    metrics = self._calculate_performance_metrics(
                        prediction,
                        actual_result
                    )
                    
                    # Update database
                    cursor.execute('''
                        UPDATE predictions 
                        SET actual_result_json = ?, performance_metrics_json = ?
                        WHERE id = ?
                    ''', (
                        json.dumps(actual_result),
                        json.dumps(metrics),
                        prediction_id
                    ))
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error updating actual result: {e}")
            
    def get_current_metrics(self) -> MonitoringMetrics:
        """Get current monitoring metrics"""
        try:
            with self.lock:
                if not self.recent_predictions:
                    return MonitoringMetrics(
                        timestamp=datetime.now().isoformat(),
                        prediction_count=0,
                        avg_confidence=0.0,
                        calibration_score=0.0,
                        error_rate=0.0,
                        alerts_triggered=0,
                        response_time_ms=0.0
                    )
                    
                # Calculate metrics
                confidences = [
                    p['prediction'].get('confidence', 0)
                    for p in self.recent_predictions
                ]
                
                response_times = [
                    p['response_time']
                    for p in self.recent_predictions
                ]
                
                return MonitoringMetrics(
                    timestamp=datetime.now().isoformat(),
                    prediction_count=len(self.recent_predictions),
                    avg_confidence=float(np.mean(confidences)),
                    calibration_score=self._calculate_calibration_score(),
                    error_rate=self._calculate_error_rate(),
                    alerts_triggered=self._count_recent_alerts(),
                    response_time_ms=float(np.mean(response_times))
                )
                
        except Exception as e:
            logger.error(f"Error getting current metrics: {e}")
            return MonitoringMetrics(
                timestamp=datetime.now().isoformat(),
                prediction_count=0,
                avg_confidence=0.0,
                calibration_score=0.0,
                error_rate=0.0,
                alerts_triggered=0,
                response_time_ms=0.0
            )
            
    def _check_alerts(self, prediction: Dict[str, Any], response_time: float) -> None:
        """Check for alert conditions"""
        try:
            alerts = []
            
            # Check confidence
            if prediction.get('confidence', 1.0) < self.alert_thresholds['min_confidence']:
                alerts.append('Low confidence prediction')
                
            # Check response time
            if response_time > self.alert_thresholds['max_response_time']:
                alerts.append('Slow response time')
                
            # Check for anomalies
            if self._is_prediction_anomaly(prediction):
                alerts.append('Prediction anomaly detected')
                
            # Log alerts
            if alerts:
                logger.warning(
                    f"⚠️ Alerts for prediction: {', '.join(alerts)}"
                )
                
        except Exception as e:
            logger.error(f"Error checking alerts: {e}")
            
    def _is_prediction_anomaly(self, prediction: Dict[str, Any]) -> bool:
        """Detect anomalous predictions using statistical analysis"""
        try:
            if len(self.recent_predictions) < 30:  # Need enough data
                return False
                
            # Get relevant values for comparison
            if 'goals' in prediction:
                values = [
                    p['prediction'].get('goals', {}).get('total_xg', 0)
                    for p in self.recent_predictions[-30:]
                ]
                current = prediction.get('goals', {}).get('total_xg', 0)
            else:
                values = [
                    p['prediction'].get('confidence', 0)
                    for p in self.recent_predictions[-30:]
                ]
                current = prediction.get('confidence', 0)
                
            # Calculate z-score
            mean = np.mean(values)
            std = np.std(values) or 1.0  # Avoid division by zero
            z_score = abs((current - mean) / std)
            
            return z_score > self.alert_thresholds['anomaly_z_score']
            
        except Exception as e:
            logger.error(f"Error checking for anomalies: {e}")
            return False
            
    def _calculate_calibration_score(self) -> float:
        """Calculate calibration score using recent predictions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent predictions with results
                cursor.execute('''
                    SELECT prediction_json, actual_result_json
                    FROM predictions
                    WHERE actual_result_json IS NOT NULL
                    ORDER BY timestamp DESC
                    LIMIT 1000
                ''')
                
                rows = cursor.fetchall()
                
                if not rows:
                    return 0.0
                    
                # Calculate Brier score
                squared_errors = []
                
                for pred_json, actual_json in rows:
                    pred = json.loads(pred_json)
                    actual = json.loads(actual_json)
                    
                    if 'probabilities' in pred:
                        probs = pred['probabilities']
                        actual_outcome = actual.get('outcome')
                        
                        if actual_outcome in probs:
                            error = probs[actual_outcome] - 1.0
                            squared_errors.append(error * error)
                            
                if squared_errors:
                    return float(1.0 - np.mean(squared_errors))
                    
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating calibration score: {e}")
            return 0.0
            
    def _calculate_error_rate(self) -> float:
        """Calculate recent prediction error rate"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent predictions with results
                cursor.execute('''
                    SELECT prediction_json, actual_result_json
                    FROM predictions
                    WHERE actual_result_json IS NOT NULL
                    ORDER BY timestamp DESC
                    LIMIT 1000
                ''')
                
                rows = cursor.fetchall()
                
                if not rows:
                    return 0.0
                    
                # Count errors
                errors = 0
                total = len(rows)
                
                for pred_json, actual_json in rows:
                    pred = json.loads(pred_json)
                    actual = json.loads(actual_json)
                    
                    predicted = pred.get('most_likely_outcome')
                    actual_outcome = actual.get('outcome')
                    
                    if predicted != actual_outcome:
                        errors += 1
                        
                return float(errors) / total if total > 0 else 0.0
                
        except Exception as e:
            logger.error(f"Error calculating error rate: {e}")
            return 0.0
            
    def _calculate_performance_metrics(
        self,
        prediction: Dict[str, Any],
        actual: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate detailed performance metrics for a prediction"""
        try:
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'accuracy': 0,
                'confidence_calibration': 0.0,
                'probability_error': 0.0
            }
            
            # Accuracy (1 if correct, 0 if wrong)
            pred_outcome = prediction.get('most_likely_outcome')
            actual_outcome = actual.get('outcome')
            metrics['accuracy'] = 1 if pred_outcome == actual_outcome else 0
            
            # Confidence calibration
            confidence = prediction.get('confidence', 0)
            metrics['confidence_calibration'] = abs(confidence - metrics['accuracy'])
            
            # Probability error
            if 'probabilities' in prediction and actual_outcome:
                predicted_prob = prediction['probabilities'].get(actual_outcome, 0)
                metrics['probability_error'] = abs(1.0 - predicted_prob)
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
