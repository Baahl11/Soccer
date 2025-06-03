"""
Real-time Performance Monitoring System

This module implements a comprehensive monitoring system for tracking
model performance, detecting anomalies, and triggering retraining
when performance degrades.

Key features:
1. Real-time performance tracking
2. Anomaly detection in predictions
3. Automated retraining triggers
4. Performance dashboards
5. Alert system for performance issues
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
import logging
from datetime import datetime, timedelta
import json
import sqlite3
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Data class for storing performance metrics."""
    timestamp: datetime
    metric_name: str
    metric_value: float
    model_version: str
    data_source: str
    additional_info: Optional[Dict[str, Any]] = None

@dataclass
class Alert:
    """Data class for performance alerts."""
    timestamp: datetime
    alert_type: str
    severity: str  # 'low', 'medium', 'high', 'critical'
    message: str
    metric_name: str
    current_value: float
    threshold: float
    suggested_action: str

class PerformanceMonitor:
    """    Real-time performance monitoring system for soccer prediction models.
    """
    
    def __init__(self, db_path: str = "performance_monitor.db",
                 alert_thresholds: Optional[Dict[str, Dict]] = None,
                 monitoring_window: int = 100):
        """
        Initialize the performance monitor.
        
        Args:
            db_path: Path to SQLite database for storing metrics
            alert_thresholds: Dictionary defining alert thresholds for metrics
            monitoring_window: Number of recent predictions to consider for monitoring
        """
        self.db_path = db_path
        self.monitoring_window = monitoring_window
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Default alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'accuracy': {'low': 0.35, 'medium': 0.30, 'high': 0.25, 'critical': 0.20},
            'brier_score': {'low': 0.30, 'medium': 0.35, 'high': 0.40, 'critical': 0.50},
            'log_loss': {'low': 1.2, 'medium': 1.5, 'high': 2.0, 'critical': 2.5},
            'roi': {'low': -0.05, 'medium': -0.10, 'high': -0.20, 'critical': -0.30}
        }
        
        # Recent data storage
        self.recent_predictions = deque(maxlen=monitoring_window)
        self.recent_metrics = deque(maxlen=monitoring_window)
        self.recent_alerts = deque(maxlen=50)
        
        # Callbacks for alerts and retraining
        self.alert_callbacks: List[Callable] = []
        self.retraining_callbacks: List[Callable] = []
        
        # Initialize database
        self._init_database()
        
    def _init_database(self) -> None:
        """Initialize SQLite database for storing metrics and alerts."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    data_source TEXT NOT NULL,
                    additional_info TEXT
                )
            ''')
            
            # Alerts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    message TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    current_value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    suggested_action TEXT NOT NULL
                )
            ''')
            
            # Predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    match_id TEXT,
                    home_team TEXT,
                    away_team TEXT,
                    predicted_probs TEXT NOT NULL,
                    actual_outcome INTEGER,
                    model_version TEXT NOT NULL,
                    confidence_score REAL                )
            ''')
            conn.commit()
    
    def log_prediction(self, match_id: str, home_team: str, away_team: str,
                      predicted_probs: np.ndarray, model_version: str,
                      confidence_score: Optional[float] = None, actual_outcome: Optional[int] = None) -> None:
        """
        Log a prediction for monitoring.
        
        Args:
            match_id: Unique identifier for the match
            home_team: Home team name
            away_team: Away team name
            predicted_probs: Array of predicted probabilities
            model_version: Version of the model making the prediction
            confidence_score: Optional confidence score
            actual_outcome: Actual match outcome (if known)
        """
        timestamp = datetime.now()
        
        # Store in recent predictions
        prediction_data = {
            'timestamp': timestamp,
            'match_id': match_id,
            'home_team': home_team,
            'away_team': away_team,
            'predicted_probs': predicted_probs.tolist(),
            'actual_outcome': actual_outcome,
            'model_version': model_version,
            'confidence_score': confidence_score
        }
        
        self.recent_predictions.append(prediction_data)
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions 
                (timestamp, match_id, home_team, away_team, predicted_probs, 
                 actual_outcome, model_version, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp.isoformat(), match_id, home_team, away_team,
                json.dumps(predicted_probs.tolist()), actual_outcome,
                model_version, confidence_score
            ))
            conn.commit()
    
    def log_metric(self, metric: PerformanceMetric) -> None:
        """
        Log a performance metric.
        
        Args:
            metric: PerformanceMetric object to log
        """
        self.recent_metrics.append(metric)
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO metrics 
                (timestamp, metric_name, metric_value, model_version, data_source, additional_info)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                metric.timestamp.isoformat(), metric.metric_name, metric.metric_value,
                metric.model_version, metric.data_source,
                json.dumps(metric.additional_info) if metric.additional_info else None
            ))
            conn.commit()
        
        # Check for alerts
        self._check_alert_conditions(metric)
    
    def _check_alert_conditions(self, metric: PerformanceMetric) -> None:
        """
        Check if a metric triggers any alert conditions.
        
        Args:
            metric: Performance metric to check
        """
        metric_name = metric.metric_name.lower()
        metric_value = metric.metric_value
        
        if metric_name not in self.alert_thresholds:
            return
        
        thresholds = self.alert_thresholds[metric_name]
        
        # Determine severity level
        severity = None
        threshold_value = None
        
        # For metrics where lower is worse (accuracy, roi)
        if metric_name in ['accuracy', 'roi']:
            if metric_value <= thresholds['critical']:
                severity = 'critical'
                threshold_value = thresholds['critical']
            elif metric_value <= thresholds['high']:
                severity = 'high'
                threshold_value = thresholds['high']
            elif metric_value <= thresholds['medium']:
                severity = 'medium'
                threshold_value = thresholds['medium']
            elif metric_value <= thresholds['low']:
                severity = 'low'
                threshold_value = thresholds['low']
          # For metrics where higher is worse (brier_score, log_loss)
        elif metric_name in ['brier_score', 'log_loss']:
            if metric_value >= thresholds['critical']:
                severity = 'critical'
                threshold_value = thresholds['critical']
            elif metric_value >= thresholds['high']:
                severity = 'high'
                threshold_value = thresholds['high']
            elif metric_value >= thresholds['medium']:
                severity = 'medium'
                threshold_value = thresholds['medium']
            elif metric_value >= thresholds['low']:
                severity = 'low'
                threshold_value = thresholds['low']
        
        if severity and threshold_value is not None:
            self._create_alert(metric, severity, threshold_value)
    
    def _create_alert(self, metric: PerformanceMetric, severity: str, threshold: float) -> None:
        """
        Create and process an alert.
        
        Args:
            metric: Performance metric that triggered the alert
            severity: Alert severity level
            threshold: Threshold value that was crossed
        """
        # Generate suggested action
        suggested_actions = {
            'low': 'Monitor closely for continued degradation',
            'medium': 'Review recent predictions and data quality',
            'high': 'Consider model retraining or parameter adjustment',
            'critical': 'Immediate model retraining required'
        }
        
        alert = Alert(
            timestamp=datetime.now(),
            alert_type='performance_degradation',
            severity=severity,
            message=f"{metric.metric_name} {metric.metric_value:.4f} crossed {severity} threshold {threshold:.4f}",
            metric_name=metric.metric_name,
            current_value=metric.metric_value,
            threshold=threshold,            suggested_action=suggested_actions[severity]
        )
        
        self.recent_alerts.append(alert)
        
        # Store in database
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO alerts 
                (timestamp, alert_type, severity, message, metric_name, 
                 current_value, threshold, suggested_action)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.timestamp.isoformat(), alert.alert_type, alert.severity,
                alert.message, alert.metric_name, alert.current_value,
                alert.threshold, alert.suggested_action
            ))
            conn.commit()
        
        # Execute alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error executing alert callback: {e}")
        
        # Trigger retraining for critical alerts
        if severity == 'critical':
            self._trigger_retraining(alert)
    
    def _trigger_retraining(self, alert: Alert) -> None:
        """
        Trigger model retraining process.
        
        Args:
            alert: Alert that triggered the retraining
        """
        logger.warning(f"Triggering retraining due to critical alert: {alert.message}")
        
        for callback in self.retraining_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error executing retraining callback: {e}")
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get performance summary for the last N hours.
        
        Args:
            hours: Number of hours to include in summary
            
        Returns:
            summary: Performance summary dictionary
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            # Get recent metrics
            metrics_df = pd.read_sql_query('''
                SELECT * FROM metrics 
                WHERE datetime(timestamp) >= datetime(?)
                ORDER BY timestamp DESC
            ''', conn, params=(cutoff_time.isoformat(),))
            
            # Get recent alerts
            alerts_df = pd.read_sql_query('''
                SELECT * FROM alerts 
                WHERE datetime(timestamp) >= datetime(?)
                ORDER BY timestamp DESC
            ''', conn, params=(cutoff_time.isoformat(),))
            
            # Get recent predictions
            predictions_df = pd.read_sql_query('''
                SELECT * FROM predictions 
                WHERE datetime(timestamp) >= datetime(?)
                ORDER BY timestamp DESC
            ''', conn, params=(cutoff_time.isoformat(),))
        
        # Calculate summary statistics
        summary = {
            'time_period': f"Last {hours} hours",
            'total_predictions': len(predictions_df),
            'total_alerts': len(alerts_df),
            'alert_breakdown': alerts_df['severity'].value_counts().to_dict() if not alerts_df.empty else {},
            'recent_metrics': {},
            'model_performance': {}
        }
        
        # Summarize metrics by type
        if not metrics_df.empty:
            for metric_name in metrics_df['metric_name'].unique():
                metric_data = metrics_df[metrics_df['metric_name'] == metric_name]
                summary['recent_metrics'][metric_name] = {
                    'count': len(metric_data),
                    'latest_value': float(metric_data.iloc[0]['metric_value']),
                    'average': float(metric_data['metric_value'].mean()),
                    'trend': self._calculate_trend(metric_data['metric_value'].values)
                }
        
        return summary
    
    def _calculate_trend(self, values: np.ndarray) -> str:
        """
        Calculate trend direction for a series of values.
        
        Args:
            values: Array of metric values (most recent first)
            
        Returns:
            trend: 'improving', 'declining', or 'stable'
        """
        if len(values) < 3:
            return 'insufficient_data'
        
        # Simple linear trend calculation
        x = np.arange(len(values))
        slope = np.polyfit(x, values[::-1], 1)[0]  # Reverse to get chronological order
        
        threshold = 0.01 * np.std(values)  # 1% of standard deviation
        
        if slope > threshold:
            return 'improving'
        elif slope < -threshold:
            return 'declining'
        else:
            return 'stable'
    
    def add_alert_callback(self, callback: Callable) -> None:
        """Add a callback function to be called when alerts are triggered."""
        self.alert_callbacks.append(callback)
    
    def add_retraining_callback(self, callback: Callable) -> None:
        """Add a callback function to be called when retraining is triggered."""
        self.retraining_callbacks.append(callback)
    
    def start_monitoring(self, check_interval: int = 300) -> None:
        """
        Start continuous monitoring in a separate thread.
        
        Args:
            check_interval: Interval between checks in seconds
        """
        if self.is_monitoring:
            logger.warning("Monitoring is already running")
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(check_interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started performance monitoring with {check_interval}s intervals")
    
    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Stopped performance monitoring")
    
    def _monitoring_loop(self, check_interval: int) -> None:
        """Main monitoring loop that runs in a separate thread."""
        while self.is_monitoring:
            try:
                # Perform periodic checks
                self._periodic_health_check()
                time.sleep(check_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(check_interval)
    
    def _periodic_health_check(self) -> None:
        """Perform periodic health checks on the system."""
        # Check for data freshness
        if self.recent_predictions:
            last_prediction_time = self.recent_predictions[-1]['timestamp']
            time_since_last = datetime.now() - last_prediction_time
            
            if time_since_last > timedelta(hours=6):
                alert = Alert(
                    timestamp=datetime.now(),
                    alert_type='data_freshness',
                    severity='medium',
                    message=f"No predictions received for {time_since_last}",
                    metric_name='data_freshness',
                    current_value=time_since_last.total_seconds() / 3600,
                    threshold=6.0,
                    suggested_action='Check data pipeline and model deployment'
                )
                self.recent_alerts.append(alert)
          # Check for prediction confidence anomalies
        if len(self.recent_predictions) >= 10:
            recent_confidences = [
                p.get('confidence_score', 0.5) 
                for p in list(self.recent_predictions)[-10:]
                if p.get('confidence_score') is not None
            ]
            
            if recent_confidences:
                avg_confidence = np.mean(recent_confidences)
                if avg_confidence < 0.3:
                    alert = Alert(
                        timestamp=datetime.now(),
                        alert_type='low_confidence',
                        severity='medium',
                        message=f"Average prediction confidence dropped to {avg_confidence:.3f}",
                        metric_name='confidence',
                        current_value=float(avg_confidence),
                        threshold=0.3,
                        suggested_action='Review model inputs and feature quality'
                    )
                    self.recent_alerts.append(alert)
    
    def export_metrics(self, filepath: str, hours: int = 24) -> None:
        """
        Export metrics to a file.
        
        Args:
            filepath: Path to export file
            hours: Number of hours of data to export
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with sqlite3.connect(self.db_path) as conn:
            metrics_df = pd.read_sql_query('''
                SELECT * FROM metrics 
                WHERE datetime(timestamp) >= datetime(?)
                ORDER BY timestamp DESC
            ''', conn, params=(cutoff_time.isoformat(),))
        
        metrics_df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(metrics_df)} metrics to {filepath}")


# Example alert callback functions
def email_alert_callback(alert: Alert) -> None:
    """Example email alert callback."""
    logger.warning(f"EMAIL ALERT [{alert.severity.upper()}]: {alert.message}")
    # Here you would integrate with your email system

def slack_alert_callback(alert: Alert) -> None:
    """Example Slack alert callback."""
    logger.warning(f"SLACK ALERT [{alert.severity.upper()}]: {alert.message}")
    # Here you would integrate with Slack API

def retraining_callback(alert: Alert) -> None:
    """Example retraining callback."""
    logger.critical(f"RETRAINING TRIGGERED: {alert.message}")
    # Here you would trigger your model retraining pipeline


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Initialize monitor
    monitor = PerformanceMonitor()
    
    # Add callbacks
    monitor.add_alert_callback(email_alert_callback)
    monitor.add_alert_callback(slack_alert_callback)
    monitor.add_retraining_callback(retraining_callback)
    
    # Simulate some predictions and metrics
    for i in range(20):
        # Log a prediction
        predicted_probs = np.random.dirichlet([2, 1, 2])
        monitor.log_prediction(
            match_id=f"match_{i}",
            home_team="Team A",
            away_team="Team B", 
            predicted_probs=predicted_probs,
            model_version="v1.0",
            confidence_score=np.random.uniform(0.3, 0.9),
            actual_outcome=np.random.choice(3)
        )
        
        # Log some metrics (simulate declining performance)
        accuracy = 0.45 - (i * 0.01)  # Declining accuracy
        brier_score = 0.25 + (i * 0.005)  # Increasing Brier score
        
        monitor.log_metric(PerformanceMetric(
            timestamp=datetime.now(),
            metric_name='accuracy',
            metric_value=accuracy,
            model_version='v1.0',
            data_source='validation_set'
        ))
        
        monitor.log_metric(PerformanceMetric(
            timestamp=datetime.now(),
            metric_name='brier_score', 
            metric_value=brier_score,
            model_version='v1.0',
            data_source='validation_set'
        ))
        
        time.sleep(0.1)  # Small delay
    
    # Get performance summary
    summary = monitor.get_performance_summary(hours=1)
    print("\n=== PERFORMANCE SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    
    # Export metrics
    monitor.export_metrics("performance_metrics.csv", hours=1)
    
    print(f"\nTotal alerts generated: {len(monitor.recent_alerts)}")
    for alert in monitor.recent_alerts:
        print(f"[{alert.severity.upper()}] {alert.message}")
