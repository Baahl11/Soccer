"""
Advanced Performance Monitoring Dashboard for Auto-Updating ELO System
Real-time monitoring, analytics, and reporting for system performance optimization.

Author: GitHub Copilot
Date: May 27, 2025
Version: 2.0.0
"""

import json
import logging
import time
import statistics
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import sqlite3
import threading
import psutil
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/performance_dashboard.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SystemHealthMetrics:
    """System health and resource usage metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_io: Dict[str, int] = field(default_factory=dict)
    process_count: int = 0
    
@dataclass
class EloPerformanceMetrics:
    """Enhanced ELO system performance metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    total_operations: int = 0
    new_teams_added: int = 0
    ratings_updated: int = 0
    average_operation_time: float = 0.0
    cache_hit_rate: float = 0.0
    error_count: int = 0
    matches_processed: int = 0
    leagues_active: int = 0
    rating_variance: float = 0.0
    
@dataclass
class AlertConfig:
    """Configuration for performance alerts"""
    cpu_threshold: float = 80.0
    memory_threshold: float = 85.0
    error_rate_threshold: float = 5.0
    operation_time_threshold: float = 2.0
    cache_hit_rate_threshold: float = 75.0

class PerformanceDashboard:
    """
    Advanced performance monitoring dashboard with real-time analytics,
    system health monitoring, and automated alerting.
    """
    
    def __init__(self, db_path: str = 'data/performance_metrics.db', 
                 alert_config: Optional[AlertConfig] = None):
        """Initialize the performance dashboard"""
        self.db_path = db_path
        self.alert_config = alert_config or AlertConfig()
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Recent metrics cache (last 100 readings)
        self.recent_system_metrics = deque(maxlen=100)
        self.recent_elo_metrics = deque(maxlen=100)
        
        # Alert history
        self.alert_history = []
        
        # Initialize database
        self._init_database()
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logger.info("Performance Dashboard initialized")
    
    def _init_database(self):
        """Initialize SQLite database for metrics storage"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    disk_usage REAL,
                    network_bytes_sent INTEGER,
                    network_bytes_recv INTEGER,
                    process_count INTEGER
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS elo_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    total_operations INTEGER,
                    new_teams_added INTEGER,
                    ratings_updated INTEGER,
                    average_operation_time REAL,
                    cache_hit_rate REAL,
                    error_count INTEGER,
                    matches_processed INTEGER,
                    leagues_active INTEGER,
                    rating_variance REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    alert_type TEXT,
                    severity TEXT,
                    message TEXT,
                    resolved BOOLEAN DEFAULT FALSE
                )
            ''')
            
        logger.info(f"Database initialized at {self.db_path}")
    
    def collect_system_metrics(self) -> SystemHealthMetrics:
        """Collect current system health metrics"""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Network I/O
            net_io = psutil.net_io_counters()
            network_data = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            metrics = SystemHealthMetrics(
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_io=network_data,
                process_count=process_count
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemHealthMetrics()
    
    def collect_elo_metrics(self, elo_system) -> EloPerformanceMetrics:
        """Collect ELO system performance metrics"""
        try:
            from auto_updating_elo import METRICS
            
            # Calculate derived metrics
            total_ops = METRICS.total_operations
            cache_total = METRICS.cache_hits + METRICS.cache_misses
            cache_hit_rate = (METRICS.cache_hits / cache_total * 100) if cache_total > 0 else 0
            
            # Calculate average operation time
            all_times = []
            for times_list in METRICS.operation_times.values():
                all_times.extend(times_list)
            avg_time = statistics.mean(all_times) if all_times else 0
            
            # Calculate rating variance across all teams
            try:
                all_ratings = list(elo_system.elo_rating.ratings.values())
                rating_variance = statistics.variance(all_ratings) if len(all_ratings) > 1 else 0
            except:
                rating_variance = 0
            
            # Count active leagues
            active_leagues = len(set(elo_system.league_tiers.keys()))
            
            metrics = EloPerformanceMetrics(
                total_operations=total_ops,
                new_teams_added=METRICS.new_teams_added,
                ratings_updated=len(METRICS.rating_adjustments),
                average_operation_time=avg_time,
                cache_hit_rate=cache_hit_rate,
                error_count=0,  # Will be enhanced later
                matches_processed=total_ops,  # Approximate
                leagues_active=active_leagues,
                rating_variance=rating_variance
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting ELO metrics: {e}")
            return EloPerformanceMetrics()
    
    def store_metrics(self, system_metrics: SystemHealthMetrics, 
                     elo_metrics: EloPerformanceMetrics):
        """Store metrics in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Store system metrics
                conn.execute('''
                    INSERT INTO system_metrics 
                    (timestamp, cpu_usage, memory_usage, disk_usage, 
                     network_bytes_sent, network_bytes_recv, process_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    system_metrics.timestamp.isoformat(),
                    system_metrics.cpu_usage,
                    system_metrics.memory_usage,
                    system_metrics.disk_usage,
                    system_metrics.network_io.get('bytes_sent', 0),
                    system_metrics.network_io.get('bytes_recv', 0),
                    system_metrics.process_count
                ))
                
                # Store ELO metrics
                conn.execute('''
                    INSERT INTO elo_metrics 
                    (timestamp, total_operations, new_teams_added, ratings_updated,
                     average_operation_time, cache_hit_rate, error_count,
                     matches_processed, leagues_active, rating_variance)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    elo_metrics.timestamp.isoformat(),
                    elo_metrics.total_operations,
                    elo_metrics.new_teams_added,
                    elo_metrics.ratings_updated,
                    elo_metrics.average_operation_time,
                    elo_metrics.cache_hit_rate,
                    elo_metrics.error_count,
                    elo_metrics.matches_processed,
                    elo_metrics.leagues_active,
                    elo_metrics.rating_variance
                ))
                
        except Exception as e:
            logger.error(f"Error storing metrics: {e}")
    
    def check_alerts(self, system_metrics: SystemHealthMetrics, 
                    elo_metrics: EloPerformanceMetrics):
        """Check for alert conditions and trigger alerts"""
        alerts = []
        
        # System health alerts
        if system_metrics.cpu_usage > self.alert_config.cpu_threshold:
            alerts.append({
                'type': 'system',
                'severity': 'WARNING',
                'message': f'High CPU usage: {system_metrics.cpu_usage:.1f}%'
            })
        
        if system_metrics.memory_usage > self.alert_config.memory_threshold:
            alerts.append({
                'type': 'system',
                'severity': 'WARNING',
                'message': f'High memory usage: {system_metrics.memory_usage:.1f}%'
            })
        
        # ELO system alerts
        if elo_metrics.average_operation_time > self.alert_config.operation_time_threshold:
            alerts.append({
                'type': 'performance',
                'severity': 'WARNING',
                'message': f'Slow operations: {elo_metrics.average_operation_time:.2f}s average'
            })
        
        if elo_metrics.cache_hit_rate < self.alert_config.cache_hit_rate_threshold:
            alerts.append({
                'type': 'performance',
                'severity': 'INFO',
                'message': f'Low cache hit rate: {elo_metrics.cache_hit_rate:.1f}%'
            })
        
        # Store and log alerts
        for alert in alerts:
            self._store_alert(alert)
            logger.warning(f"ALERT [{alert['severity']}]: {alert['message']}")
    
    def _store_alert(self, alert: Dict[str, str]):
        """Store alert in database and history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO alerts (timestamp, alert_type, severity, message)
                    VALUES (?, ?, ?, ?)
                ''', (
                    datetime.now().isoformat(),
                    alert['type'],
                    alert['severity'],
                    alert['message']
                ))
            
            self.alert_history.append({
                'timestamp': datetime.now(),
                **alert
            })
            
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    def start_monitoring(self, elo_system, interval: int = 60):
        """Start continuous monitoring in background thread"""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(elo_system, interval),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Started monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Stopped monitoring")
    
    def _monitoring_loop(self, elo_system, interval: int):
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect metrics
                system_metrics = self.collect_system_metrics()
                elo_metrics = self.collect_elo_metrics(elo_system)
                
                # Store in cache
                self.recent_system_metrics.append(system_metrics)
                self.recent_elo_metrics.append(elo_metrics)
                
                # Store in database
                self.store_metrics(system_metrics, elo_metrics)
                
                # Check for alerts
                self.check_alerts(system_metrics, elo_metrics)
                
                # Sleep until next interval
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval)
    
    def generate_dashboard_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate comprehensive dashboard report"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Query database for metrics
            with sqlite3.connect(self.db_path) as conn:
                # System metrics
                system_df = pd.read_sql_query('''
                    SELECT * FROM system_metrics 
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                ''', conn, params=[start_time.isoformat(), end_time.isoformat()])
                
                # ELO metrics
                elo_df = pd.read_sql_query('''
                    SELECT * FROM elo_metrics 
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                ''', conn, params=[start_time.isoformat(), end_time.isoformat()])
                
                # Alerts
                alerts_df = pd.read_sql_query('''
                    SELECT * FROM alerts 
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp DESC
                ''', conn, params=[start_time.isoformat(), end_time.isoformat()])
            
            # Generate summary statistics
            report = {
                'report_period': f'{hours} hours',
                'generated_at': datetime.now().isoformat(),
                'system_summary': self._generate_system_summary(system_df),
                'elo_summary': self._generate_elo_summary(elo_df),
                'alerts_summary': self._generate_alerts_summary(alerts_df),
                'recommendations': self._generate_recommendations(system_df, elo_df, alerts_df)
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating dashboard report: {e}")
            return {'error': str(e)}
    
    def _generate_system_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate system performance summary"""
        if df.empty:
            return {'status': 'No data available'}
        
        return {
            'avg_cpu_usage': df['cpu_usage'].mean(),
            'max_cpu_usage': df['cpu_usage'].max(),
            'avg_memory_usage': df['memory_usage'].mean(),
            'max_memory_usage': df['memory_usage'].max(),
            'avg_disk_usage': df['disk_usage'].mean(),
            'data_points': len(df)
        }
    
    def _generate_elo_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate ELO system summary"""
        if df.empty:
            return {'status': 'No data available'}
        
        latest = df.iloc[-1] if not df.empty else {}
        
        return {
            'total_operations': int(latest.get('total_operations', 0)),
            'new_teams_added': int(latest.get('new_teams_added', 0)),
            'avg_operation_time': df['average_operation_time'].mean(),
            'avg_cache_hit_rate': df['cache_hit_rate'].mean(),
            'active_leagues': int(latest.get('leagues_active', 0)),
            'rating_variance': latest.get('rating_variance', 0),
            'data_points': len(df)
        }
    
    def _generate_alerts_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate alerts summary"""
        if df.empty:
            return {'total_alerts': 0, 'by_severity': {}, 'by_type': {}}
        
        return {
            'total_alerts': len(df),
            'by_severity': df['severity'].value_counts().to_dict(),
            'by_type': df['alert_type'].value_counts().to_dict(),
            'recent_alerts': df.head(5).to_dict('records')
        }
    
    def _generate_recommendations(self, system_df: pd.DataFrame, 
                                elo_df: pd.DataFrame, 
                                alerts_df: pd.DataFrame) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # System recommendations
        if not system_df.empty:
            if system_df['cpu_usage'].mean() > 70:
                recommendations.append("Consider optimizing CPU-intensive operations")
            if system_df['memory_usage'].mean() > 80:
                recommendations.append("Monitor memory usage - consider implementing data cleanup")
        
        # ELO system recommendations
        if not elo_df.empty:
            if elo_df['cache_hit_rate'].mean() < 80:
                recommendations.append("Improve caching strategy for better performance")
            if elo_df['average_operation_time'].mean() > 1.0:
                recommendations.append("Optimize slow operations for better response time")
        
        # Alert-based recommendations
        if not alerts_df.empty and len(alerts_df) > 10:
            recommendations.append("High alert frequency - review system configuration")
        
        return recommendations
    
    def create_performance_plots(self, hours: int = 24, save_path: str = 'reports/performance_plots.png'):
        """Create comprehensive performance visualization plots"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            # Query data
            with sqlite3.connect(self.db_path) as conn:
                system_df = pd.read_sql_query('''
                    SELECT * FROM system_metrics 
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                ''', conn, params=[start_time.isoformat(), end_time.isoformat()])
                
                elo_df = pd.read_sql_query('''
                    SELECT * FROM elo_metrics 
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp
                ''', conn, params=[start_time.isoformat(), end_time.isoformat()])
            
            if system_df.empty and elo_df.empty:
                logger.warning("No data available for plotting")
                return
            
            # Convert timestamps
            if not system_df.empty:
                system_df['timestamp'] = pd.to_datetime(system_df['timestamp'])
            if not elo_df.empty:
                elo_df['timestamp'] = pd.to_datetime(elo_df['timestamp'])
            
            # Create subplot figure
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'ELO System Performance Dashboard - Last {hours} Hours', fontsize=16)
            
            # System CPU Usage
            if not system_df.empty:
                axes[0, 0].plot(system_df['timestamp'], system_df['cpu_usage'], 'b-', linewidth=2)
                axes[0, 0].axhline(y=80, color='r', linestyle='--', alpha=0.7, label='Warning Threshold')
                axes[0, 0].set_title('CPU Usage (%)')
                axes[0, 0].set_ylabel('Percentage')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # System Memory Usage
            if not system_df.empty:
                axes[0, 1].plot(system_df['timestamp'], system_df['memory_usage'], 'g-', linewidth=2)
                axes[0, 1].axhline(y=85, color='r', linestyle='--', alpha=0.7, label='Warning Threshold')
                axes[0, 1].set_title('Memory Usage (%)')
                axes[0, 1].set_ylabel('Percentage')
                axes[0, 1].legend()
                axes[0, 1].grid(True, alpha=0.3)
            
            # ELO Operations
            if not elo_df.empty:
                axes[0, 2].plot(elo_df['timestamp'], elo_df['total_operations'], 'm-', linewidth=2)
                axes[0, 2].set_title('Total ELO Operations')
                axes[0, 2].set_ylabel('Count')
                axes[0, 2].grid(True, alpha=0.3)
            
            # Cache Hit Rate
            if not elo_df.empty:
                axes[1, 0].plot(elo_df['timestamp'], elo_df['cache_hit_rate'], 'c-', linewidth=2)
                axes[1, 0].axhline(y=75, color='r', linestyle='--', alpha=0.7, label='Warning Threshold')
                axes[1, 0].set_title('Cache Hit Rate (%)')
                axes[1, 0].set_ylabel('Percentage')
                axes[1, 0].legend()
                axes[1, 0].grid(True, alpha=0.3)
            
            # Operation Time
            if not elo_df.empty:
                axes[1, 1].plot(elo_df['timestamp'], elo_df['average_operation_time'], 'orange', linewidth=2)
                axes[1, 1].axhline(y=2.0, color='r', linestyle='--', alpha=0.7, label='Warning Threshold')
                axes[1, 1].set_title('Average Operation Time (s)')
                axes[1, 1].set_ylabel('Seconds')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
            
            # Rating Variance
            if not elo_df.empty:
                axes[1, 2].plot(elo_df['timestamp'], elo_df['rating_variance'], 'brown', linewidth=2)
                axes[1, 2].set_title('Rating Variance')
                axes[1, 2].set_ylabel('Variance')
                axes[1, 2].grid(True, alpha=0.3)
            
            # Format x-axis for all subplots
            for ax in axes.flat:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
                ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Performance plots saved to {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating performance plots: {e}")

if __name__ == "__main__":
    # Example usage
    dashboard = PerformanceDashboard()
    
    # For testing without actual ELO system
    class MockEloSystem:
        def __init__(self):
            self.league_tiers = {i: 1500 + i for i in range(1, 10)}
            self.elo_rating = type('MockRating', (), {'ratings': {i: 1500 + i*10 for i in range(100)}})()
    
    mock_elo = MockEloSystem()
    
    # Collect sample metrics
    system_metrics = dashboard.collect_system_metrics()
    elo_metrics = dashboard.collect_elo_metrics(mock_elo)
    
    print("System Metrics:", asdict(system_metrics))
    print("ELO Metrics:", asdict(elo_metrics))
    
    # Store metrics
    dashboard.store_metrics(system_metrics, elo_metrics)
    
    # Generate report
    report = dashboard.generate_dashboard_report(hours=1)
    print("\nDashboard Report:", json.dumps(report, indent=2, default=str))
