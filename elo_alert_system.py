"""
Real-Time Alert System for Auto-Updating ELO System
Advanced alerting, notification, and automated response system.

Author: GitHub Copilot
Date: May 27, 2025
Version: 2.0.0
"""

import json
import logging
import smtplib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sqlite3
import threading
from collections import defaultdict
from enum import Enum
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/alert_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class AlertType(Enum):
    """Types of alerts"""
    SYSTEM_HEALTH = "system_health"
    PERFORMANCE = "performance"
    DATA_QUALITY = "data_quality"
    SECURITY = "security"
    BUSINESS_LOGIC = "business_logic"

@dataclass
class Alert:
    """Alert data structure"""
    id: str
    timestamp: datetime
    severity: AlertSeverity
    alert_type: AlertType
    title: str
    message: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolution_notes: str = ""
    
@dataclass
class AlertRule:
    """Alert rule configuration"""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    alert_type: AlertType
    message_template: str
    cooldown_minutes: int = 5
    auto_resolve: bool = False
    resolution_condition: Optional[Callable[[Dict[str, Any]], bool]] = None

@dataclass
class NotificationConfig:
    """Configuration for alert notifications"""
    email_enabled: bool = False
    smtp_server: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_recipients: List[str] = field(default_factory=list)
    webhook_url: Optional[str] = None
    slack_webhook: Optional[str] = None

class AlertManager:
    """
    Advanced alert management system with rule-based alerting,
    notification routing, and automated response capabilities.
    """
    
    def __init__(self, db_path: str = 'data/alert_system.db',
                 notification_config: Optional[NotificationConfig] = None):
        """Initialize the alert manager"""
        self.db_path = db_path
        self.notification_config = notification_config or NotificationConfig()
        
        # Alert rules registry
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Active alerts tracking
        self.active_alerts: Dict[str, Alert] = {}
        
        # Alert history and statistics
        self.alert_history: List[Alert] = []
        self.alert_counts: Dict[AlertType, int] = defaultdict(int)
        
        # Cooldown tracking to prevent spam
        self.alert_cooldowns: Dict[str, datetime] = {}
        
        # Auto-response handlers
        self.response_handlers: Dict[AlertType, List[Callable]] = defaultdict(list)
        
        # Initialize database
        self._init_database()
        
        # Register default alert rules
        self._register_default_rules()
        
        logger.info("Alert Manager initialized")
    
    def _init_database(self):
        """Initialize SQLite database for alert storage"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TEXT,
                    resolution_notes TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alert_rules (
                    name TEXT PRIMARY KEY,
                    severity TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message_template TEXT NOT NULL,
                    cooldown_minutes INTEGER DEFAULT 5,
                    auto_resolve BOOLEAN DEFAULT FALSE,
                    enabled BOOLEAN DEFAULT TRUE
                )
            ''')
            
        logger.info(f"Alert database initialized at {self.db_path}")
    
    def _register_default_rules(self):
        """Register default alert rules for ELO system monitoring"""
        
        # System health rules
        self.register_rule(AlertRule(
            name="high_cpu_usage",
            condition=lambda data: data.get('cpu_usage', 0) > 85,
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.SYSTEM_HEALTH,
            message_template="High CPU usage detected: {cpu_usage:.1f}%",
            cooldown_minutes=10
        ))
        
        self.register_rule(AlertRule(
            name="high_memory_usage",
            condition=lambda data: data.get('memory_usage', 0) > 90,
            severity=AlertSeverity.ERROR,
            alert_type=AlertType.SYSTEM_HEALTH,
            message_template="High memory usage detected: {memory_usage:.1f}%",
            cooldown_minutes=5
        ))
        
        self.register_rule(AlertRule(
            name="disk_space_low",
            condition=lambda data: data.get('disk_usage', 0) > 95,
            severity=AlertSeverity.CRITICAL,
            alert_type=AlertType.SYSTEM_HEALTH,
            message_template="Disk space critically low: {disk_usage:.1f}% used",
            cooldown_minutes=30
        ))
        
        # Performance rules
        self.register_rule(AlertRule(
            name="slow_operations",
            condition=lambda data: data.get('average_operation_time', 0) > 3.0,
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.PERFORMANCE,
            message_template="Slow ELO operations detected: {average_operation_time:.2f}s average",
            cooldown_minutes=15
        ))
        
        self.register_rule(AlertRule(
            name="low_cache_hit_rate",
            condition=lambda data: data.get('cache_hit_rate', 100) < 70,
            severity=AlertSeverity.INFO,
            alert_type=AlertType.PERFORMANCE,
            message_template="Low cache hit rate: {cache_hit_rate:.1f}%",
            cooldown_minutes=30
        ))
        
        self.register_rule(AlertRule(
            name="high_error_rate",
            condition=lambda data: data.get('error_count', 0) > 5,
            severity=AlertSeverity.ERROR,
            alert_type=AlertType.PERFORMANCE,
            message_template="High error rate detected: {error_count} errors",
            cooldown_minutes=10
        ))
        
        # Data quality rules
        self.register_rule(AlertRule(
            name="unusual_rating_variance",
            condition=lambda data: data.get('rating_variance', 0) > 50000,
            severity=AlertSeverity.WARNING,
            alert_type=AlertType.DATA_QUALITY,
            message_template="Unusual rating variance detected: {rating_variance:.0f}",
            cooldown_minutes=60
        ))
        
        self.register_rule(AlertRule(
            name="rapid_team_addition",
            condition=lambda data: data.get('new_teams_added', 0) > 50,
            severity=AlertSeverity.INFO,
            alert_type=AlertType.DATA_QUALITY,
            message_template="Rapid team addition detected: {new_teams_added} new teams",
            cooldown_minutes=120
        ))
        
        logger.info(f"Registered {len(self.alert_rules)} default alert rules")
    
    def register_rule(self, rule: AlertRule):
        """Register a new alert rule"""
        self.alert_rules[rule.name] = rule
        
        # Store in database
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT OR REPLACE INTO alert_rules 
                    (name, severity, alert_type, message_template, cooldown_minutes, auto_resolve)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    rule.name,
                    rule.severity.value,
                    rule.alert_type.value,
                    rule.message_template,
                    rule.cooldown_minutes,
                    rule.auto_resolve
                ))
        except Exception as e:
            logger.error(f"Error storing alert rule: {e}")
        
        logger.info(f"Registered alert rule: {rule.name}")
    
    def evaluate_rules(self, data: Dict[str, Any]) -> List[Alert]:
        """Evaluate all active rules against provided data"""
        triggered_alerts = []
        
        for rule_name, rule in self.alert_rules.items():
            try:
                # Check cooldown
                if self._is_in_cooldown(rule_name):
                    continue
                
                # Evaluate condition
                if rule.condition(data):
                    alert = self._create_alert(rule, data)
                    triggered_alerts.append(alert)
                    
                    # Set cooldown
                    self.alert_cooldowns[rule_name] = datetime.now()
                    
                    logger.info(f"Alert triggered: {rule_name}")
                
                # Check for auto-resolution
                elif rule.auto_resolve and rule_name in self.active_alerts:
                    if rule.resolution_condition and rule.resolution_condition(data):
                        self._resolve_alert(rule_name, "Auto-resolved by rule condition")
                
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
        
        return triggered_alerts
    
    def _is_in_cooldown(self, rule_name: str) -> bool:
        """Check if alert rule is in cooldown period"""
        if rule_name not in self.alert_cooldowns:
            return False
        
        rule = self.alert_rules.get(rule_name)
        if not rule:
            return False
        
        cooldown_end = self.alert_cooldowns[rule_name] + timedelta(minutes=rule.cooldown_minutes)
        return datetime.now() < cooldown_end
    
    def _create_alert(self, rule: AlertRule, data: Dict[str, Any]) -> Alert:
        """Create an alert from a triggered rule"""
        alert_id = f"{rule.name}_{int(time.time())}"
        
        # Format message with data
        try:
            message = rule.message_template.format(**data)
        except:
            message = rule.message_template
        
        alert = Alert(
            id=alert_id,
            timestamp=datetime.now(),
            severity=rule.severity,
            alert_type=rule.alert_type,
            title=rule.name.replace('_', ' ').title(),
            message=message,
            source="elo_alert_system",
            metadata=data.copy()
        )
        
        # Store alert
        self._store_alert(alert)
        
        # Add to active alerts
        self.active_alerts[alert_id] = alert
        
        # Update statistics
        self.alert_counts[rule.alert_type] += 1
        
        return alert
    
    def _store_alert(self, alert: Alert):
        """Store alert in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO alerts 
                    (id, timestamp, severity, alert_type, title, message, source, 
                     metadata, resolved, resolved_at, resolution_notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.id,
                    alert.timestamp.isoformat(),
                    alert.severity.value,
                    alert.alert_type.value,
                    alert.title,
                    alert.message,
                    alert.source,
                    json.dumps(alert.metadata),
                    alert.resolved,
                    alert.resolved_at.isoformat() if alert.resolved_at else None,
                    alert.resolution_notes
                ))
        except Exception as e:
            logger.error(f"Error storing alert: {e}")
    
    def process_alerts(self, alerts: List[Alert]):
        """Process triggered alerts - send notifications and trigger responses"""
        for alert in alerts:
            try:
                # Send notifications
                if self.notification_config.email_enabled:
                    self._send_email_notification(alert)
                
                if self.notification_config.webhook_url:
                    self._send_webhook_notification(alert)
                
                # Trigger automated responses
                for handler in self.response_handlers[alert.alert_type]:
                    threading.Thread(
                        target=self._safe_execute_handler,
                        args=(handler, alert),
                        daemon=True
                    ).start()
                
                logger.info(f"Processed alert: {alert.title}")
                
            except Exception as e:
                logger.error(f"Error processing alert {alert.id}: {e}")
    
    def _send_email_notification(self, alert: Alert):
        """Send email notification for alert"""
        try:
            if not self.notification_config.email_recipients:
                return
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = self.notification_config.smtp_username
            msg['To'] = ', '.join(self.notification_config.email_recipients)
            msg['Subject'] = f"[{alert.severity.value}] ELO System Alert: {alert.title}"
            
            # Email body
            body = f"""
ELO System Alert

Severity: {alert.severity.value}
Type: {alert.alert_type.value}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

Message: {alert.message}

Alert ID: {alert.id}
Source: {alert.source}

Metadata:
{json.dumps(alert.metadata, indent=2)}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.notification_config.smtp_server, self.notification_config.smtp_port) as server:
                server.starttls()
                server.login(self.notification_config.smtp_username, self.notification_config.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email notification sent for alert: {alert.id}")
            
        except Exception as e:
            logger.error(f"Error sending email notification: {e}")
    
    def _send_webhook_notification(self, alert: Alert):
        """Send webhook notification for alert"""
        try:
            import requests

            url = self.notification_config.webhook_url
            if url is None:
                return
            
            payload = {
                'alert_id': alert.id,
                'timestamp': alert.timestamp.isoformat(),
                'severity': alert.severity.value,
                'type': alert.alert_type.value,
                'title': alert.title,
                'message': alert.message,
                'source': alert.source,
                'metadata': alert.metadata
            }
            
            response = requests.post(
                url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
            logger.info(f"Webhook notification sent for alert: {alert.id}")
            
        except Exception as e:
            logger.error(f"Error sending webhook notification: {e}")
    
    def _safe_execute_handler(self, handler: Callable, alert: Alert):
        """Safely execute response handler"""
        try:
            handler(alert)
        except Exception as e:
            logger.error(f"Error in response handler: {e}")
    
    def register_response_handler(self, alert_type: AlertType, handler: Callable[[Alert], None]):
        """Register automated response handler for alert type"""
        self.response_handlers[alert_type].append(handler)
        logger.info(f"Registered response handler for {alert_type.value}")
    
    def resolve_alert(self, alert_id: str, resolution_notes: str = ""):
        """Manually resolve an alert"""
        self._resolve_alert(alert_id, resolution_notes)
    
    def _resolve_alert(self, alert_id: str, resolution_notes: str):
        """Internal method to resolve an alert"""
        try:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                alert.resolution_notes = resolution_notes
                
                # Update database
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        UPDATE alerts 
                        SET resolved = TRUE, resolved_at = ?, resolution_notes = ?
                        WHERE id = ?
                    ''', (alert.resolved_at.isoformat(), resolution_notes, alert_id))
                
                # Remove from active alerts
                del self.active_alerts[alert_id]
                
                logger.info(f"Alert resolved: {alert_id}")
        
        except Exception as e:
            logger.error(f"Error resolving alert {alert_id}: {e}")
    
    def get_alert_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert statistics for specified time period"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total alerts in period
                cursor.execute('''
                    SELECT COUNT(*) FROM alerts 
                    WHERE timestamp >= ? AND timestamp <= ?
                ''', (start_time.isoformat(), end_time.isoformat()))
                total_alerts = cursor.fetchone()[0]
                
                # Alerts by severity
                cursor.execute('''
                    SELECT severity, COUNT(*) FROM alerts 
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY severity
                ''', (start_time.isoformat(), end_time.isoformat()))
                by_severity = dict(cursor.fetchall())
                
                # Alerts by type
                cursor.execute('''
                    SELECT alert_type, COUNT(*) FROM alerts 
                    WHERE timestamp >= ? AND timestamp <= ?
                    GROUP BY alert_type
                ''', (start_time.isoformat(), end_time.isoformat()))
                by_type = dict(cursor.fetchall())
                
                # Resolution rate
                cursor.execute('''
                    SELECT COUNT(*) FROM alerts 
                    WHERE timestamp >= ? AND timestamp <= ? AND resolved = TRUE
                ''', (start_time.isoformat(), end_time.isoformat()))
                resolved_count = cursor.fetchone()[0]
                
                resolution_rate = (resolved_count / total_alerts * 100) if total_alerts > 0 else 0
                
                return {
                    'period_hours': hours,
                    'total_alerts': total_alerts,
                    'active_alerts': len(self.active_alerts),
                    'resolved_alerts': resolved_count,
                    'resolution_rate': resolution_rate,
                    'alerts_by_severity': by_severity,
                    'alerts_by_type': by_type,
                    'generated_at': datetime.now().isoformat()
                }
        
        except Exception as e:
            logger.error(f"Error getting alert statistics: {e}")
            return {'error': str(e)}

# Example response handlers
def cpu_spike_handler(alert: Alert):
    """Example handler for CPU spike alerts"""
    logger.info(f"CPU spike detected: {alert.message}")
    # Could trigger process cleanup, scaling, etc.

def memory_leak_handler(alert: Alert):
    """Example handler for memory alerts"""
    logger.info(f"Memory issue detected: {alert.message}")
    # Could trigger garbage collection, restart services, etc.

def performance_degradation_handler(alert: Alert):
    """Example handler for performance alerts"""
    logger.info(f"Performance issue detected: {alert.message}")
    # Could trigger cache clearing, optimization routines, etc.

if __name__ == "__main__":
    # Example usage
    alert_manager = AlertManager()
    
    # Register response handlers
    alert_manager.register_response_handler(AlertType.SYSTEM_HEALTH, cpu_spike_handler)
    alert_manager.register_response_handler(AlertType.SYSTEM_HEALTH, memory_leak_handler)
    alert_manager.register_response_handler(AlertType.PERFORMANCE, performance_degradation_handler)
    
    # Example alert evaluation
    test_data = {
        'cpu_usage': 90.0,
        'memory_usage': 75.0,
        'average_operation_time': 4.5,
        'cache_hit_rate': 65.0,
        'error_count': 8
    }
    
    alerts = alert_manager.evaluate_rules(test_data)
    alert_manager.process_alerts(alerts)
    
    # Get statistics
    stats = alert_manager.get_alert_statistics(hours=1)
    print(json.dumps(stats, indent=2))
