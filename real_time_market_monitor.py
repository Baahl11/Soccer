"""
Real-time market monitor for soccer prediction system.

This module provides real-time monitoring of market movements and
sends alerts when significant changes are detected.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta
import json
import os
from market_integration import MarketDataIntegrator, create_market_monitor
from odds_analyzer import OddsAnalyzer
from market_visualizations import create_visualizations_for_fixture

logger = logging.getLogger(__name__)

# Configuration
MONITOR_INTERVAL = 300  # Check every 5 minutes by default
ALERT_THRESHOLD = 0.05  # Alert on 5% odds change
VISUALIZATION_ON_ALERT = True  # Generate visualizations when alerts occur

class RealTimeMarketMonitor:
    """
    Real-time monitor for market movements.
    """
    def __init__(self, check_interval: int = MONITOR_INTERVAL):
        """
        Initialize the real-time market monitor.
        
        Args:
            check_interval: Interval in seconds between market checks
        """
        self.check_interval = check_interval
        self.monitored_fixtures: Dict[int, Dict[str, Any]] = {}
        self.alert_handlers: List[Callable[[Dict[str, Any]], None]] = []
        self.is_running = False
        self.monitor_thread = None
        self.integrator = MarketDataIntegrator()
        self.odds_analyzer = OddsAnalyzer()
        self.last_alert_time: Dict[int, datetime] = {}
        
        # Create log directory for alerts
        self.alert_log_dir = "market_alerts"
        if not os.path.exists(self.alert_log_dir):
            os.makedirs(self.alert_log_dir)
    
    def add_fixture(self, fixture_id: int, fixture_data: Optional[Dict[str, Any]] = None) -> None:
        """
        Add a fixture to the monitoring list.
        
        Args:
            fixture_id: ID of the fixture to monitor
            fixture_data: Optional fixture data for context
        """
        if fixture_id not in self.monitored_fixtures:
            # Initialize monitoring data
            if fixture_data is None:
                from data import get_fixture_data
                fixture_data = get_fixture_data(fixture_id) or {}
            
            self.monitored_fixtures[fixture_id] = {
                "fixture_id": fixture_id,
                "fixture_data": fixture_data,
                "last_checked": None,
                "initial_odds": None,
                "latest_odds": None,
                "significant_movements": False,
                "latest_movement_data": None,
                "alerts": []
            }
            
            logger.info(f"Added fixture {fixture_id} to real-time monitoring")
            
            # Initial odds snapshot if already running
            if self.is_running:
                self._check_fixture(fixture_id)
    
    def remove_fixture(self, fixture_id: int) -> None:
        """
        Remove a fixture from the monitoring list.
        
        Args:
            fixture_id: ID of the fixture to remove
        """
        if fixture_id in self.monitored_fixtures:
            del self.monitored_fixtures[fixture_id]
            logger.info(f"Removed fixture {fixture_id} from real-time monitoring")
    
    def add_alert_handler(self, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Add a callback handler for market movement alerts.
        
        Args:
            handler: Callback function that takes alert data as argument
        """
        self.alert_handlers.append(handler)
    
    def start(self) -> None:
        """
        Start the real-time market monitoring thread.
        """
        if not self.is_running:
            self.is_running = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            logger.info(f"Real-time market monitoring started with interval of {self.check_interval} seconds")
    
    def stop(self) -> None:
        """
        Stop the real-time market monitoring thread.
        """
        if self.is_running:
            self.is_running = False
            if self.monitor_thread:
                # Allow the thread to terminate naturally at next check
                self.monitor_thread.join(timeout=self.check_interval + 5)
            logger.info("Real-time market monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """
        Main monitoring loop that periodically checks for market movements.
        """
        while self.is_running:
            try:
                # Check all monitored fixtures
                for fixture_id in list(self.monitored_fixtures.keys()):
                    self._check_fixture(fixture_id)
                
                # Sleep for the check interval
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in market monitoring loop: {e}")
                # Continue the loop after error
                time.sleep(self.check_interval)
    
    def _check_fixture(self, fixture_id: int) -> None:
        """
        Check a specific fixture for market movements.
        
        Args:
            fixture_id: ID of the fixture to check
        """
        try:
            fixture_data = self.monitored_fixtures[fixture_id]
            
            # Get current odds data
            odds_data = self.odds_analyzer.get_fixture_odds(fixture_id)
            if not odds_data:
                logger.warning(f"Could not retrieve odds for fixture {fixture_id}")
                return
            
            # Record in tracker for history
            self.odds_analyzer.movement_tracker.record_odds(fixture_id, odds_data)
            
            # Update monitoring data
            now = datetime.now()
            fixture_data["last_checked"] = now
            
            # Store initial odds if first check
            if fixture_data["initial_odds"] is None:
                fixture_data["initial_odds"] = odds_data
            
            fixture_data["latest_odds"] = odds_data
            
            # Analyze for significant movements
            movement_data = self.integrator.analyze_market_movements(fixture_id)
            fixture_data["latest_movement_data"] = movement_data
            
            # Check if movements are significant
            if movement_data.get("significant_movements", False):
                fixture_data["significant_movements"] = True
                
                # Determine if we should alert (avoid too frequent alerts)
                should_alert = False
                if fixture_id not in self.last_alert_time:
                    should_alert = True
                else:
                    # Alert if more than 30 minutes since last alert
                    last_alert = self.last_alert_time[fixture_id]
                    if (now - last_alert).total_seconds() > 1800:  # 30 minutes
                        should_alert = True
                
                if should_alert:
                    self._trigger_alert(fixture_id, movement_data)
                    self.last_alert_time[fixture_id] = now
                    
                    # Generate visualizations if enabled
                    if VISUALIZATION_ON_ALERT:
                        viz_paths = create_visualizations_for_fixture(fixture_id)
                        fixture_data["visualizations"] = viz_paths
            
        except Exception as e:
            logger.error(f"Error checking fixture {fixture_id}: {e}")
    
    def _trigger_alert(self, fixture_id: int, movement_data: Dict[str, Any]) -> None:
        """
        Trigger alerts for significant market movements.
        
        Args:
            fixture_id: ID of the fixture
            movement_data: Movement analysis data
        """
        try:
            # Build alert data
            fixture_data = self.monitored_fixtures[fixture_id]
            fixture_info = fixture_data["fixture_data"]
            
            alert_data = {
                "fixture_id": fixture_id,
                "home_team": fixture_info.get("teams", {}).get("home", {}).get("name", "Home"),
                "away_team": fixture_info.get("teams", {}).get("away", {}).get("name", "Away"),
                "league": fixture_info.get("league", {}).get("name", "Unknown"),
                "match_date": fixture_info.get("fixture", {}).get("date", "Unknown"),
                "alert_time": datetime.now().isoformat(),
                "movements": movement_data.get("movements", []),
                "implications": movement_data.get("implications", [])
            }
            
            # Add to fixture alert history
            fixture_data["alerts"].append(alert_data)
            
            # Log the alert
            self._log_alert(alert_data)
            
            # Call all alert handlers
            for handler in self.alert_handlers:
                try:
                    handler(alert_data)
                except Exception as e:
                    logger.error(f"Error in alert handler: {e}")
                    
        except Exception as e:
            logger.error(f"Error triggering alert for fixture {fixture_id}: {e}")
    
    def _log_alert(self, alert_data: Dict[str, Any]) -> None:
        """
        Log alert data to file.
        
        Args:
            alert_data: Alert data to log
        """
        try:
            fixture_id = alert_data["fixture_id"]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"alert_{fixture_id}_{timestamp}.json"
            filepath = os.path.join(self.alert_log_dir, filename)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(alert_data, f, indent=2)
                
            logger.info(f"Market alert logged to {filepath}")
            
        except Exception as e:
            logger.error(f"Error logging market alert: {e}")
    
    def get_monitored_fixtures(self) -> Dict[int, Dict[str, Any]]:
        """
        Get data for all monitored fixtures.
        
        Returns:
            Dictionary with monitoring data for all fixtures
        """
        return self.monitored_fixtures
    
    def get_fixture_status(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        """
        Get monitoring status for a specific fixture.
        
        Args:
            fixture_id: ID of the fixture
            
        Returns:
            Monitoring data for the fixture or None if not monitored
        """
        return self.monitored_fixtures.get(fixture_id)

# Singleton instance for use across the application
_monitor_instance = None

def get_market_monitor(check_interval: int = MONITOR_INTERVAL) -> RealTimeMarketMonitor:
    """
    Get or create the singleton RealTimeMarketMonitor instance.
    
    Args:
        check_interval: Interval in seconds between market checks
        
    Returns:
        The global RealTimeMarketMonitor instance
    """
    global _monitor_instance
    if _monitor_instance is None:
        _monitor_instance = RealTimeMarketMonitor(check_interval=check_interval)
    return _monitor_instance

def monitor_upcoming_fixtures(days_ahead: int = 3) -> None:
    """
    Start monitoring upcoming fixtures.
    
    Args:
        days_ahead: Number of days ahead to include fixtures for
    """
    try:
        from data import get_upcoming_fixtures
        
        # Get upcoming fixtures
        fixtures = get_upcoming_fixtures(days_ahead)
        if not fixtures:
            logger.warning(f"No upcoming fixtures found for the next {days_ahead} days")
            return
        
        # Get monitor instance
        monitor = get_market_monitor()
        
        # Add fixtures to monitor
        for fixture in fixtures:
            fixture_id = fixture.get("fixture", {}).get("id")
            if fixture_id:
                monitor.add_fixture(fixture_id, fixture_data=fixture)
        
        # Start monitoring if not already running
        if not monitor.is_running:
            monitor.start()
            
        logger.info(f"Started monitoring {len(fixtures)} upcoming fixtures")
        
    except Exception as e:
        logger.error(f"Error setting up monitoring for upcoming fixtures: {e}")

def print_console_notification(alert_data: Dict[str, Any]) -> None:
    """
    Print a notification to the console when a significant movement is detected.
    
    Args:
        alert_data: Alert data
    """
    fixture_id = alert_data["fixture_id"]
    home_team = alert_data["home_team"]
    away_team = alert_data["away_team"]
    league = alert_data["league"]
    
    print("\n" + "="*80)
    print(f"⚠️  SIGNIFICANT MARKET MOVEMENT DETECTED ⚠️")
    print(f"Fixture: {home_team} vs {away_team} (ID: {fixture_id})")
    print(f"League: {league}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-"*80)
    
    # Print movements
    movements = alert_data.get("movements", [])
    if movements:
        print("Detected Movements:")
        for idx, movement in enumerate(movements, 1):
            market = movement.get("market", "Unknown")
            selection = movement.get("selection", "Unknown")
            change = movement.get("change", 0)
            trend = movement.get("trend", "Unknown")
            
            direction = "⬇️ DECREASING" if trend == "decreasing" else "⬆️ INCREASING"
            print(f"  {idx}. {market} - {selection}: {direction} by {abs(change)*100:.1f}%")
    
    # Print implications
    implications = alert_data.get("implications", [])
    if implications:
        print("\nImplications:")
        for idx, imp in enumerate(implications, 1):
            print(f"  {idx}. {imp.get('description', '')}")
    
    print("="*80 + "\n")

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Register console notification handler
    monitor = get_market_monitor(check_interval=60)  # check every minute in demo mode
    monitor.add_alert_handler(print_console_notification)
    
    # Start monitoring upcoming fixtures
    monitor_upcoming_fixtures(days_ahead=3)
    
    # Keep the script running
    try:
        print("Market monitoring active. Press Ctrl+C to exit.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        monitor.stop()
        print("Market monitoring stopped.")
