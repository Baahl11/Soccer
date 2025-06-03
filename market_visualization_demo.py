"""
Demo script for market visualizations and real-time monitoring.

This script demonstrates both the market visualization capabilities and
the real-time market monitoring functionality.
"""

import logging
import time
import argparse
from typing import Dict, Any, List
import os
from datetime import datetime, timedelta

from market_visualizations import (
    plot_odds_evolution,
    plot_implied_probability_trends,
    plot_market_movement_analysis,
    plot_model_vs_market,
    create_visualizations_for_fixture
)
from real_time_market_monitor import (
    get_market_monitor,
    print_console_notification,
    monitor_upcoming_fixtures
)
from market_integration import MarketDataIntegrator, create_market_monitor
from predictions import make_global_prediction
from data import get_fixture_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_visualizations(fixture_id: int):
    """
    Demonstrate market visualizations for a specific fixture.
    
    Args:
        fixture_id: ID of the fixture to visualize
    """
    print(f"\n=== Market Visualizations Demo for Fixture {fixture_id} ===\n")
    
    # Create all visualizations
    viz_paths = create_visualizations_for_fixture(fixture_id)
    
    if not viz_paths:
        print("No visualizations could be generated. Check if market data is available.")
        return
    
    # Print paths to generated visualizations
    print(f"Generated {len(viz_paths)} visualizations:")
    for viz_type, path in viz_paths.items():
        print(f"  {viz_type}: {path}")
    
    # Attempt to open the first visualization in default image viewer
    if viz_paths:
        first_viz = next(iter(viz_paths.values()))
        try:
            import subprocess
            subprocess.run(['start', first_viz], shell=True, check=True)
            print(f"\nOpened visualization: {first_viz}")
        except Exception as e:
            print(f"Could not automatically open visualization: {e}")
            print(f"Please open the files manually to view them.")

def demo_real_time_monitoring(minutes: int = 5):
    """
    Demonstrate real-time market monitoring for a specified duration.
    
    Args:
        minutes: Number of minutes to run the monitoring demo
    """
    print(f"\n=== Real-Time Market Monitoring Demo ({minutes} minutes) ===\n")
    
    # Initialize monitor
    monitor = get_market_monitor(check_interval=30)  # Check every 30 seconds for demo
    
    # Register alert handler
    monitor.add_alert_handler(print_console_notification)
    
    # Start monitoring upcoming fixtures
    print("Setting up monitoring for upcoming fixtures...")
    monitor_upcoming_fixtures(days_ahead=2)
    
    # Print initial status
    fixtures = monitor.get_monitored_fixtures()
    print(f"Monitoring {len(fixtures)} fixtures in real-time.")
    
    if not fixtures:
        print("No fixtures are being monitored. Check if there are upcoming fixtures.")
        return
    
    print("\nMonitored fixtures:")
    for fixture_id, data in fixtures.items():
        fixture_info = data.get("fixture_data", {})
        home_team = fixture_info.get("teams", {}).get("home", {}).get("name", "Home")
        away_team = fixture_info.get("teams", {}).get("away", {}).get("name", "Away")
        match_date = fixture_info.get("fixture", {}).get("date", "Unknown")
        print(f"  ID: {fixture_id} - {home_team} vs {away_team} ({match_date})")
    
    # Run monitoring for specified duration
    print(f"\nMonitoring in progress for {minutes} minutes...")
    print("Press Ctrl+C to stop early.")
    
    try:
        # Start time
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=minutes)
        
        # Progress indicator
        while datetime.now() < end_time:
            elapsed = (datetime.now() - start_time).total_seconds()
            total = minutes * 60
            percent = min(100, int(elapsed / total * 100))
            
            print(f"\rProgress: {percent}% complete | Elapsed: {int(elapsed)}s | Press Ctrl+C to stop", end='')
            time.sleep(1)
            
        print("\n\nMonitoring demo completed.")
        
    except KeyboardInterrupt:
        print("\n\nMonitoring demo stopped by user.")
    finally:
        # Stop monitoring
        monitor.stop()
        
        # Show summary
        fixtures = monitor.get_monitored_fixtures()
        alerts_count = 0
        for fixture_id, data in fixtures.items():
            alerts_count += len(data.get("alerts", []))
        
        print(f"\nMonitoring summary:")
        print(f"  Fixtures monitored: {len(fixtures)}")
        print(f"  Alerts triggered: {alerts_count}")
        
        if alerts_count > 0:
            print("\nAlert logs saved to the 'market_alerts' directory.")

def run_combined_demo(fixture_id: int, monitor_minutes: int = 5):
    """
    Run both visualizations and monitoring demos.
    
    Args:
        fixture_id: ID of the fixture for visualizations
        monitor_minutes: Minutes to run the monitoring demo
    """
    # Run the visualizations demo
    demo_visualizations(fixture_id)
    
    # Run the real-time monitoring demo
    demo_real_time_monitoring(minutes=monitor_minutes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market Visualizations and Monitoring Demo")
    parser.add_argument("--fixture", type=int, default=1208825, help="Fixture ID for visualization demo")
    parser.add_argument("--time", type=int, default=5, help="Minutes to run monitoring demo")
    parser.add_argument("--viz-only", action="store_true", help="Run only visualizations demo")
    parser.add_argument("--monitor-only", action="store_true", help="Run only monitoring demo")
    
    args = parser.parse_args()
    
    if args.viz_only:
        demo_visualizations(args.fixture)
    elif args.monitor_only:
        demo_real_time_monitoring(minutes=args.time)
    else:
        run_combined_demo(args.fixture, args.time)
