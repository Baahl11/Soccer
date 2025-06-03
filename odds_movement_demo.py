"""
Odds Movement Demo

This script demonstrates the odds movement tracking functionality.
It shows how to:
1. Track odds movements for a fixture
2. Detect significant market shifts
3. Calibrate predictions using market data
"""

import logging
import time
from typing import Dict, Any, List, Optional
from odds_analyzer import OddsAnalyzer
from predictions import make_global_prediction
from data import get_fixture_data
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_odds_movement(fixture_id: int):
    """
    Demonstrate odds movement tracking for a fixture
    
    Args:
        fixture_id: The ID of the fixture to analyze
    """
    print(f"\n=== Odds Movement Demo for Fixture {fixture_id} ===\n")
    
    # Initialize the odds analyzer
    odds_analyzer = OddsAnalyzer()
    
    # Get the fixture information
    fixture_data = get_fixture_data(fixture_id)
    if not fixture_data or 'response' not in fixture_data or not fixture_data['response']:
        print(f"Error: Fixture {fixture_id} not found")
        return
        
    response = fixture_data['response'][0]
    teams = response.get('teams', {})
    home_team = teams.get('home', {}).get('name', 'Home Team')
    away_team = teams.get('away', {}).get('name', 'Away Team')
    
    print(f"Fixture: {home_team} vs {away_team}\n")      # Get initial odds
    print("Retrieving initial odds data...")
    initial_odds = odds_analyzer.get_fixture_odds(fixture_id)
    # Check if we have valid odds data
    if not initial_odds:
        print("Error: Could not retrieve odds data for this fixture")
        return
    
    # Simulate market recording by retrieving odds with slight modifications
    print("\nSimulating market activity...")
    
    # Function to simulate odds change
    def simulate_odds_change(odds_data: Dict[str, Any], market: str, selection: str, change_pct: float) -> Dict[str, Any]:
        """Simulate change in odds for a specific market selection"""
        modified_odds = odds_data.copy()
        bookmakers = modified_odds.get('bookmakers', [])
        
        for bookie in bookmakers:
            bets = bookie.get('bets', [])
            market_data = next((bet for bet in bets if bet.get('name') == market), None)
            
            if market_data:
                values = market_data.get('values', [])
                for value in values:
                    if value.get('value', '') == selection:
                        current_odd = float(value.get('odd', 0))
                        new_odd = current_odd * (1 + change_pct)
                        value['odd'] = str(round(new_odd, 2))
        
        return modified_odds
    
    # Simulate market activity with several recordings
    for i in range(3):
        # Simulate random changes to home and away odds
        if i == 0:
            # First change - small decrease in home odds
            modified_odds = simulate_odds_change(initial_odds, "Match Winner", "Home", -0.05)
            print(f"Recorded odds change: Home team odds decreased by 5%")
        elif i == 1:
            # Second change - increase in away odds
            modified_odds = simulate_odds_change(initial_odds, "Match Winner", "Away", 0.10)
            print(f"Recorded odds change: Away team odds increased by 10%")
        else:
            # Third change - large decrease in home odds (significant)
            modified_odds = simulate_odds_change(initial_odds, "Match Winner", "Home", -0.15)
            print(f"Recorded odds change: Home team odds decreased by 15%")
        
        # Record the modified odds
        odds_analyzer.movement_tracker.record_odds(fixture_id, modified_odds)
        time.sleep(1)  # Short delay to ensure timestamps differ
    
    # Analyze the movement
    print("\n--- Market Movement Analysis ---")
    movement_analysis = odds_analyzer.get_odds_movement_analysis(fixture_id)
    
    # Print movement analysis
    print(f"Movement detected: {movement_analysis.get('movement_detected', False)}")
    print(f"Market confidence: {movement_analysis.get('market_confidence', 0):.2f}")
    
    # Check for significant movements
    significant_movements = odds_analyzer.detect_significant_market_movements(fixture_id)
    
    print("\n--- Significant Market Movements ---")
    if significant_movements.get("significant_movements", False):
        for movement in significant_movements.get("movements", []):
            market = movement.get('market', 'Unknown')
            selection = movement.get('selection', 'Unknown')
            change = movement.get('change', 0)
            trend = movement.get('trend', 'Unknown')
            prev_odds = movement.get('previous_odds', 0)
            curr_odds = movement.get('current_odds', 0)
            
            print(f"Market: {market}, Selection: {selection}")
            print(f"  Change: {change*100:.1f}%, Trend: {trend}")
            print(f"  Previous odds: {prev_odds:.2f}, Current odds: {curr_odds:.2f}")
        
        print("\n--- Market Implications ---")
        for implication in significant_movements.get("implications", []):
            print(f"{implication['type']} ({implication['strength']}): {implication['description']}")
    else:
        print("No significant movements detected")
    
    # Demonstrate market calibration
    print("\n--- Market Calibration Demo ---")
    
    # Get base prediction
    base_prediction = make_global_prediction(fixture_id)
    
    # Apply market calibration
    calibrated_prediction = odds_analyzer.calibrate_prediction_with_market(base_prediction, fixture_id)
    
    # Print calibration results
    print("\nBase vs Calibrated Prediction:")
    print(f"Home win: {base_prediction['probabilities']['home_win']:.2f} → {calibrated_prediction.get('prob_home_win', 0):.2f}")
    print(f"Draw:     {base_prediction['probabilities']['draw']:.2f} → {calibrated_prediction.get('prob_draw', 0):.2f}")
    print(f"Away win: {base_prediction['probabilities']['away_win']:.2f} → {calibrated_prediction.get('prob_away_win', 0):.2f}")
    
    # Print calibration metadata
    calibration_meta = calibrated_prediction.get("market_calibration", {})
    if calibration_meta:
        print(f"\nCalibration weight: {calibration_meta.get('market_weight', 0):.2f}")
        print(f"Market confidence: {calibration_meta.get('market_confidence', 0):.2f}")
    
    print("\n=== Demo Complete ===")
    
if __name__ == "__main__":
    # Use a real fixture ID for demonstration - Premier League match
    demo_fixture_id = 1090421  # Real Madrid vs Barcelona from La Liga 
    
    try:
        demo_odds_movement(demo_fixture_id)
    except Exception as e:
        print(f"Error during demo: {e}")
