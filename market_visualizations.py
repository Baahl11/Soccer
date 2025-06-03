"""
Market visualizations for soccer prediction system.

This module provides functions to visualize market trends and movements,
helping to understand how the market evolves and impacts predictions.
"""

import matplotlib.pyplot as plt
import matplotlib.dates
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple, Optional
import logging
import os
from market_integration import MarketDataIntegrator
from odds_analyzer import OddsAnalyzer

logger = logging.getLogger(__name__)

# Ensure the plots directory exists
PLOTS_DIR = "plots"
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def plot_odds_evolution(fixture_id: int, save_path: Optional[str] = None) -> str:
    """
    Plot the evolution of odds for a specific fixture over time.
    
    Args:
        fixture_id: ID of the fixture to visualize
        save_path: Optional path to save the visualization (if None, auto-generated)
        
    Returns:
        Path where the visualization was saved
    """
    try:
        # Get historical odds data
        analyzer = OddsAnalyzer()
        history = analyzer.movement_tracker.get_historical_odds(fixture_id)
        
        if not history or len(history) < 2:
            logger.warning(f"Insufficient historical data for fixture {fixture_id}")
            return ""
        
        # Set up figure
        plt.figure(figsize=(12, 8))
          # Extract timestamps and convert to datetime
        timestamps = [datetime.fromisoformat(entry.get("timestamp", "")) if entry.get("timestamp") else datetime.now() for entry in history]
        
        # Prepare data for plotting
        home_odds = []
        draw_odds = []
        away_odds = []
        over_odds = []
        under_odds = []
        
        # Extract key market data from history
        for entry in history:
            odds_data = entry.get("odds", {})
            match_winner = odds_data.get("Match Winner", {})
            
            # Extract 1X2 odds
            if match_winner:
                home_odds.append(match_winner.get("home", {}).get("odds", None))
                draw_odds.append(match_winner.get("draw", {}).get("odds", None))
                away_odds.append(match_winner.get("away", {}).get("odds", None))
            
            # Extract Over/Under odds
            ou_data = odds_data.get("Over/Under", {})
            if ou_data:
                for outcome, data in ou_data.items():
                    if "over 2.5" in outcome.lower():
                        over_odds.append(data.get("odds", None))
                    elif "under 2.5" in outcome.lower():
                        under_odds.append(data.get("odds", None))
        
        # Plot 1X2 odds evolution
        plt.subplot(2, 1, 1)
        if home_odds:
            plt.plot(timestamps, home_odds, 'b-', label='Home')
        if draw_odds:
            plt.plot(timestamps, draw_odds, 'g-', label='Draw')
        if away_odds:
            plt.plot(timestamps, away_odds, 'r-', label='Away')
        
        plt.title(f'1X2 Odds Evolution for Fixture {fixture_id}')
        plt.ylabel('Odds')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Plot Over/Under odds evolution
        plt.subplot(2, 1, 2)
        if over_odds:
            plt.plot(timestamps, over_odds, 'b-', label='Over 2.5')
        if under_odds:
            plt.plot(timestamps, under_odds, 'r-', label='Under 2.5')
        
        plt.title('Over/Under 2.5 Odds Evolution')
        plt.ylabel('Odds')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save the figure
        if not save_path:
            save_path = os.path.join(PLOTS_DIR, f"odds_evolution_{fixture_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path
        
    except Exception as e:
        logger.error(f"Error creating odds evolution plot: {e}")
        return ""

def plot_implied_probability_trends(fixture_id: int, save_path: Optional[str] = None) -> str:
    """
    Plot the trends of implied probabilities derived from odds.
    
    Args:
        fixture_id: ID of the fixture to visualize
        save_path: Optional path to save the visualization
        
    Returns:
        Path where the visualization was saved
    """
    try:
        # Get historical odds data
        analyzer = OddsAnalyzer()
        history = analyzer.movement_tracker.get_historical_odds(fixture_id)
        
        if not history or len(history) < 2:
            logger.warning(f"Insufficient historical data for fixture {fixture_id}")
            return ""
        
        # Set up figure
        plt.figure(figsize=(12, 8))
          # Extract timestamps
        timestamps = [datetime.fromisoformat(entry.get("timestamp", "")) if entry.get("timestamp") else datetime.now() for entry in history]
        
        # Arrays to store implied probabilities over time
        home_probs = []
        draw_probs = []
        away_probs = []
        
        # Calculate implied probabilities from odds
        for entry in history:
            odds_data = entry.get("odds", {})
            match_winner = odds_data.get("Match Winner", {})
            
            if match_winner and all(k in match_winner for k in ["home", "draw", "away"]):
                # Get odds
                home_odd = match_winner.get("home", {}).get("odds", 0)
                draw_odd = match_winner.get("draw", {}).get("odds", 0)
                away_odd = match_winner.get("away", {}).get("odds", 0)
                
                if home_odd and draw_odd and away_odd:
                    # Calculate raw implied probabilities
                    raw_home_prob = 1 / home_odd
                    raw_draw_prob = 1 / draw_odd
                    raw_away_prob = 1 / away_odd
                    
                    # Normalize to account for overround
                    total = raw_home_prob + raw_draw_prob + raw_away_prob
                    home_probs.append(raw_home_prob / total)
                    draw_probs.append(raw_draw_prob / total)
                    away_probs.append(raw_away_prob / total)
        
        # Plot implied probability trends
        if home_probs:
            plt.plot(timestamps, home_probs, 'b-', linewidth=2, label='Home Win')
        if draw_probs:
            plt.plot(timestamps, draw_probs, 'g-', linewidth=2, label='Draw')
        if away_probs:
            plt.plot(timestamps, away_probs, 'r-', linewidth=2, label='Away Win')
        
        plt.title(f'Market Implied Probability Trends for Fixture {fixture_id}')
        plt.ylabel('Implied Probability')
        plt.xlabel('Time')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
          # Add annotations for the latest values
        if home_probs:
            timestamp_val = matplotlib.dates.date2num(timestamps[-1])
            plt.annotate(f'{home_probs[-1]:.2f}', 
                        xy=(timestamp_val, home_probs[-1]),
                        xytext=(5, 0), 
                        textcoords='offset points')
        
        if draw_probs:
            timestamp_val = matplotlib.dates.date2num(timestamps[-1])
            plt.annotate(f'{draw_probs[-1]:.2f}', 
                        xy=(timestamp_val, draw_probs[-1]),
                        xytext=(5, 0), 
                        textcoords='offset points')
        
        if away_probs:
            timestamp_val = matplotlib.dates.date2num(timestamps[-1])
            plt.annotate(f'{away_probs[-1]:.2f}', 
                        xy=(timestamp_val, away_probs[-1]),
                        xytext=(5, 0), 
                        textcoords='offset points')
        
        # Save the figure
        if not save_path:
            save_path = os.path.join(PLOTS_DIR, f"implied_probs_{fixture_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path
        
    except Exception as e:
        logger.error(f"Error creating implied probability plot: {e}")
        return ""

def plot_market_movement_analysis(fixture_id: int, save_path: Optional[str] = None) -> str:
    """
    Plot analysis of significant market movements.
    
    Args:
        fixture_id: ID of the fixture to visualize
        save_path: Optional path to save the visualization
        
    Returns:
        Path where the visualization was saved
    """
    try:
        # Get market movements
        integrator = MarketDataIntegrator()
        movement_analysis = integrator.analyze_market_movements(fixture_id)
        
        if not movement_analysis.get("significant_movements", False):
            logger.info(f"No significant movements detected for fixture {fixture_id}")
            return ""
        
        # Set up figure
        plt.figure(figsize=(12, 10))
        
        # Extract movement data
        movements = movement_analysis.get("movements", [])
        
        # Group by market
        markets = {}
        for move in movements:
            market_name = move.get("market", "Unknown")
            if market_name not in markets:
                markets[market_name] = []
            markets[market_name].append(move)
        
        # Plot each market in a separate subplot
        num_markets = len(markets)
        if num_markets == 0:
            logger.warning("No market data available to plot")
            return ""
        
        idx = 1
        for market_name, moves in markets.items():
            plt.subplot(num_markets, 1, idx)
            
            # Prepare data for this market
            selections = []
            changes = []
            colors = []
            
            for move in moves:
                selection = move.get("selection", "Unknown")
                change = move.get("change", 0)
                trend = move.get("trend", "stable")
                
                selections.append(selection)
                changes.append(abs(change) * 100)  # Convert to percentage
                # Green for decreasing odds (increasing probability), Red for increasing odds
                colors.append('green' if trend == 'decreasing' else 'red')
            
            # Create horizontal bar chart
            y_pos = np.arange(len(selections))
            plt.barh(y_pos, changes, color=colors)
            plt.yticks(y_pos, selections)
            plt.xlabel('% Change in Odds')
            plt.title(f'{market_name}')
            
            # Add value annotations
            for i, v in enumerate(changes):
                plt.text(v + 0.1, i, f"{v:.1f}%", va='center')
            
            idx += 1
        
        plt.suptitle(f'Significant Market Movements for Fixture {fixture_id}')
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate the title
        
        # Save the figure
        if not save_path:
            save_path = os.path.join(PLOTS_DIR, f"market_movements_{fixture_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path
        
    except Exception as e:
        logger.error(f"Error creating market movement plot: {e}")
        return ""

def plot_model_vs_market(fixture_id: int, prediction: Dict[str, Any], save_path: Optional[str] = None) -> str:
    """
    Plot comparison between model prediction and market implied probabilities.
    
    Args:
        fixture_id: ID of the fixture
        prediction: Prediction dictionary with probabilities
        save_path: Optional path to save the visualization
        
    Returns:
        Path where the visualization was saved
    """
    try:
        # Get market data
        integrator = MarketDataIntegrator()
        market_features = integrator.extract_odds_features(fixture_id)
        
        # Check if we have implied probabilities
        if not all(k in market_features for k in ["implied_prob_home", "implied_prob_draw", "implied_prob_away"]):
            logger.warning(f"Missing market implied probabilities for fixture {fixture_id}")
            return ""
        
        # Check if prediction has required probabilities
        if not all(k in prediction for k in ["prob_home_win", "prob_draw", "prob_away_win"]):
            logger.warning(f"Missing prediction probabilities for fixture {fixture_id}")
            return ""
        
        # Set up figure
        plt.figure(figsize=(10, 6))
        
        # Data for plotting
        outcomes = ['Home Win', 'Draw', 'Away Win']
        model_probs = [
            prediction.get("prob_home_win", 0), 
            prediction.get("prob_draw", 0), 
            prediction.get("prob_away_win", 0)
        ]
        market_probs = [
            market_features.get("implied_prob_home", 0),
            market_features.get("implied_prob_draw", 0),
            market_features.get("implied_prob_away", 0)
        ]
        
        # Calculate differences
        diffs = [model - market for model, market in zip(model_probs, market_probs)]
        
        # Create grouped bar chart
        x = np.arange(len(outcomes))
        width = 0.35
        
        plt.bar(x - width/2, model_probs, width, label='Model Prediction', color='blue', alpha=0.7)
        plt.bar(x + width/2, market_probs, width, label='Market Implied', color='green', alpha=0.7)
        
        # Add labels and title
        plt.xlabel('Outcome')
        plt.ylabel('Probability')
        plt.title(f'Model vs Market: Fixture {fixture_id} ({prediction.get("home_team", "Home")} vs {prediction.get("away_team", "Away")})')
        plt.xticks(x, outcomes)
        plt.ylim(0, 1)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value annotations
        for i, (model_p, market_p, diff) in enumerate(zip(model_probs, market_probs, diffs)):
            plt.text(i - width/2, model_p + 0.02, f"{model_p:.2f}", ha='center')
            plt.text(i + width/2, market_p + 0.02, f"{market_p:.2f}", ha='center')
            
            # Add difference annotation in the middle
            color = 'green' if diff > 0 else 'red'
            plt.text(i, min(model_p, market_p) - 0.07, f"Diff: {diff:+.2f}", ha='center', color=color, fontweight='bold')
        
        # Save the figure
        if not save_path:
            save_path = os.path.join(PLOTS_DIR, f"model_vs_market_{fixture_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        
        plt.savefig(save_path)
        plt.close()
        
        return save_path
        
    except Exception as e:
        logger.error(f"Error creating model vs market plot: {e}")
        return ""

def create_visualizations_for_fixture(fixture_id: int) -> Dict[str, str]:
    """
    Create all market visualizations for a specific fixture.
    
    Args:
        fixture_id: ID of the fixture
        
    Returns:
        Dictionary with paths to generated visualizations
    """
    result = {}
    
    try:
        # Generate odds evolution plot
        odds_evolution_path = plot_odds_evolution(fixture_id)
        if odds_evolution_path:
            result["odds_evolution"] = odds_evolution_path
        
        # Generate implied probability trends
        implied_probs_path = plot_implied_probability_trends(fixture_id)
        if implied_probs_path:
            result["implied_probabilities"] = implied_probs_path
        
        # Generate market movement analysis
        movement_path = plot_market_movement_analysis(fixture_id)
        if movement_path:
            result["market_movements"] = movement_path        # Get prediction and generate model vs market
        from predictions import make_global_prediction
        from data import get_fixture_data
        
        try:
            # Get the prediction using the fixture ID
            prediction = make_global_prediction(fixture_id)
            if prediction:
                model_vs_market_path = plot_model_vs_market(fixture_id, prediction)
                if model_vs_market_path:
                    result["model_vs_market"] = model_vs_market_path
        except Exception as e:
            logger.error(f"Error getting prediction for fixture {fixture_id}: {e}")
    
    except Exception as e:
        logger.error(f"Error creating visualizations for fixture {fixture_id}: {e}")
    
    return result
