"""
Demo script for using the corner prediction model.

Shows how to use the VotingEnsembleCornersModel to make predictions
for upcoming matches.
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime
from voting_ensemble_corners import VotingEnsembleCornersModel

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def predict_for_match(home_team_id, away_team_id, home_stats, away_stats, league_id):
    """Make predictions for a specific match"""
    
    # Initialize the model
    model = VotingEnsembleCornersModel()
    
    # Make prediction
    prediction = model.predict_corners(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        home_stats=home_stats,
        away_stats=away_stats,
        league_id=league_id
    )
    
    return prediction

def main():
    """Main function to demonstrate model usage"""
    print("\n===== CORNER PREDICTION DEMO =====")
    
    # Sample match data - Manchester United vs Liverpool
    home_team_id = 33  # Manchester United
    away_team_id = 40  # Liverpool
    league_id = 39     # Premier League
    
    # Sample team stats (would typically come from API or database)
    home_stats = {
        'avg_corners_for': 5.8,        # Average corners earned by home team
        'avg_corners_against': 4.2,    # Average corners conceded by home team
        'form_score': 65,              # Form rating out of 100
        'possession': 52,              # Average possession percentage
        'shots_on_goal': 5.2,          # Average shots on target per game
        'attack_strength': 1.12        # Attack strength rating (>1 is above average)
    }
    
    away_stats = {
        'avg_corners_for': 6.2,        # Average corners earned by away team
        'avg_corners_against': 3.8,    # Average corners conceded by away team
        'form_score': 72,              # Form rating out of 100
        'possession': 58,              # Average possession percentage
        'shots_on_goal': 6.1,          # Average shots on target per game
        'attack_strength': 1.24        # Attack strength rating (>1 is above average)
    }
    
    # Make the prediction
    prediction = predict_for_match(home_team_id, away_team_id, home_stats, away_stats, league_id)
    
    # Display results
    print("\n===== MATCH INFORMATION =====")
    print("Home Team: Manchester United")
    print("Away Team: Liverpool")
    print("League: Premier League")
    
    print("\n===== PREDICTION RESULTS =====")
    print(f"Total Corners: {prediction['total']:.1f}")
    print(f"Home Team Corners: {prediction['home']:.1f}")
    print(f"Away Team Corners: {prediction['away']:.1f}")
    
    print("\n===== OVER/UNDER PROBABILITIES =====")
    print(f"Over 8.5 corners: {prediction.get('over_8.5', 0) * 100:.1f}%")
    print(f"Over 9.5 corners: {prediction.get('over_9.5', 0) * 100:.1f}%")
    print(f"Over 10.5 corners: {prediction.get('over_10.5', 0) * 100:.1f}%")
    print(f"Over 11.5 corners: {prediction.get('over_11.5', 0) * 100:.1f}%")
    
    if 'confidence' in prediction:
        print(f"\nPrediction Confidence: {prediction['confidence'] * 100:.1f}%")
    
    print("\n===== HOW TO USE THIS MODEL =====")
    print("1. Collect pre-match statistics from API or database")
    print("2. Initialize the model: model = VotingEnsembleCornersModel()")
    print("3. Call model.predict_corners() with team IDs and statistics")
    print("4. Use the predictions for your betting strategies")

if __name__ == "__main__":
    main()
