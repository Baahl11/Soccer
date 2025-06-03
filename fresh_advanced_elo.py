"""
Advanced ELO Rating Features Demo

This script demonstrates advanced ELO rating features including:
1. Form-adjusted ELO ratings
2. Comprehensive team matchup analysis
3. Tournament/competition performance projection
"""

import logging
import json
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple, Union, Sequence
from team_elo_rating import get_elo_ratings_for_match, TeamEloRating
from prediction_integration import enrich_prediction_with_contextual_data

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedEloAnalytics:
    """Class to demonstrate advanced ELO analytics features"""
    
    def __init__(self):
        """Initialize the advanced ELO analytics system"""
        self.elo_system = TeamEloRating()
        self.form_factor_weight = 0.25  # Weight for recent form in adjusting ratings
    
    def get_form_adjusted_rating(self, team_id: int, recent_results: List[str]) -> float:
        """
        Calculate a form-adjusted ELO rating based on recent results
        
        Args:
            team_id: The ID of the team
            recent_results: List of recent results (W, D, L)
            
        Returns:
            Form-adjusted ELO rating
        """
        base_rating = self.elo_system.get_rating(team_id)
        
        # Calculate form factor based on recent results
        form_factor = 0
        for result in recent_results:
            if result == 'W':
                form_factor += 1
            elif result == 'L':
                form_factor -= 1
                
        # Adjust rating based on form factor
        adjusted_rating = base_rating + (form_factor * self.form_factor_weight * 10)
        logger.info(f"Team {team_id} base rating: {base_rating}, form adjusted: {adjusted_rating}")
        return adjusted_rating
    
    def analyze_team_matchup(self, home_team_id: int, away_team_id: int, league_id: int) -> Dict[str, Any]:
        """
        Provide a comprehensive ELO-based match analysis for two teams
        
        Args:
            home_team_id: Home team ID
            away_team_id: Away team ID
            league_id: League ID
            
        Returns:
            Detailed matchup analysis
        """
        # Get base ELO data
        elo_data = get_elo_ratings_for_match(home_team_id, away_team_id, league_id)
        
        # Create a sample prediction to enhance
        base_prediction = {
            'predicted_home_goals': 1.5,
            'predicted_away_goals': 1.3,
            'total_goals': 2.8,
            'prob_over_2_5': 0.6,
            'prob_btts': 0.65,
            'prob_1': 0.45,
            'prob_X': 0.28,
            'prob_2': 0.27,
        }
        
        # Apply contextual enhancement
        enhanced = enrich_prediction_with_contextual_data(
            base_prediction, 
            home_team_id=home_team_id, 
            away_team_id=away_team_id
        )
        
        # Calculate form-adjusted ELO ratings
        home_recent_results = ['W', 'D', 'L', 'W', 'D']  # Would normally fetch this
        away_recent_results = ['L', 'W', 'W', 'D', 'L']  # Would normally fetch this
        
        home_adjusted = self.get_form_adjusted_rating(home_team_id, home_recent_results)
        away_adjusted = self.get_form_adjusted_rating(away_team_id, away_recent_results)
        
        # Calculate upset potential
        raw_elo_diff = elo_data['home_elo'] - elo_data['away_elo']
        adjusted_elo_diff = home_adjusted - away_adjusted
        
        # If adjusted difference is significantly lower than raw, upset potential is higher
        upset_factor = max(0, (raw_elo_diff - adjusted_elo_diff) / 50)
        
        if raw_elo_diff > 0 and upset_factor > 0.5:
            upset_type = "Away team upset potential"
            upset_potential = upset_factor * 10  # Scale to 0-10
        elif raw_elo_diff < 0 and upset_factor > 0.5:
            upset_type = "Home team upset potential"
            upset_potential = upset_factor * 10
        else:
            upset_type = "Low upset potential"
            upset_potential = upset_factor * 5
            
        analysis = {
            'raw_elo': {
                'home': elo_data['home_elo'],
                'away': elo_data['away_elo'],
                'difference': raw_elo_diff
            },
            'form_adjusted_elo': {
                'home': home_adjusted,
                'away': away_adjusted,
                'difference': adjusted_elo_diff
            },
            'upset_potential': {
                'value': upset_potential,
                'description': upset_type
            },
            'enhanced_prediction': enhanced
        }
        
        return analysis
    
    def visualize_elo_comparison(self, teams_data: Sequence[Tuple[int, str, Union[float, int]]]) -> None:
        """
        Create a bar chart comparing ELO ratings of specified teams
        
        Args:
            teams_data: List of tuples with (team_id, team_name, optional_rating)
                        If optional_rating is not provided, it will be fetched from the ELO system
        """
        team_names = [team[1] for team in teams_data]
        
        # Get ratings - either from the provided data or from the ELO system
        ratings = []
        for team in teams_data:
            if len(team) >= 3:
                ratings.append(float(team[2]))
            else:
                ratings.append(self.elo_system.get_rating(team[0]))
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        bars = plt.bar(team_names, ratings, color='skyblue')
        
        # Add the values on top of the bars
        for bar, rating in zip(bars, ratings):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f'{rating:.0f}',
                ha='center',
                fontsize=9
            )
        
        plt.title('Team ELO Rating Comparison')
        plt.xlabel('Teams')
        plt.ylabel('ELO Rating')
        plt.ylim(min(ratings) - 100, max(ratings) + 100)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('elo_comparison.png')
        plt.close()


def demonstrate_advanced_elo():
    """Demonstrate the advanced ELO features"""
    analytics = AdvancedEloAnalytics()
    
    # Demonstrate form-adjusted rating
    team_id = 42
    recent_results = ['W', 'W', 'D', 'L', 'W']
    adjusted_rating = analytics.get_form_adjusted_rating(team_id, recent_results)
    print(f"Form-adjusted rating for team {team_id}: {adjusted_rating}")
    
    # Demonstrate team matchup analysis
    matchup = analytics.analyze_team_matchup(39, 40, 39)
    print("\nTeam Matchup Analysis:")
    pprint(matchup)
    
    # Demonstrate ratings comparison visualization
    teams = [
        (42, "Liverpool", 1850),
        (56, "Manchester City", 1820),
        (61, "Bayern Munich", 1830),
        (72, "Real Madrid", 1840),
        (81, "Barcelona", 1810),
        (49, "Chelsea", 1770)
    ]
    analytics.visualize_elo_comparison(teams)
    print("ELO rating comparison saved as 'elo_comparison.png'")
    
if __name__ == "__main__":
    demonstrate_advanced_elo()