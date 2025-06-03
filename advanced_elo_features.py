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
        
        # Calculate form factor based on last 5 matches (or fewer if not available)
        if not recent_results:
            return base_rating
            
        # Points for each result: Win = 1, Draw = 0.5, Loss = 0
        result_values = {'W': 1.0, 'D': 0.5, 'L': 0.0}
        
        # Calculate decreasing weights for older results (most recent has highest weight)
        num_results = len(recent_results)
        weights = [0.6, 0.3, 0.1]  # Weights for last 3 matches
        weights = weights[:num_results] if num_results < len(weights) else weights
        
        # Fill with equal weights if more than 3 matches
        if num_results > len(weights):
            remaining_weight = 1.0 - sum(weights)
            equal_weight = remaining_weight / (num_results - len(weights))
            weights.extend([equal_weight] * (num_results - len(weights)))
            
        # Normalize weights to sum to 1
        weights = [w / sum(weights) for w in weights]
        
        # Calculate weighted form score
        form_score = sum(result_values.get(result, 0) * weight 
                         for result, weight in zip(recent_results, weights))
        
        # Adjust rating: neutral form = 0.5, gives no adjustment
        form_adjustment = (form_score - 0.5) * 200  # Scale to ±100 ELO points
        
        # Apply the adjustment with the form factor weight
        adjusted_rating = base_rating + (form_adjustment * self.form_factor_weight)
        
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
            'confidence': 0.6
        }
        
        # Get enhanced prediction
        enhanced = enrich_prediction_with_contextual_data(
            base_prediction,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            league_id=league_id
        )
        
        # Sample recent form (would come from database in real scenario)
        home_recent_form = ['W', 'W', 'D', 'L', 'W']
        away_recent_form = ['D', 'W', 'L', 'W', 'D']
        
        # Get form-adjusted ratings
        home_adjusted = self.get_form_adjusted_rating(home_team_id, home_recent_form)
        away_adjusted = self.get_form_adjusted_rating(away_team_id, away_recent_form)
        
        # Create detailed analysis
        analysis = {
            'base_elo': {
                'home': elo_data['home_elo'],
                'away': elo_data['away_elo'],
                'diff': elo_data['elo_diff']
            },
            'form_adjusted_elo': {
                'home': home_adjusted,
                'away': away_adjusted,
                'diff': home_adjusted - away_adjusted
            },
            'win_probabilities': {
                'base_elo': {
                    'home': elo_data['elo_win_probability'],
                    'draw': elo_data['elo_draw_probability'],
                    'away': elo_data['elo_loss_probability']
                },
                'statistical_model': {
                    'home': base_prediction['prob_1'],
                    'draw': base_prediction['prob_X'],
                    'away': base_prediction['prob_2']
                },
                'blended': enhanced.get('blended_probabilities', {})
            },
            'expected_goals': {
                'statistical': {
                    'home': base_prediction['predicted_home_goals'],
                    'away': base_prediction['predicted_away_goals'],
                    'diff': base_prediction['predicted_home_goals'] - base_prediction['predicted_away_goals']
                },
                'elo_based': {
                    'goal_diff': elo_data.get('elo_expected_goal_diff', 0),
                }
            },
            'match_insights': enhanced.get('elo_insights', {}),
            'competitiveness': enhanced.get('elo_enhanced_metrics', {}).get('competitiveness_rating', 5)
        }
        
        # Calculate upset potential
        favored_team = "home" if analysis['base_elo']['diff'] > 0 else "away"
        underdog_team = "away" if favored_team == "home" else "home"
        elo_diff_abs = abs(analysis['base_elo']['diff'])
        form_diff = analysis['form_adjusted_elo']['diff']
        
        # If form adjustment narrows the gap significantly, there's upset potential
        if (favored_team == "home" and form_diff < elo_diff_abs * 0.7) or \
           (favored_team == "away" and form_diff > -elo_diff_abs * 0.7):
            analysis['upset_potential'] = {
                'rating': 'High',
                'reason': f"Recent form significantly narrows the gap between teams"
            }
        elif elo_diff_abs < 100:
            analysis['upset_potential'] = {
                'rating': 'Moderate',
                'reason': f"Teams are relatively evenly matched"
            }
        else:
            analysis['upset_potential'] = {
                'rating': 'Low',
                'reason': f"Significant gap between teams unlikely to be overcome"
            }
            
        return analysis
    
    def project_tournament_performance(self, team_ids: List[int], 
                                      tournament_id: int) -> Dict[str, Any]:
        """
        Project team performance in a tournament based on ELO ratings
        
        Args:
            team_ids: List of team IDs in the tournament
            tournament_id: Tournament/League ID
            
        Returns:
            Tournament performance projection
        """
        if not team_ids:
            return {"error": "No teams provided"}
        
        # Get ratings for all teams
        ratings = {}
        for team_id in team_ids:
            ratings[team_id] = self.elo_system.get_rating(team_id)
        
        # Sort teams by rating
        sorted_teams = sorted(ratings.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate win probability matrix (each team vs each other team)
        num_teams = len(team_ids)
        win_matrix = np.zeros((num_teams, num_teams))
        
        for i, (team1_id, _) in enumerate(sorted_teams):
            for j, (team2_id, _) in enumerate(sorted_teams):
                if i != j:
                    # Get match data with team1 as home (simplification)
                    match_data = get_elo_ratings_for_match(team1_id, team2_id, tournament_id)
                    win_matrix[i, j] = match_data['elo_win_probability']
        
        # Calculate expected points (simple model: win=3, draw=1, loss=0)
        expected_points = []
        for i, (team_id, rating) in enumerate(sorted_teams):
            # Expected points against all other teams (home and away)
            pts = 0
            for j in range(num_teams):
                if i != j:
                    # Home game
                    home_win_prob = win_matrix[i, j]
                    home_draw_prob = get_elo_ratings_for_match(team_id, sorted_teams[j][0], 
                                                             tournament_id)['elo_draw_probability']
                    pts += (3 * home_win_prob + home_draw_prob)
                    
                    # Away game
                    away_win_prob = 1 - win_matrix[j, i] - \
                                  get_elo_ratings_for_match(sorted_teams[j][0], team_id, 
                                                          tournament_id)['elo_draw_probability']
                    away_draw_prob = get_elo_ratings_for_match(sorted_teams[j][0], team_id, 
                                                             tournament_id)['elo_draw_probability']
                    pts += (3 * away_win_prob + away_draw_prob)
            
            expected_points.append((team_id, pts / 2))  # Average of home/away
        
        # Sort by expected points
        expected_points.sort(key=lambda x: x[1], reverse=True)
        
        # Format results
        team_projections = []
        for rank, (team_id, points) in enumerate(expected_points, 1):
            team_projections.append({
                'rank': rank,
                'team_id': team_id,
                'elo_rating': ratings[team_id],
                'expected_points': round(points, 1),
                'win_probability': round(sum(win_matrix[list(ratings.keys()).index(team_id)]) / (num_teams - 1), 3),
                'title_chance': self._calculate_title_chance(rank, num_teams)
            })
        
        return {
            'tournament_id': tournament_id,
            'teams_analyzed': num_teams,
            'projections': team_projections
        }
    
    def _calculate_title_chance(self, rank: int, total_teams: int) -> float:
        """Calculate simplified title chance based on rank"""
        if rank == 1:
            return 0.4
        elif rank == 2:
            return 0.25
        elif rank == 3:
            return 0.15
        elif rank <= total_teams // 4:
            return 0.05
        else:
            return max(0.01, 0.15 / (rank - 3))
    
    def visualize_elo_comparison(self, teams_data: Sequence[Tuple[int, str, Union[float, int]]]) -> None:
        """
        Visualize ELO ratings for multiple teams
        
        Args:
            teams_data: Sequence of tuples (team_id, team_name, elo_rating)
        """
        team_names = [team[1] for team in teams_data]
        ratings = [float(team[2]) for team in teams_data]
        
        plt.figure(figsize=(10, 6))
        plt.barh(team_names, ratings, color='cornflowerblue')
        plt.axvline(x=1500, color='red', linestyle='--', label='Average (1500)')
        plt.xlabel('ELO Rating')
        plt.title('Team ELO Rating Comparison')
        plt.tight_layout()
        plt.savefig('elo_comparison.png')
        plt.close()

def demonstrate_advanced_elo():
    """Run a demonstration of advanced ELO features"""
    analytics = AdvancedEloAnalytics()
    
    print("=" * 80)
    print("ADVANCED ELO ANALYTICS DEMONSTRATION")
    print("=" * 80)
    
    # 1. Team matchup analysis
    print("\n1. COMPREHENSIVE TEAM MATCHUP ANALYSIS")
    print("-" * 50)
    # Premier League teams: Man City vs Liverpool
    matchup = analytics.analyze_team_matchup(39, 40, 39)
    pprint(matchup)
    
    # 2. Tournament projection
    print("\n\n2. TOURNAMENT PERFORMANCE PROJECTION")
    print("-" * 50)
    # Sample Premier League teams
    premier_teams = [33, 39, 40, 42, 47, 48, 49, 50, 51, 52]  
    projection = analytics.project_tournament_performance(premier_teams, 39)
    pprint(projection)
    
    # 3. Form-adjusted ratings demonstration
    print("\n\n3. FORM-ADJUSTED RATINGS DEMONSTRATION")
    print("-" * 50)
    team_id = 39  # Man City
    
    # Different form scenarios
    great_form = ['W', 'W', 'W', 'W', 'D']
    avg_form = ['W', 'D', 'L', 'W', 'D']
    poor_form = ['L', 'L', 'D', 'L', 'W']
    
    base = analytics.elo_system.get_rating(team_id)
    great = analytics.get_form_adjusted_rating(team_id, great_form)
    avg = analytics.get_form_adjusted_rating(team_id, avg_form)
    poor = analytics.get_form_adjusted_rating(team_id, poor_form)
    
    print(f"Team ID: {team_id}")
    print(f"Base ELO rating: {base}")
    print(f"Form-adjusted ratings:")
    print(f"  - Great form {great_form}: {great} ({great-base:+.1f})")
    print(f"  - Average form {avg_form}: {avg} ({avg-base:+.1f})")
    print(f"  - Poor form {poor_form}: {poor} ({poor-base:+.1f})")
    
    # Visualize some team comparisons
    print("\n\n4. ELO RATING VISUALIZATION")
    print("-" * 50)
    teams = [
        (39, "Man City", 1900),
        (40, "Liverpool", 1850),
        (33, "Manchester United", 1780),
        (47, "Tottenham", 1760),
        (42, "Arsenal", 1790),
        (49, "Chelsea", 1770)
    ]
    analytics.visualize_elo_comparison(teams)
    print("ELO rating comparison saved as 'elo_comparison.png'")
    
if __name__ == "__main__":
    demonstrate_advanced_elo()
