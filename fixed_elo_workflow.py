"""
ELO Enhanced Prediction Integration Workflow

This script demonstrates a complete workflow for using the ELO enhanced prediction system
in a real-world scenario, including data gathering, prediction, and result visualization.
"""

import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import os
import math
from pathlib import Path

# Import prediction modules
from prediction_integration import make_integrated_prediction, enrich_prediction_with_contextual_data
from team_elo_rating import get_elo_ratings_for_match, TeamEloRating
from fresh_advanced_elo import AdvancedEloAnalytics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create output directory for visualizations
RESULTS_DIR = Path("elo_prediction_results")
RESULTS_DIR.mkdir(exist_ok=True)

class ELOEnhancedPredictionWorkflow:
    """Workflow class for ELO-enhanced predictions"""
    
    def __init__(self, data_cache_dir: str = "elo_cache"):
        """Initialize the workflow"""
        self.data_cache_dir = data_cache_dir
        Path(data_cache_dir).mkdir(exist_ok=True)
        
        # Initialize advanced ELO analytics
        self.elo_analytics = AdvancedEloAnalytics()
        
        # Initialize a simple match database
        self.match_database = []
    
    def get_upcoming_matches(self, league_id: int, days_ahead: int = 7) -> List[Dict[str, Any]]:
        """
        Get upcoming matches for a league (simulated)
        
        Args:
            league_id: League ID
            days_ahead: Number of days ahead to look for matches
            
        Returns:
            List of upcoming match data
        """
        # NOTE: In a real system, this would pull data from an API or database
        # This is a mockup with sample data
        
        # Map league IDs to names
        league_names = {
            39: "Premier League",
            140: "La Liga",
            135: "Serie A",
            78: "Bundesliga",
            61: "Ligue 1",
            2: "UEFA Champions League",
            3: "UEFA Europa League",
            848: "UEFA Conference League"
        }
        
        # Sample teams for each league
        league_teams = {
            39: [  # Premier League
                (33, "Manchester United"),
                (39, "Manchester City"),
                (40, "Liverpool"),
                (42, "Arsenal"),
                (47, "Tottenham"),
                (48, "West Ham"),
                (49, "Chelsea"),
                (50, "Wolverhampton"),
                (51, "Brighton"),
                (52, "Leicester")
            ],
            140: [  # La Liga
                (529, "Barcelona"),
                (530, "Atletico Madrid"),
                (541, "Real Madrid"),
                (532, "Valencia"),
                (543, "Real Betis")
            ]
        }
        
        # Default to Premier League if league not found
        teams = league_teams.get(league_id, league_teams[39])
        league_name = league_names.get(league_id, "Unknown League")
        
        # Generate sample fixtures
        fixtures = []
        today = datetime.now()
        
        for i in range(days_ahead):
            match_date = today + timedelta(days=i+1)
            # Create 2-3 matches per day
            for _ in range(min(len(teams) // 2, 3)):
                # Pick random home and away teams (different teams)
                available_teams = teams.copy()
                home_idx = np.random.randint(0, len(available_teams))
                home_team = available_teams.pop(home_idx)
                
                away_idx = np.random.randint(0, len(available_teams))
                away_team = available_teams.pop(away_idx)
                
                # Create fixture
                fixture = {
                    'fixture_id': 1000000 + len(fixtures),
                    'league_id': league_id,
                    'league_name': league_name,
                    'match_date': match_date.strftime("%Y-%m-%d %H:%M:%S"),
                    'home_team_id': home_team[0],
                    'home_team_name': home_team[1],
                    'away_team_id': away_team[0],
                    'away_team_name': away_team[1]
                }
                
                fixtures.append(fixture)
        
        return fixtures
    
    def make_predictions_for_matches(self, fixtures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Make predictions for a list of matches
        
        Args:
            fixtures: List of fixtures to predict
            
        Returns:
            List of predictions
        """
        predictions = []
        
        for fixture in fixtures:
            logger.info(f"Making prediction for {fixture['home_team_name']} vs {fixture['away_team_name']}")
            
            # In a real system, use make_integrated_prediction
            # Here we'll use a simplified approach with just ELO enhancement
            
            # Create a base prediction
            base_prediction = self._create_sample_prediction(
                fixture['home_team_id'], 
                fixture['away_team_id']
            )
            
            # Enhance with ELO data
            enhanced = enrich_prediction_with_contextual_data(
                base_prediction,
                home_team_id=fixture['home_team_id'],
                away_team_id=fixture['away_team_id'],
                league_id=fixture['league_id']
            )
            
            # Add fixture metadata
            enhanced.update({
                'fixture_id': fixture['fixture_id'],
                'league_id': fixture['league_id'],
                'league_name': fixture['league_name'],
                'match_date': fixture['match_date'],
                'home_team_id': fixture['home_team_id'],
                'home_team_name': fixture['home_team_name'],
                'away_team_id': fixture['away_team_id'],
                'away_team_name': fixture['away_team_name'],
                'prediction_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            # Add additional insights from advanced ELO analytics
            matchup_analysis = self.elo_analytics.analyze_team_matchup(
                fixture['home_team_id'],
                fixture['away_team_id'],
                fixture['league_id']
            )
            
            enhanced['advanced_elo_analysis'] = {
                'form_adjusted_elo': matchup_analysis['form_adjusted_elo'],
                'upset_potential': matchup_analysis['upset_potential']
            }
            
            predictions.append(enhanced)
            
            # Save to database
            self.match_database.append({
                'fixture': fixture,
                'prediction': enhanced
            })
        
        return predictions
    
    def _create_sample_prediction(self, home_team_id: int, away_team_id: int) -> Dict[str, Any]:
        """Create a sample base prediction with some randomness for demo purposes"""
        # Base values
        home_goals_base = 1.4
        away_goals_base = 1.1
        
        # Add slight randomness for demo variety
        home_goals = max(0, home_goals_base + np.random.normal(0, 0.3))
        away_goals = max(0, away_goals_base + np.random.normal(0, 0.3))
        
        # Calculate probability values
        total_goals = home_goals + away_goals
        prob_over_2_5 = 1 - self._simple_poisson_cdf(total_goals, 2.5)
        
        # BTTS probability
        prob_btts = (1 - np.exp(-home_goals)) * (1 - np.exp(-away_goals))
        
        # Simple 1X2 probabilities
        prob_home = max(0.1, min(0.8, 0.45 + np.random.normal(0, 0.1)))
        prob_draw = max(0.1, min(0.5, 0.27 + np.random.normal(0, 0.05)))
        prob_away = max(0.1, min(0.7, 1 - prob_home - prob_draw))
        
        # Normalize to ensure they sum to 1
        total = prob_home + prob_draw + prob_away
        prob_home /= total
        prob_draw /= total
        prob_away /= total
        
        # Base confidence
        confidence = max(0.4, min(0.8, 0.6 + np.random.normal(0, 0.1)))
        
        # Create prediction dictionary
        prediction = {
            'predicted_home_goals': home_goals,
            'predicted_away_goals': away_goals,
            'total_goals': total_goals,
            'prob_over_2_5': prob_over_2_5,
            'prob_btts': prob_btts,
            'prob_1': prob_home,
            'prob_X': prob_draw,
            'prob_2': prob_away,
            'confidence': confidence,
            'prediction': 'Home' if prob_home > max(prob_draw, prob_away) else 
                         'Draw' if prob_draw > max(prob_home, prob_away) else 'Away',
            'method': 'statistical'
        }
        
        return prediction
        
    def _simple_poisson_cdf(self, lambda_val: float, k: float) -> float:
        """Simple Poisson CDF calculation"""
        k_floor = int(k)
        result = 0
        for i in range(k_floor + 1):
            result += (lambda_val ** i) * np.exp(-lambda_val) / math.factorial(i)
        return result
    
    def visualize_prediction(self, prediction: Dict[str, Any], output_file: Optional[str] = None) -> None:
        """
        Create a visualization of a prediction with ELO enhancements
        
        Args:
            prediction: The prediction to visualize
            output_file: Output file path (if None, derived from match details)
        """
        # Create figure with 2x2 subplots
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f"{prediction['home_team_name']} vs {prediction['away_team_name']}\n{prediction['league_name']} - {prediction['match_date']}", 
                     fontsize=16)
        
        # 1. Win probability chart
        axs[0, 0].bar(['Home', 'Draw', 'Away'], 
                     [prediction['blended_probabilities']['home_win'], 
                      prediction['blended_probabilities']['draw'], 
                      prediction['blended_probabilities']['away_win']], 
                     color=['blue', 'gray', 'red'])
        axs[0, 0].set_title('Win Probability Distribution')
        axs[0, 0].set_ylabel('Probability')
        
        # Add text labels to bars
        for i, prob in enumerate([prediction['blended_probabilities']['home_win'], 
                                 prediction['blended_probabilities']['draw'], 
                                 prediction['blended_probabilities']['away_win']]):
            axs[0, 0].text(i, prob + 0.02, f'{prob:.0%}', ha='center')
        
        # 2. ELO Rating comparison
        rating_diff = prediction['elo_ratings']['elo_diff']
        home_elo = prediction['elo_ratings']['home_elo']
        away_elo = prediction['elo_ratings']['away_elo']
        
        # Calculate adjusted ranges for visualization
        max_elo = max(home_elo, away_elo)
        min_elo = min(home_elo, away_elo)
        padding = max(100, abs(home_elo - away_elo) * 0.5)
        
        axs[0, 1].bar(['Home', 'Away'], [home_elo, away_elo], color=['blue', 'red'])
        axs[0, 1].set_title('Team ELO Ratings')
        axs[0, 1].set_ylabel('ELO Rating')
        axs[0, 1].set_ylim([min(1400, min_elo - padding), max(1600, max_elo + padding)])
        
        # Add text labels
        axs[0, 1].text(0, home_elo + 20, f'{home_elo:.0f}', ha='center')
        axs[0, 1].text(1, away_elo + 20, f'{away_elo:.0f}', ha='center')
        axs[0, 1].text(0.5, max_elo + padding/2, 
                     f"Difference: {rating_diff:+.0f} to {'Home' if rating_diff > 0 else 'Away'}", 
                     ha='center')
        
        # 3. Expected Goals
        axs[1, 0].bar(['Home', 'Away'], 
                     [prediction['predicted_home_goals'], prediction['predicted_away_goals']], 
                     color=['blue', 'red'])
        axs[1, 0].set_title('Expected Goals')
        axs[1, 0].set_ylabel('Goals')
        
        # Add text labels
        axs[1, 0].text(0, prediction['predicted_home_goals'] + 0.1, 
                     f"{prediction['predicted_home_goals']:.1f}", ha='center')
        axs[1, 0].text(1, prediction['predicted_away_goals'] + 0.1, 
                     f"{prediction['predicted_away_goals']:.1f}", ha='center')
        
        # 4. Key insights text box
        insights_text = []
        
        # Add ELO insights
        if 'elo_insights' in prediction:
            for key, value in prediction['elo_insights'].items():
                insights_text.append(f"• {value}")
        
        # Add competitiveness rating
        if 'elo_enhanced_metrics' in prediction:
            comp_rating = prediction['elo_enhanced_metrics']['competitiveness_rating']
            insights_text.append(f"• Match Competitiveness: {comp_rating}/10")
            
            if 'margin_category' in prediction['elo_enhanced_metrics']:
                insights_text.append(f"• {prediction['elo_enhanced_metrics']['margin_category']}")
        
        # Add advanced analysis insights
        if 'advanced_elo_analysis' in prediction and 'upset_potential' in prediction['advanced_elo_analysis']:
            upset = prediction['advanced_elo_analysis']['upset_potential']
            insights_text.append(f"• Upset Potential: {upset['value']:.1f}/10")
            if len(insights_text) < 7:  # Limit number of insights
                insights_text.append(f"  {upset['description']}")
        
        axs[1, 1].axis('off')  # Turn off axis
        axs[1, 1].text(0, 0.9, "KEY INSIGHTS", fontsize=14, fontweight='bold')
        
        y_pos = 0.8
        for insight in insights_text:
            axs[1, 1].text(0, y_pos, insight, fontsize=11)
            y_pos -= 0.12
        
        # Add prediction summary and confidence
        pred_direction = prediction.get('prediction', '')
        if 'enhanced_confidence' in prediction:
            confidence = prediction['enhanced_confidence']['score']
        else:
            confidence = prediction.get('confidence', 0)
        
        confidence_text = f"Prediction: {pred_direction} Win\nConfidence: {confidence:.0%}"
        axs[1, 1].text(0, 0.1, confidence_text, fontsize=14, fontweight='bold')
        
        # Adjust layout
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save or show
        if output_file:
            plt.savefig(output_file)
        else:
            # Generate filename from match details
            home = prediction['home_team_name'].replace(' ', '_')
            away = prediction['away_team_name'].replace(' ', '_')
            date = prediction['match_date'].split(' ')[0]
            filename = f"{RESULTS_DIR}/prediction_{home}_vs_{away}_{date}.png"
            plt.savefig(filename)
            logger.info(f"Saved visualization to {filename}")
        
        plt.close()
    
    def generate_match_report(self, prediction: Dict[str, Any], output_file: Optional[str] = None) -> None:
        """
        Generate a detailed match report with ELO analysis
        
        Args:
            prediction: The prediction to report on
            output_file: Output file path (if None, derived from match details)
        """
        if not output_file:
            # Generate filename from match details
            home = prediction['home_team_name'].replace(' ', '_')
            away = prediction['away_team_name'].replace(' ', '_')
            date = prediction['match_date'].split(' ')[0]
            output_file = f"{RESULTS_DIR}/report_{home}_vs_{away}_{date}.txt"
        
        with open(output_file, 'w') as f:
            # Header
            f.write(f"{'='*80}\n")
            f.write(f"MATCH PREDICTION REPORT\n")
            f.write(f"{'='*80}\n\n")
            
            # Match details
            f.write(f"Match: {prediction['home_team_name']} vs {prediction['away_team_name']}\n")
            f.write(f"Competition: {prediction['league_name']}\n")
            f.write(f"Date: {prediction['match_date']}\n")
            f.write(f"Report generated: {prediction['prediction_timestamp']}\n\n")
            
            # ELO Ratings
            f.write(f"{'-'*80}\n")
            f.write(f"ELO RATINGS\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{prediction['home_team_name']}: {prediction['elo_ratings']['home_elo']:.0f}\n")
            f.write(f"{prediction['away_team_name']}: {prediction['elo_ratings']['away_elo']:.0f}\n")
            f.write(f"Difference: {prediction['elo_ratings']['elo_diff']:+.0f} to {'Home' if prediction['elo_ratings']['elo_diff'] > 0 else 'Away'}\n\n")
            
            # Win Probabilities
            f.write(f"{'-'*80}\n")
            f.write(f"WIN PROBABILITIES\n")
            f.write(f"{'-'*80}\n")
            f.write(f"Statistical Model:\n")
            f.write(f"  Home Win: {prediction['prob_1']:.1%}\n")
            f.write(f"  Draw: {prediction['prob_X']:.1%}\n")
            f.write(f"  Away Win: {prediction['prob_2']:.1%}\n\n")
            
            f.write(f"ELO-Based Probabilities:\n")
            f.write(f"  Home Win: {prediction['elo_probabilities']['win']:.1%}\n")
            f.write(f"  Draw: {prediction['elo_probabilities']['draw']:.1%}\n")
            f.write(f"  Away Win: {prediction['elo_probabilities']['loss']:.1%}\n\n")
            
            f.write(f"Blended Probabilities (Final):\n")
            f.write(f"  Home Win: {prediction['blended_probabilities']['home_win']:.1%}\n")
            f.write(f"  Draw: {prediction['blended_probabilities']['draw']:.1%}\n")
            f.write(f"  Away Win: {prediction['blended_probabilities']['away_win']:.1%}\n")
            f.write(f"  Blend weights: {prediction['blended_probabilities']['blend_weight']['model']:.0%} model, ")
            f.write(f"{prediction['blended_probabilities']['blend_weight']['elo']:.0%} ELO\n\n")
            
            # Expected Goals
            f.write(f"{'-'*80}\n")
            f.write(f"EXPECTED GOALS\n")
            f.write(f"{'-'*80}\n")
            f.write(f"{prediction['home_team_name']}: {prediction['predicted_home_goals']:.2f}\n")
            f.write(f"{prediction['away_team_name']}: {prediction['predicted_away_goals']:.2f}\n")
            f.write(f"Total: {prediction['total_goals']:.2f}\n")
            f.write(f"ELO-based expected goal difference: {prediction.get('elo_expected_goal_diff', 0):.2f}\n\n")
            
            # Additional Probabilities
            f.write(f"Over 2.5 goals: {prediction['prob_over_2_5']:.1%}\n")
            f.write(f"Both teams to score: {prediction['prob_btts']:.1%}\n\n")
            
            # ELO Insights and Analysis
            f.write(f"{'-'*80}\n")
            f.write(f"ELO INSIGHTS & ANALYSIS\n")
            f.write(f"{'-'*80}\n")
            
            if 'elo_insights' in prediction:
                for key, value in prediction['elo_insights'].items():
                    f.write(f"{value}\n")
            
            if 'elo_enhanced_metrics' in prediction:
                f.write(f"\nEnhanced Metrics:\n")
                f.write(f"- Competitiveness Rating: {prediction['elo_enhanced_metrics']['competitiveness_rating']}/10\n")
                f.write(f"- Expected Margin: {prediction['elo_enhanced_metrics']['expected_margin']:.2f}\n")
                f.write(f"- Margin Category: {prediction['elo_enhanced_metrics']['margin_category']}\n\n")
            
            # Advanced Analysis
            if 'advanced_elo_analysis' in prediction:
                f.write(f"{'-'*80}\n")
                f.write(f"ADVANCED ANALYSIS\n")
                f.write(f"{'-'*80}\n")
                
                if 'form_adjusted_elo' in prediction['advanced_elo_analysis']:
                    form_ratings = prediction['advanced_elo_analysis']['form_adjusted_elo']
                    f.write(f"Form-Adjusted Ratings:\n")
                    f.write(f"  {prediction['home_team_name']}: {form_ratings['home']:.0f}\n")
                    f.write(f"  {prediction['away_team_name']}: {form_ratings['away']:.0f}\n")
                    f.write(f"  Difference: {form_ratings['difference']:+.0f}\n\n")
                
                if 'upset_potential' in prediction['advanced_elo_analysis']:
                    upset = prediction['advanced_elo_analysis']['upset_potential']
                    f.write(f"Upset Potential: {upset['value']:.1f}/10\n")
                    f.write(f"Description: {upset['description']}\n\n")
            
            # Final Prediction Summary
            f.write(f"{'-'*80}\n")
            f.write(f"PREDICTION SUMMARY\n")
            f.write(f"{'-'*80}\n")
            pred_direction = prediction.get('prediction', '')
            
            # Get confidence
            if 'enhanced_confidence' in prediction:
                confidence = prediction['enhanced_confidence']['score']
                confidence_factors = prediction['enhanced_confidence'].get('factors', {})
                
                f.write(f"Prediction: {pred_direction} Win\n")
                f.write(f"Confidence: {confidence:.1%}\n")
                f.write(f"Confidence Factors:\n")
                for factor, value in confidence_factors.items():
                    f.write(f"  - {factor}: {value:+.0%}\n")
            else:
                confidence = prediction.get('confidence', 0)
                f.write(f"Prediction: {pred_direction} Win\n")
                f.write(f"Confidence: {confidence:.1%}\n")
        
        logger.info(f"Generated match report: {output_file}")

def run_elo_prediction_workflow():
    """Run the complete ELO prediction workflow"""
    workflow = ELOEnhancedPredictionWorkflow()
    
    # Get upcoming matches for Premier League
    league_id = 39  # Premier League
    print(f"Getting upcoming matches for league ID {league_id}...")
    fixtures = workflow.get_upcoming_matches(league_id, days_ahead=3)
    
    print(f"Found {len(fixtures)} upcoming matches")
    
    # Make predictions for all matches
    print("Making predictions...")
    predictions = workflow.make_predictions_for_matches(fixtures)
    
    # Generate visualizations and reports for each match
    print("Generating visualizations and reports...")
    for prediction in predictions:
        workflow.visualize_prediction(prediction)
        workflow.generate_match_report(prediction)
    
    print(f"All predictions complete. Results saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    run_elo_prediction_workflow()
