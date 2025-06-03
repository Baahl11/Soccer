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
        # Inicializar validador de coherencia y analizador táctico
        from prediction_coherence_validator import CoherenceValidator
        from enhanced_tactical_analyzer import EnhancedTacticalAnalyzer
        
        coherence_validator = CoherenceValidator()
        tactical_analyzer = EnhancedTacticalAnalyzer()
        predictions = []
        
        for fixture in fixtures:
            logger.info(f"Making prediction for {fixture['home_team_name']} vs {fixture['away_team_name']}")
              # Intentar usar el make_integrated_prediction primero
            try:
                # Importar la función aquí para evitar dependencias circulares
                from prediction_integration import make_integrated_prediction
                
                # Usar el sistema integrado completo - ahora compatible con fixtures sintéticos
                if fixture['fixture_id'] and isinstance(fixture['fixture_id'], int):
                    # Para fixtures sintéticos (>= 1000000), pasar los datos del fixture
                    enhanced = make_integrated_prediction(fixture['fixture_id'], fixture_data=fixture)
                    if enhanced:
                        logger.info(f"Prediction made using integrated system for fixture {fixture['fixture_id']}")
                    else:
                        raise ValueError("No prediction returned from integrated system")
                else:
                    raise ValueError("No valid fixture_id for integrated prediction")
                    
            except Exception as e:
                # Si falla, usar el enfoque simplificado con mejoras ELO
                logger.warning(f"Failed to use integrated prediction: {e}. Using simplified approach.")
                
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
            
            # Validar coherencia entre ratings Elo y predicciones de goles
            try:
                elo_ratings = {
                    'home': enhanced.get('advanced_elo_analysis', {}).get('form_adjusted_elo', {}).get('home', 1500),
                    'away': enhanced.get('advanced_elo_analysis', {}).get('form_adjusted_elo', {}).get('away', 1500),
                    'elo_diff': enhanced.get('advanced_elo_analysis', {}).get('form_adjusted_elo', {}).get('difference', 0)
                }
                
                # Verificar si la predicción es coherente con los ratings Elo
                is_coherent = coherence_validator.is_prediction_coherent_with_elo(enhanced, elo_ratings)
                
                if not is_coherent:
                    logger.warning(f"Predicción incoherente con Elo para {fixture['home_team_name']} vs {fixture['away_team_name']}. Ajustando...")
                    enhanced = coherence_validator.validate_and_adjust_goal_predictions(enhanced, elo_ratings)
                    enhanced['adjusted_for_coherence'] = True
                else:
                    enhanced['adjusted_for_coherence'] = False
            except Exception as e:
                logger.warning(f"Error validando coherencia: {e}")
            
            # Añadir análisis táctico mejorado para cualquier liga
            try:
                home_profile = tactical_analyzer.get_team_tactical_profile(
                    fixture['home_team_id'], 
                    fixture['home_team_name']
                )
                
                away_profile = tactical_analyzer.get_team_tactical_profile(
                    fixture['away_team_id'], 
                    fixture['away_team_name']
                )
                
                matchup = tactical_analyzer.analyze_tactical_matchup(home_profile, away_profile)
                
                enhanced['tactical_analysis'] = {
                    'home_team_profile': home_profile,
                    'away_team_profile': away_profile,
                    'tactical_matchup': matchup
                }
                
                logger.info(f"Added enhanced tactical analysis for {fixture['home_team_name']} vs {fixture['away_team_name']}")
            except Exception as e:
                logger.warning(f"Error generando análisis táctico: {e}")
            
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
        Create a visual representation of the prediction including real-time odds
        
        Args:
            prediction: The prediction data to visualize
            output_file: Optional path to save the visualization
        """
        plt.style.use('seaborn')
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Match Prediction Analysis: {prediction['home_team_name']} vs {prediction['away_team_name']}", 
                    fontsize=16, fontweight='bold')
        
        # Goals prediction plot (top left)
        goals = [prediction['predicted_home_goals'], prediction['predicted_away_goals']]
        teams = [prediction['home_team_name'], prediction['away_team_name']]
        axs[0, 0].bar(teams, goals)
        axs[0, 0].set_title('Predicted Goals')
        
        # Real-time odds plot (top right)
        if 'real_time_odds' in prediction:
            odds = prediction['real_time_odds']
            outcomes = ['Home Win', 'Draw', 'Away Win']
            probabilities = [odds['home_win']*100, odds['draw']*100, odds['away_win']*100]
            
            axs[0, 1].bar(outcomes, probabilities)
            axs[0, 1].set_title('Real-time Win Probabilities')
            axs[0, 1].set_ylim(0, 100)
            for i, prob in enumerate(probabilities):
                axs[0, 1].text(i, prob+1, f'{prob:.1f}%', ha='center')
        
        # Provider reliability comparison (bottom left)
        if 'real_time_odds' in prediction and 'providers_data' in prediction['real_time_odds']:
            providers = list(prediction['real_time_odds']['providers_data'].keys())
            reliability_scores = [data.get('reliability_score', 0) 
                               for data in prediction['real_time_odds']['providers_data'].values()]
            
            if providers and reliability_scores:
                axs[1, 0].bar(providers, reliability_scores)
                axs[1, 0].set_title('Provider Reliability Scores')
                plt.setp(axs[1, 0].xaxis.get_majorticklabels(), rotation=45)
        
        # Key insights text (bottom right)
        insights_text = []
        
        # Add prediction summary and confidence
        pred_direction = prediction.get('prediction', '')
        if 'enhanced_confidence' in prediction:
            confidence = prediction['enhanced_confidence']['score']
        else:
            confidence = prediction.get('confidence', 0)
        
        # Add real-time odds insights
        if 'real_time_odds' in prediction:
            odds = prediction['real_time_odds']
            max_prob = max(odds['home_win'], odds['draw'], odds['away_win'])
            if max_prob == odds['home_win']:
                insights_text.append(f"Bookmakers favor {prediction['home_team_name']}")
            elif max_prob == odds['away_win']:
                insights_text.append(f"Bookmakers favor {prediction['away_team_name']}")
            else:
                insights_text.append("Bookmakers suggest a likely draw")
        
        axs[1, 1].axis('off')
        y_pos = 0.95
        for insight in insights_text:
            axs[1, 1].text(0, y_pos, insight, fontsize=11)
            y_pos -= 0.12
        
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
        Generate a detailed match report with ELO analysis and real-time odds
        
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
            f.write(f"Date: {prediction['match_date']}\n")
            f.write(f"League/Competition: {prediction.get('competition', 'Unknown')}\n\n")
            
            # Real-time odds analysis
            if 'real_time_odds' in prediction:
                odds = prediction['real_time_odds']
                f.write(f"{'='*40}\n")
                f.write("REAL-TIME ODDS ANALYSIS\n")
                f.write(f"{'='*40}\n\n")
                
                f.write("Aggregated Probabilities:\n")
                f.write(f"Home Win: {odds['home_win']*100:.1f}%\n")
                f.write(f"Draw: {odds['draw']*100:.1f}%\n")
                f.write(f"Away Win: {odds['away_win']*100:.1f}%\n\n")
                
                f.write("Provider Details:\n")
                for provider, data in odds['providers_data'].items():
                    f.write(f"\n{provider}:\n")
                    f.write(f"  Reliability Score: {data.get('reliability_score', 'N/A')}\n")
                    f.write(f"  Home Win: {data.get('home_win', 0)*100:.1f}%\n")
                    f.write(f"  Draw: {data.get('draw', 0)*100:.1f}%\n")
                    f.write(f"  Away Win: {data.get('away_win', 0)*100:.1f}%\n")
                f.write(f"\nLast Update: {odds['last_update']}\n\n")
            
            # Prediction details
            f.write(f"{'='*40}\n")
            f.write("PREDICTION DETAILS\n")
            f.write(f"{'='*40}\n\n")
            
            f.write(f"Predicted Score:\n")
            f.write(f"{prediction['home_team_name']}: {prediction['predicted_home_goals']:.2f}\n")
            f.write(f"{prediction['away_team_name']}: {prediction['predicted_away_goals']:.2f}\n\n")
            
            # Source contributions
            f.write("Model Contributions:\n")
            for source, contribution in prediction['source_contributions'].items():
                f.write(f"\n{source.title()}:\n")
                f.write(f"  Weight: {contribution['weight']}\n")
                if 'home_goals_contribution' in contribution:
                    f.write(f"  Home Goals Impact: {contribution['home_goals_contribution']:.2f}\n")
                    f.write(f"  Away Goals Impact: {contribution['away_goals_contribution']:.2f}\n")
                if source == 'odds' and 'odds_reliability' in contribution:
                    f.write(f"  Odds Reliability Score: {contribution['odds_reliability']:.2f}\n")
        
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
