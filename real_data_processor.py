"""
Real Data Processor for Match Winner Prediction

This module integrates with the existing data infrastructure to load real match data
for testing and evaluating the match winner prediction model. It processes historical 
match data to extract relevant features for prediction and evaluation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
import os
from datetime import datetime, timedelta

# Import local modules
from data import get_api_instance
from team_form import get_team_form, get_head_to_head_analysis
from match_winner import predict_match_winner, MatchOutcome

logger = logging.getLogger(__name__)

class RealDataProcessor:
    """
    Process real match data for the winner prediction model.
    
    This class handles loading, preprocessing, and evaluating match data
    for the winner prediction model using actual historical data.
    """
    
    def __init__(self):
        """Initialize the real data processor."""
        self.api = get_api_instance()
        # Leagues with good data quality
        self.major_leagues = {
            39: "Premier League",
            140: "La Liga",
            78: "Bundesliga",
            135: "Serie A",
            61: "Ligue 1"
        }
    
    def load_historical_matches(self, league_id: int, season: int, 
                               limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load historical match data for a specific league and season.
        
        Args:
            league_id: The ID of the league
            season: The season year (e.g., 2021 for 2021/2022)
            limit: Optional limit on the number of matches to return
            
        Returns:
            DataFrame containing historical match data
        """
        logger.info(f"Loading historical match data for league {league_id}, season {season}")
        
        try:
            # Get historical data from the API
            df = self.api.get_historical_data(league_id=league_id, season=season)
            
            if df.empty:
                logger.warning(f"No historical data found for league {league_id}, season {season}")
                return pd.DataFrame()
            
            # Filter out matches with missing essential data
            df = df.dropna(subset=['home_goals', 'away_goals', 'home_team_id', 'away_team_id'])
            
            # Add match outcome column
            df['outcome'] = df.apply(self._determine_outcome, axis=1)
            
            # Sort by date (newest first)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values(by='date', ascending=False)
            
            # Apply limit if specified
            if limit and limit > 0:
                df = df.head(limit)
                
            logger.info(f"Loaded {len(df)} matches successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            return pd.DataFrame()
    
    def _determine_outcome(self, row: pd.Series) -> str:
        """Determine match outcome (home_win, draw, away_win) from score."""
        home_goals = row.get('home_goals', 0)
        away_goals = row.get('away_goals', 0)
        
        if home_goals > away_goals:
            return MatchOutcome.HOME_WIN.value
        elif home_goals < away_goals:
            return MatchOutcome.AWAY_WIN.value
        else:
            return MatchOutcome.DRAW.value
    
    def prepare_prediction_data(self, matches: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Prepare match data for prediction.
        
        Args:
            matches: DataFrame containing match data
            
        Returns:
            List of dictionaries with prepared data for each match
        """
        prepared_data = []
        
        for _, match in matches.iterrows():
            try:
                # Extract required data
                home_team_id = int(match['home_team_id'])
                away_team_id = int(match['away_team_id'])
                league_id = int(match.get('league_id', 0))
                  # Extract or estimate xG values
                home_xg = float(match.get('home_xg', match.get('home_goals', 0)))
                away_xg = float(match.get('away_xg', match.get('away_goals', 0)))
                
                # Get form data
                home_form = self._get_team_form_data(home_team_id)
                away_form = self._get_team_form_data(away_team_id)
                
                # Get head-to-head data
                h2h = self._get_head_to_head_data(home_team_id, away_team_id)
                
                # Extract context factors if available
                context_factors = self._extract_context_factors(match)
                
                # Create prediction input
                match_data = {
                    'fixture_id': match.get('fixture_id'),
                    'date': match.get('date'),
                    'home_team_id': home_team_id,
                    'away_team_id': away_team_id,
                    'league_id': league_id,
                    'home_xg': home_xg,
                    'away_xg': away_xg,
                    'home_form': home_form,
                    'away_form': away_form,
                    'h2h': h2h,
                    'context_factors': context_factors,
                    'actual_outcome': match.get('outcome')
                }
                
                prepared_data.append(match_data)
                
            except Exception as e:
                logger.warning(f"Error preparing match data: {e}")
                continue
                
        logger.info(f"Prepared {len(prepared_data)} matches for prediction")
        return prepared_data
    
    def _get_team_form_data(self, team_id: int) -> Dict[str, Any]:
        """Get team form data or create default if unavailable."""
        try:
            form = get_team_form(team_id, 0, None)  # Using 0 as a default league ID
            if not form:
                # Create default form data if not available
                form = {
                    'form_trend': 0.0,
                    'matches_played': 5,
                    'consistency': 0.5
                }
            return form
        except Exception as e:
            logger.warning(f"Could not get team form for team {team_id}: {e}")
            return {
                'form_trend': 0.0,
                'matches_played': 0,
                'consistency': 0.5
            }
    
    def _get_head_to_head_data(self, home_team_id: int, away_team_id: int) -> Dict[str, Any]:
        """Get head-to-head data or create default if unavailable."""
        try:
            h2h = get_head_to_head_analysis(home_team_id, away_team_id)
            if not h2h:
                # Create default h2h data if not available
                h2h = {
                    'matches_played': 0,
                    'home_win_pct': 0.5,
                    'draw_pct': 0.25,
                    'away_win_pct': 0.25
                }
            return h2h
        except Exception as e:
            logger.warning(f"Could not get h2h data for teams {home_team_id} vs {away_team_id}: {e}")
            return {
                'matches_played': 0,
                'home_win_pct': 0.5,
                'draw_pct': 0.25,
                'away_win_pct': 0.25
            }
    
    def _extract_context_factors(self, match: pd.Series) -> Dict[str, Any]:
        """Extract context factors from match data."""
        context = {}
        
        # Check if there's weather info
        if 'weather' in match and match['weather']:
            weather = match['weather']
            context['weather_data_available'] = True
            if isinstance(weather, dict):
                context['is_rainy'] = 'rain' in str(weather.get('condition', '')).lower()
                context['is_snowy'] = 'snow' in str(weather.get('condition', '')).lower()
        
        # Check for importance/high stakes
        stage = str(match.get('stage', '')).lower()
        context['high_stakes'] = any(x in stage for x in ['final', 'semi', 'playoff', 'promotion', 'relegation'])
        
        # Check for derby matches
        context['is_derby'] = match.get('is_derby', False)
        
        return context
    
    def predict_and_evaluate(self, matches_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run predictions on match data and evaluate accuracy.
        
        Args:
            matches_data: List of processed match data dictionaries
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not matches_data:
            return {'error': 'No match data provided'}
        
        predictions = []
        correct_predictions = 0
        total_predictions = len(matches_data)
        
        for match_data in matches_data:
            try:
                # Make prediction
                prediction = predict_match_winner(
                    home_team_id=match_data['home_team_id'],
                    away_team_id=match_data['away_team_id'],
                    home_xg=match_data['home_xg'],
                    away_xg=match_data['away_xg'],
                    home_form=match_data['home_form'],
                    away_form=match_data['away_form'],
                    h2h=match_data['h2h'],
                    league_id=match_data['league_id'],
                    context_factors=match_data['context_factors']
                )
                
                # Add actual outcome and check if correct
                actual_outcome = match_data.get('actual_outcome')
                predicted_outcome = prediction['most_likely_outcome']
                
                match_result = {
                    'fixture_id': match_data.get('fixture_id'),
                    'date': match_data.get('date'),
                    'home_team_id': match_data['home_team_id'],
                    'away_team_id': match_data['away_team_id'],
                    'actual_outcome': actual_outcome,
                    'predicted_outcome': predicted_outcome,
                    'prediction_correct': actual_outcome == predicted_outcome,
                    'prediction': prediction
                }
                
                predictions.append(match_result)
                
                if actual_outcome == predicted_outcome:
                    correct_predictions += 1
                    
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                continue
                
        # Calculate overall metrics
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Calculate accuracy by outcome type
        outcome_stats = self._calculate_outcome_statistics(predictions)
        
        # Calculate confidence correlation
        confidence_correlation = self._calculate_confidence_correlation(predictions)
        
        evaluation_results = {
            'total_matches': total_predictions,
            'correct_predictions': correct_predictions,
            'overall_accuracy': round(accuracy * 100, 2),
            'outcome_statistics': outcome_stats,
            'confidence_correlation': confidence_correlation,
            'predictions': predictions
        }
        return evaluation_results
    
    def _calculate_outcome_statistics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate accuracy statistics by outcome type."""
        stats = {
            MatchOutcome.HOME_WIN.value: {'correct': 0, 'total': 0, 'accuracy': 0.0},
            MatchOutcome.DRAW.value: {'correct': 0, 'total': 0, 'accuracy': 0.0},
            MatchOutcome.AWAY_WIN.value: {'correct': 0, 'total': 0, 'accuracy': 0.0}
        }
        
        for pred in predictions:
            actual = pred.get('actual_outcome')
            if actual not in stats:
                continue
                
            stats[actual]['total'] += 1
            if pred.get('prediction_correct', False):
                stats[actual]['correct'] += 1
        
        # Calculate accuracy for each outcome
        for outcome, data in stats.items():
            if data['total'] > 0:
                data['accuracy'] = round(float(data['correct']) / float(data['total']) * 100.0, 2)
        
        return stats
    
    def _calculate_confidence_correlation(self, predictions: List[Dict[str, Any]]) -> float:
        """
        Calculate how well confidence scores correlate with prediction accuracy.
        Higher values indicate confidence scores are more predictive of accuracy.
        """
        if not predictions:
            return 0.0
            
        confidence_values = []
        correctness_values = []
        
        for pred in predictions:
            try:
                confidence = pred.get('prediction', {}).get('confidence', {}).get('score', 0.5)
                is_correct = 1 if pred.get('prediction_correct', False) else 0
                
                confidence_values.append(confidence)
                correctness_values.append(is_correct)
            except:
                pass
                
        if not confidence_values:
            return 0.0
            
        # Calculate correlation coefficient
        try:
            correlation = np.corrcoef(confidence_values, correctness_values)[0, 1]
            return round(float(correlation), 3)
        except:
            return 0.0
    
    def generate_evaluation_report(self, evaluation: Dict[str, Any]) -> str:
        """
        Generate a human-readable evaluation report.
        
        Args:
            evaluation: Evaluation results dictionary
            
        Returns:
            Formatted evaluation report
        """
        report = []
        report.append("=== Match Winner Prediction Model Evaluation ===")
        report.append(f"Total matches evaluated: {evaluation.get('total_matches', 0)}")
        report.append(f"Overall accuracy: {evaluation.get('overall_accuracy', 0)}%")
        report.append(f"Confidence correlation: {evaluation.get('confidence_correlation', 0)}")
        
        # Outcome statistics
        report.append("\nAccuracy by outcome type:")
        for outcome, stats in evaluation.get('outcome_statistics', {}).items():
            report.append(f"  {outcome.upper()}: {stats.get('accuracy', 0)}% ({stats.get('correct', 0)}/{stats.get('total', 0)})")
        
        # Sample predictions (first 5)
        predictions = evaluation.get('predictions', [])
        if predictions:
            report.append("\nSample predictions (first 5):")
            for i, pred in enumerate(predictions[:5]):
                report.append(f"\nMatch {i+1}:")
                report.append(f"  Teams: {pred.get('home_team_id')} vs {pred.get('away_team_id')}")
                report.append(f"  Actual outcome: {pred.get('actual_outcome')}")
                report.append(f"  Predicted outcome: {pred.get('predicted_outcome')}")
                report.append(f"  Prediction correct: {'Yes' if pred.get('prediction_correct') else 'No'}")
                
                # Add probability details
                probs = pred.get('prediction', {}).get('probabilities', {})
                if probs:
                    report.append(f"  Home win: {probs.get('home_win', 0)}%")
                    report.append(f"  Draw: {probs.get('draw', 0)}%")
                    report.append(f"  Away win: {probs.get('away_win', 0)}%")
        
        return "\n".join(report)


def run_real_data_evaluation(league_id: int = 39, season: int = 2022, limit: int = 100) -> Dict[str, Any]:
    """
    Run a complete evaluation of the match winner prediction model on real data.
    
    Args:
        league_id: League ID to evaluate (default: Premier League)
        season: Season year to evaluate (default: 2022)
        limit: Number of matches to evaluate (default: 100)
        
    Returns:
        Evaluation results dictionary
    """
    processor = RealDataProcessor()
    
    # Load historical matches
    matches_df = processor.load_historical_matches(league_id, season, limit)
    
    if matches_df.empty:
        logger.error(f"No matches found for league {league_id}, season {season}")
        return {'error': 'No matches found'}
    
    # Prepare data for prediction
    prepared_data = processor.prepare_prediction_data(matches_df)
    
    # Run predictions and evaluate
    evaluation = processor.predict_and_evaluate(prepared_data)
    
    # Generate report
    report = processor.generate_evaluation_report(evaluation)
    logger.info(f"\n{report}")
    
    return evaluation


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run evaluation for Premier League (ID: 39) 2022 season
    run_real_data_evaluation(league_id=39, season=2022, limit=50)
