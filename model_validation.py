"""
Validation of goals prediction model accuracy using historical data.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
import logging
from .prediction_integration_enhanced import generate_enhanced_goals_prediction
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import norm

logger = logging.getLogger(__name__)

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard regression metrics"""
    return {
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
        'r2': float(r2_score(y_true, y_pred))
    }

def calculate_probability_metrics(
    true_probabilities: np.ndarray,
    predicted_probabilities: np.ndarray
) -> Dict[str, float]:
    """Calculate probability prediction metrics like Brier score"""
    brier = np.mean((true_probabilities - predicted_probabilities) ** 2)
    log_loss = -np.mean(true_probabilities * np.log(predicted_probabilities + 1e-10))
    
    return {
        'brier_score': float(brier),
        'log_loss': float(log_loss)
    }

def validate_goals_model(
    historical_matches: List[Dict[str, Any]],
    weather_data: Dict[str, str],  # Match ID to weather data
    league_data: Dict[str, Any]  # League metadata
) -> Dict[str, Any]:
    """
    Validate goals model performance on historical matches with enhanced metrics.
    
    Args:
        historical_matches: List of historical match data
        weather_data: Weather data for matches
        league_data: League-specific information
        
    Returns:
        Dictionary containing comprehensive validation metrics including:
        - Basic regression metrics (RMSE, MAE, R², MAPE)
        - Probability calibration metrics
        - Range-specific accuracy
        - Trend prediction accuracy
        - Advanced metrics (Brier score, sharpness, etc.)
    """
    try:
        results = []
        total_predictions = len(historical_matches)
        successful_predictions = 0
        
        # Arrays for storing actual vs predicted values
        actual_totals = []
        predicted_totals = []
        actual_over25 = []
        predicted_over25 = []
        actual_btts = []
        predicted_btts = []
        
        # Arrays for advanced metrics
        actual_home = []
        predicted_home = []
        actual_away = []
        predicted_away = []
        actual_trends = []
        predicted_trends = []
        previous_total = None
        
        for match in historical_matches:            try:
                # Get prediction
                prediction = generate_enhanced_goals_prediction(
                    home_team_id=match['home_team_id'],
                    away_team_id=match['away_team_id'],
                    league_id=match['league_id'],
                    home_form=match.get('home_form', {}),
                    away_form=match.get('away_form', {}),
                    h2h=match.get('h2h', {}),
                    weather_data=weather_data.get(str(match['match_id'])),
                    elo_ratings=match.get('elo_ratings'),
                    context_factors=match.get('context_factors')
                )
                
                # Basic goals data
                actual_home.append(match['home_goals'])
                actual_away.append(match['away_goals'])
                predicted_home.append(prediction['home_xg'])
                predicted_away.append(prediction['away_xg'])
                
                actual_total = match['home_goals'] + match['away_goals']
                predicted_total = prediction['total_xg']
                
                # Store values for main metrics
                actual_totals.append(actual_total)
                predicted_totals.append(predicted_total)
                
                # Over/Under metrics
                actual_over25.append(1.0 if actual_total > 2.5 else 0.0)
                predicted_over25.append(prediction['over_under'].get('over_2.5', 0.5))
                
                # BTTS metrics
                actual_btts.append(1.0 if match['home_goals'] > 0 and match['away_goals'] > 0 else 0.0)
                predicted_btts.append(prediction['btts_prob'])
                
                # Trend metrics
                if previous_total is not None:
                    actual_trends.append(1.0 if actual_total > previous_total else 0.0)
                    predicted_trends.append(1.0 if predicted_total > previous_total else 0.0)
                previous_total = actual_total
                
                successful_predictions += 1
                
            except Exception as e:
                logger.error(f"Error processing match {match.get('match_id')}: {e}")
                continue
                
        # Calculate metrics
        if successful_predictions > 0:
            # Convert lists to numpy arrays for calculations
            actual_totals = np.array(actual_totals)
            predicted_totals = np.array(predicted_totals)
            actual_over25 = np.array(actual_over25)
            predicted_over25 = np.array(predicted_over25)
            actual_btts = np.array(actual_btts)
            predicted_btts = np.array(predicted_btts)
            actual_home = np.array(actual_home)
            predicted_home = np.array(predicted_home)
            actual_away = np.array(actual_away)
            predicted_away = np.array(predicted_away)
            
            validation_results = {}
            
            # 1. Basic regression metrics for total goals
            validation_results['regression_metrics'] = {
                'rmse': float(np.sqrt(mean_squared_error(actual_totals, predicted_totals))),
                'mae': float(mean_absolute_error(actual_totals, predicted_totals)),
                'r2': float(r2_score(actual_totals, predicted_totals)),
                'mape': float(np.mean(np.abs((actual_totals - predicted_totals) / (actual_totals + 1e-8))) * 100)
            }
            
            # 2. Range-specific accuracy
            def calculate_range_accuracy(y_true, y_pred, lower, upper):
                true_in_range = (y_true >= lower) & (y_true <= upper)
                pred_in_range = (y_pred >= lower) & (y_pred <= upper)
                return float(np.mean(true_in_range == pred_in_range))
                
            validation_results['range_metrics'] = {
                'low_scoring_accuracy': calculate_range_accuracy(actual_totals, predicted_totals, 0, 2),
                'high_scoring_accuracy': calculate_range_accuracy(actual_totals, predicted_totals, 3, float('inf')),
                'exact_score_accuracy': float(np.mean(np.round(predicted_totals) == actual_totals))
            }
            
            # 3. Over/Under metrics
            validation_results['over25_metrics'] = calculate_probability_metrics(
                actual_over25,
                predicted_over25
            )
            
            # 4. BTTS metrics
            validation_results['btts_metrics'] = calculate_probability_metrics(
                actual_btts,
                predicted_btts
            )
            
            # 5. Trend metrics
            if len(actual_trends) > 0:
                actual_trends = np.array(actual_trends)
                predicted_trends = np.array(predicted_trends)
                validation_results['trend_metrics'] = {
                    'trend_accuracy': float(np.mean(actual_trends == predicted_trends)),
                    'momentum_correlation': float(np.corrcoef(
                        np.convolve(actual_totals, [1/3]*3, mode='valid'),
                        np.convolve(predicted_totals, [1/3]*3, mode='valid')
                    )[0,1])
                }
            
            # 6. Model calibration metrics
            prob_true, prob_pred = np.histogram(actual_totals, bins=10)[0], np.histogram(predicted_totals, bins=10)[0]
            prob_true = prob_true / np.sum(prob_true)
            prob_pred = prob_pred / np.sum(prob_pred)
            
            validation_results['probability_metrics'] = {
                'calibration_error': float(np.mean(np.abs(prob_true - prob_pred))),
                'sharpness': float(np.std(predicted_totals))
            }
            
            # Log key metrics
            logger.info("=== Model Validation Results ===")
            logger.info(f"Total matches: {total_predictions}")
            logger.info(f"Successfully processed: {successful_predictions}")
            logger.info(f"RMSE: {validation_results['regression_metrics']['rmse']:.3f}")
            logger.info(f"R²: {validation_results['regression_metrics']['r2']:.3f}")
            logger.info(f"Low scoring accuracy: {validation_results['range_metrics']['low_scoring_accuracy']:.3f}")
            logger.info(f"Trend accuracy: {validation_results['trend_metrics']['trend_accuracy']:.3f}")
            logger.info(f"Calibration error: {validation_results['probability_metrics']['calibration_error']:.3f}")
            
            return validation_results
            
        else:
            raise ValueError("No successful predictions to analyze")
            
    except Exception as e:
        logger.error(f"Error validating goals model: {e}")
        return {
            'error': str(e),
            'total_matches': 0,
            'successful_predictions': 0
        }
        
def analyze_weather_impact(
    actual_totals: np.ndarray,
    predicted_totals: np.ndarray,
    match_ids: List[str],
    weather_data: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Analyze prediction accuracy under different weather conditions"""
    weather_conditions = {
        'clear': {'actual': [], 'predicted': []},
        'rain': {'actual': [], 'predicted': []},
        'snow': {'actual': [], 'predicted': []}
    }
    
    # Group matches by weather condition
    for i, match_id in enumerate(match_ids):
        weather = weather_data.get(str(match_id), {})
        condition = weather.get('condition', 'clear').lower()
        
        if condition in weather_conditions:
            weather_conditions[condition]['actual'].append(actual_totals[i])
            weather_conditions[condition]['predicted'].append(predicted_totals[i])
            
    # Calculate metrics for each condition
    results = {}
    for condition, data in weather_conditions.items():
        if len(data['actual']) > 0:
            actual = np.array(data['actual'])
            predicted = np.array(data['predicted'])
            
            results[condition] = {
                'matches': len(data['actual']),
                'mean_actual': float(np.mean(actual)),
                'mean_predicted': float(np.mean(predicted)),
                'rmse': float(np.sqrt(mean_squared_error(actual, predicted))),
                'mae': float(mean_absolute_error(actual, predicted))
            }
            
    return results
