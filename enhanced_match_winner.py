"""
Enhanced Match Winner Prediction Integration

This module integrates the specialized draw prediction functionality with the main match winner 
prediction system to improve overall accuracy, particularly for draw predictions which were
identified as a weakness in our evaluation.

The integration provides:
1. A wrapper for the main prediction function with enhanced draw prediction
2. Calibration utilities for better probability estimates
3. Utility functions for evaluation and reporting
"""

import logging
from typing import Dict, Any, Optional, List, Union, Tuple, cast
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression

# Import local modules
from match_winner import predict_match_winner, MatchOutcome
from team_form import get_team_form, get_head_to_head_analysis

logger = logging.getLogger(__name__)

class EnhancedPredictionSystem:
    """
    Enhanced prediction system that combines the main match winner predictions
    with specialized draw predictions and probability calibration.
    """
    
    def __init__(self):
        """Initialize the enhanced prediction system."""
        # Initialize the draw prediction component
        # Will be imported dynamically to avoid circular imports
        self.draw_predictor = None
        self.is_calibrated = False
        self.calibrators: Dict[str, Optional[IsotonicRegression]] = {
            MatchOutcome.HOME_WIN.value: None,
            MatchOutcome.DRAW.value: None,
            MatchOutcome.AWAY_WIN.value: None
        }
        
    def _ensure_draw_predictor(self):
        """Ensure draw predictor is initialized."""
        if self.draw_predictor is None:
            # Import here to avoid circular imports
            from draw_prediction import DrawPredictor
            self.draw_predictor = DrawPredictor()
        
    def calibrate_from_history(
        self, 
        historical_predictions: List[Dict[str, Any]],
        actual_outcomes: List[str]
    ) -> None:
        """
        Calibrate probability estimates based on historical prediction data.
        
        Args:
            historical_predictions: List of prediction dictionaries
            actual_outcomes: List of actual match outcomes
        """
        # Create dataframes for each outcome type
        outcome_data: Dict[str, Dict[str, List[float]]] = {}
        for outcome in MatchOutcome:
            outcome_value = outcome.value
            outcome_data[outcome_value] = {
                'predicted_probs': [],
                'actual_outcome': []
            }
        
        # Extract data for calibration
        for pred, actual in zip(historical_predictions, actual_outcomes):
            for outcome in MatchOutcome:
                outcome_value = outcome.value
                prob = pred.get('probabilities', {}).get(outcome_value, 0)
                is_correct = 1.0 if actual == outcome_value else 0.0
                
                outcome_data[outcome_value]['predicted_probs'].append(prob)
                outcome_data[outcome_value]['actual_outcome'].append(is_correct)
        
        # Train calibration models for each outcome
        for outcome in MatchOutcome:
            outcome_value = outcome.value
            probs = np.array(outcome_data[outcome_value]['predicted_probs'])
            actuals = np.array(outcome_data[outcome_value]['actual_outcome'])
            
            # Only calibrate if we have enough data
            if len(probs) >= 30:  # Minimum threshold for reliable calibration
                try:
                    calibrator = IsotonicRegression(out_of_bounds='clip')
                    calibrator.fit(probs, actuals)
                    self.calibrators[outcome_value] = calibrator
                    logger.info(f"Calibrated model for {outcome_value}")
                except Exception as e:
                    logger.error(f"Error calibrating {outcome_value}: {e}")
        
        self.is_calibrated = any(c is not None for c in self.calibrators.values())
        
    def predict(
        self, 
        home_team_id: int,
        away_team_id: int,
        league_id: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make an enhanced prediction for a match outcome.
        
        Args:
            home_team_id: ID of the home team
            away_team_id: ID of the away team
            league_id: Optional league ID for context
            **kwargs: Additional keyword arguments for the base predictor
            
        Returns:
            Enhanced prediction dictionary
        """        # Ensure we have the necessary form and h2h data
        home_form = kwargs.get('home_form')
        away_form = kwargs.get('away_form')
        h2h = kwargs.get('h2h')
        
        if home_form is None:
            home_form = get_team_form(home_team_id, league_id or 0, None)
        if away_form is None:
            away_form = get_team_form(away_team_id, league_id or 0, None)
        if h2h is None:
            h2h = get_head_to_head_analysis(home_team_id, away_team_id)
        
        # Calculate dynamic xG values specific to each team
        if 'home_xg' not in kwargs or 'away_xg' not in kwargs:
            # Import dynamic xG calculator
            from dynamic_xg_calculator import calculate_match_xg
            
            calculated_home_xg, calculated_away_xg = calculate_match_xg(
                home_team_id=home_team_id,
                away_team_id=away_team_id,
                home_form=home_form,
                away_form=away_form,
                league_id=league_id or 39,
                h2h_data=h2h
            )
            
            home_xg = kwargs.get('home_xg', calculated_home_xg)
            away_xg = kwargs.get('away_xg', calculated_away_xg)
        else:
            home_xg = kwargs.get('home_xg', 1.3)
            away_xg = kwargs.get('away_xg', 1.1)
            
        # Get base prediction with all required parameters
        base_prediction = predict_match_winner(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            league_id=league_id or 0,  # Use 0 as default league ID if None
            home_xg=home_xg,
            away_xg=away_xg,
            home_form=home_form,
            away_form=away_form,
            h2h=h2h,
            context_factors=kwargs.get('context_factors')
        )
          # Ensure draw predictor is initialized
        self._ensure_draw_predictor()
        
        # Convert probabilities from percentages to decimals for enhanced processing
        normalized_prediction = base_prediction.copy()
        base_probs = base_prediction.get('probabilities', {})
        
        # Check if probabilities are in percentage format (>1) and convert to decimals
        if any(prob > 1 for prob in base_probs.values()):
            normalized_prediction['probabilities'] = {
                'home_win': base_probs.get('home_win', 0) / 100.0,
                'draw': base_probs.get('draw', 0) / 100.0,
                'away_win': base_probs.get('away_win', 0) / 100.0
            }
        
        # Enhance with specialized draw prediction
        from draw_prediction import enhance_draw_predictions
        enhanced_prediction = enhance_draw_predictions(
            normalized_prediction,
            home_team_id,
            away_team_id,
            self.draw_predictor,
            league_id or 0  # Use 0 as default league ID if None
        )
        
        # Apply calibration if available
        if self.is_calibrated:
            probabilities = enhanced_prediction.get('probabilities', {})
            calibrated_probs = {}
            
            for outcome in MatchOutcome:
                outcome_value = outcome.value
                raw_prob = probabilities.get(outcome_value, 0)
                
                # Apply calibration if available for this outcome
                calibrator = self.calibrators[outcome_value]
                if calibrator is not None:
                    calibrated_prob = calibrator.predict([raw_prob])[0]
                    calibrated_probs[outcome_value] = max(0, min(1, calibrated_prob))
                else:
                    calibrated_probs[outcome_value] = raw_prob
            
            # Normalize probabilities
            total_prob = sum(calibrated_probs.values())
            if total_prob > 0:
                for outcome in calibrated_probs:
                    calibrated_probs[outcome] /= total_prob
            
            enhanced_prediction['probabilities'] = calibrated_probs
              # Update predicted outcome based on calibrated probabilities
            if calibrated_probs:
                # Find outcome with maximum probability
                max_outcome = max(calibrated_probs.items(), key=lambda x: x[1])
                enhanced_prediction['predicted_outcome'] = max_outcome[0]
        
        # Convert probabilities back to percentage format for API consistency
        probs = enhanced_prediction.get('probabilities', {})
        if all(prob <= 1 for prob in probs.values()):  # If they're in decimal format
            enhanced_prediction['probabilities'] = {
                'home_win': round(probs.get('home_win', 0) * 100, 1),
                'draw': round(probs.get('draw', 0) * 100, 1),
                'away_win': round(probs.get('away_win', 0) * 100, 1)
            }
        
        return enhanced_prediction

    def batch_predict(
        self,
        matches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Make predictions for a batch of matches.
        
        Args:
            matches: List of match dictionaries with home_team_id, away_team_id, etc.
            
        Returns:
            List of enhanced prediction dictionaries
        """
        predictions = []
        for match in matches:
            try:
                # Extract required fields with type safety
                home_team_id = int(match.get('home_team_id', 0) or 0)
                away_team_id = int(match.get('away_team_id', 0) or 0)
                league_id = int(match.get('league_id', 0) or 0)
                fixture_id = match.get('fixture_id')
                
                # Skip if we don't have valid IDs
                if home_team_id <= 0 or away_team_id <= 0:
                    logger.warning(f"Invalid team IDs for match {fixture_id}")
                    continue
                
                prediction = self.predict(
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    league_id=league_id,
                    fixture_id=fixture_id
                )
                # Include match info in prediction
                prediction['fixture_id'] = fixture_id
                prediction['home_team_id'] = home_team_id
                prediction['away_team_id'] = away_team_id
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting match {match.get('fixture_id')}: {e}")
                continue
                
        return predictions

def predict_with_enhanced_system(
    home_team_id: int,
    away_team_id: int,
    league_id: Optional[int] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to make a prediction using the enhanced system.
    
    Args:
        home_team_id: ID of the home team
        away_team_id: ID of the away team
        league_id: Optional league ID for context
        **kwargs: Additional keyword arguments for the predictor
        
    Returns:
        Enhanced prediction dictionary
    """
    system = EnhancedPredictionSystem()
    return system.predict(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        league_id=league_id,
        **kwargs
    )