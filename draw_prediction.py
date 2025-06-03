"""
Draw Prediction Enhancement Module

This module provides specialized features and methods for improving the prediction
of drawn matches in soccer, which was identified as a key weakness in our evaluation.
It introduces new features specifically designed to capture factors that lead to draws
and provides a specialized model focused on detecting balanced team matchups.

Key components:
1. Draw-specific feature engineering
2. Historical draw tendency analysis
3. Team matchup balance metrics
4. Specialized draw prediction model
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

# Import local modules
from team_form import get_team_form, get_head_to_head_analysis
from match_winner import MatchOutcome

logger = logging.getLogger(__name__)

class DrawPredictor:
    """
    Specialized predictor for drawn matches.
    
    This class focuses specifically on identifying balanced team matchups
    and situations likely to result in draws, addressing a key weakness
    in the main prediction model.
    """
    
    def __init__(self):
        """Initialize the draw predictor."""
        self.model = RandomForestClassifier(
            n_estimators=200, 
            max_depth=8,
            class_weight={0: 1, 1: 3}  # Increase weight for draw class
        )
        self.is_calibrated = False
        
    def extract_draw_features(
        self, 
        home_team_id: int,
        away_team_id: int,
        home_form: Dict[str, Any],
        away_form: Dict[str, Any],
        h2h_data: Dict[str, Any],
        league_id: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract features specifically designed to predict draws.
        
        Args:
            home_team_id: ID of the home team
            away_team_id: ID of the away team
            home_form: Form data for the home team
            away_form: Form data for the away team
            h2h_data: Head-to-head data between the teams
            league_id: Optional league ID for context
            
        Returns:
            DataFrame of draw-specific features
        """
        features = {}
        
        # 1. Goal differential similarity (closer teams more likely to draw)
        home_gd = home_form.get('goal_difference_avg', 0)
        away_gd = away_form.get('goal_difference_avg', 0)
        features['goal_diff_similarity'] = 1 / (1 + abs(home_gd - away_gd))
        
        # 2. Form similarity (teams in similar form more likely to draw)
        home_form_points = home_form.get('points_last_5', 0)
        away_form_points = away_form.get('points_last_5', 0)
        features['form_similarity'] = 1 / (1 + abs(home_form_points - away_form_points))
        
        # 3. Historical draw rate between the teams
        draw_rate = h2h_data.get('draw_percentage', 0)
        features['h2h_draw_rate'] = draw_rate
        
        # 4. Historical draw tendency for both teams
        features['home_draw_tendency'] = home_form.get('draw_percentage', 0)
        features['away_draw_tendency'] = away_form.get('draw_percentage', 0)
        
        # 5. Matchup intensity (high-stakes matches less likely to end in draws)
        features['is_derby'] = 1 if h2h_data.get('is_derby', False) else 0
        
        # 6. Defensive vs offensive strength comparison
        home_defense = home_form.get('defense_strength', 0)
        home_offense = home_form.get('attack_strength', 0)
        away_defense = away_form.get('defense_strength', 0)
        away_offense = away_form.get('attack_strength', 0)
        
        # When home defense matches away offense and vice versa, draws more likely
        defense_offense_balance = 1 - abs((home_defense - away_offense) - (away_defense - home_offense))
        features['tactical_balance'] = defense_offense_balance
        
        # 7. Recent draw streak features
        features['home_recent_draws'] = home_form.get('draws_last_5', 0)
        features['away_recent_draws'] = away_form.get('draws_last_5', 0)
        
        # 8. Scoring consistency (inconsistent scoring increases draw chance)
        features['home_scoring_consistency'] = 1 - home_form.get('goals_stddev', 0) / max(home_form.get('goals_avg', 1), 1)
        features['away_scoring_consistency'] = 1 - away_form.get('goals_stddev', 0) / max(away_form.get('goals_avg', 1), 1)
        
        # Convert to DataFrame
        return pd.DataFrame([features])
    
    def fit(self, X: pd.DataFrame, y: np.ndarray) -> 'DrawPredictor':
        """
        Train the draw prediction model.
        
        Args:
            X: Feature matrix
            y: Target vector (1 for draws, 0 for non-draws)
            
        Returns:
            Self for chaining
        """
        self.model.fit(X, y)
        
        # Create calibrated version of the model
        self.calibrated_model = CalibratedClassifierCV(
            base_estimator=self.model,
            method='isotonic',
            cv='prefit'
        )
        self.calibrated_model.fit(X, y)
        self.is_calibrated = True
        
        return self
    
    def predict_draw_probability(
        self, 
        home_team_id: int,
        away_team_id: int,
        league_id: Optional[int] = None
    ) -> float:
        """
        Predict the probability of a draw for a specific match.
        
        Args:
            home_team_id: ID of the home team
            away_team_id: ID of the away team
            league_id: Optional league ID for context
            
        Returns:
            Probability of a draw (0.0 to 1.0)
        """
        # Get team form data
        home_form = get_team_form(home_team_id, league_id or 0, None)
        away_form = get_team_form(away_team_id, league_id or 0, None)
        
        # Get head-to-head analysis
        h2h_data = get_head_to_head_analysis(home_team_id, away_team_id)
        
        # Extract features
        features = self.extract_draw_features(
            home_team_id, away_team_id, 
            home_form, away_form, h2h_data, 
            league_id
        )
        
        # Make prediction using calibrated model if available
        if self.is_calibrated:
            return self.calibrated_model.predict_proba(features)[0, 1]
        
        # For a new model without fit, use heuristic prediction
        if not hasattr(self, 'model') or not hasattr(self.model, 'classes_'):
            # Simple heuristic: average of key features
            draw_tendency = (features['home_draw_tendency'].iloc[0] + 
                            features['away_draw_tendency'].iloc[0]) / 2
            form_similarity = features['form_similarity'].iloc[0]
            h2h_draw_rate = features['h2h_draw_rate'].iloc[0]
            
            # Weighted average
            return 0.4 * draw_tendency + 0.3 * form_similarity + 0.3 * h2h_draw_rate
            
        # Fall back to uncalibrated model
        return self.model.predict_proba(features)[0, 1]

def enhance_draw_predictions(
    predictions: Dict[str, Any],
    home_team_id: int,
    away_team_id: int,
    draw_predictor: Optional[DrawPredictor] = None,
    league_id: int = 0
) -> Dict[str, Any]:
    """
    Enhance the existing predictions by incorporating specialized draw prediction.
    
    Args:
        predictions: Original predictions dictionary with probabilities for each outcome
        home_team_id: ID of the home team
        away_team_id: ID of the away team
        draw_predictor: Optional draw predictor instance
        league_id: Optional league ID for context
        
    Returns:
        Updated predictions dictionary with enhanced draw probability
    """
    # Create predictor if not provided
    if draw_predictor is None:
        draw_predictor = DrawPredictor()
    
    # Get specialized draw probability
    draw_prob = draw_predictor.predict_draw_probability(home_team_id, away_team_id, league_id)
    
    # Blend with original prediction
    original_draw_prob = predictions.get('probabilities', {}).get(MatchOutcome.DRAW.value, 0)
    
    # Use weighted average - more weight to specialized model
    enhanced_draw_prob = 0.7 * draw_prob + 0.3 * original_draw_prob
    
    # Adjust other probabilities proportionally
    home_win_prob = predictions.get('probabilities', {}).get(MatchOutcome.HOME_WIN.value, 0)
    away_win_prob = predictions.get('probabilities', {}).get(MatchOutcome.AWAY_WIN.value, 0)
    
    # Calculate probability adjustment
    prob_adjustment = enhanced_draw_prob - original_draw_prob
    
    if prob_adjustment > 0:
        # Need to decrease other probabilities proportionally
        total_win_prob = home_win_prob + away_win_prob
        if total_win_prob > 0:
            home_win_prob *= (1 - prob_adjustment / total_win_prob)
            away_win_prob *= (1 - prob_adjustment / total_win_prob)
    else:
        # Need to increase other probabilities proportionally
        total_win_prob = home_win_prob + away_win_prob
        if total_win_prob > 0:
            home_win_prob += -prob_adjustment * (home_win_prob / total_win_prob)
            away_win_prob += -prob_adjustment * (away_win_prob / total_win_prob)
    
    # Update predictions with new probabilities
    updated_predictions = predictions.copy()
    updated_predictions['probabilities'] = {
        MatchOutcome.HOME_WIN.value: max(0, min(1, home_win_prob)),
        MatchOutcome.DRAW.value: max(0, min(1, enhanced_draw_prob)),
        MatchOutcome.AWAY_WIN.value: max(0, min(1, away_win_prob))
    }
    
    # Update predicted outcome if draw is now most likely
    if enhanced_draw_prob > home_win_prob and enhanced_draw_prob > away_win_prob:
        updated_predictions['predicted_outcome'] = MatchOutcome.DRAW.value
    
    return updated_predictions