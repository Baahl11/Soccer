"""
Ensemble Prediction Model for 1X2 Soccer Predictions

This module implements an ensemble approach combining multiple models for soccer match predictions:
1. Base ELO model for initial probabilities
2. Specialized draw prediction model
3. League-specific adjustments
4. Dynamic probability balancing
"""

from typing import Dict, Any, Optional, List
import logging
from team_elo_rating import get_elo_ratings_for_match
from draw_prediction import enhance_draw_predictions
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelWeight:
    """Weights for different model components in the ensemble"""
    elo_weight: float = 0.5
    draw_weight: float = 0.3
    league_weight: float = 0.2

class EnsemblePredictionModel:
    def __init__(self, 
                 weights: Optional[ModelWeight] = None,
                 use_dynamic_weights: bool = True):
        """
        Initialize the ensemble prediction model
        
        Args:
            weights: Optional weights for different model components
            use_dynamic_weights: Whether to use dynamic weight adjustment
        """
        self.weights = weights or ModelWeight()
        self.use_dynamic_weights = use_dynamic_weights
        self.historical_accuracy = {
            'elo': 0.0,
            'draw': 0.0,
            'league': 0.0
        }
    
    def _calculate_dynamic_weights(self, match_context: Dict[str, Any]) -> ModelWeight:
        """
        Calculate dynamic weights based on historical accuracy and match context
        
        Args:
            match_context: Context information about the match
        
        Returns:
            Updated model weights
        """
        if not self.use_dynamic_weights:
            return self.weights
            
        weights = ModelWeight()
        total_accuracy = sum(self.historical_accuracy.values())
        
        if total_accuracy > 0:
            # Adjust weights based on historical performance
            weights.elo_weight = 0.4 + (self.historical_accuracy['elo'] / total_accuracy) * 0.2
            weights.draw_weight = 0.2 + (self.historical_accuracy['draw'] / total_accuracy) * 0.2
            weights.league_weight = 0.2 + (self.historical_accuracy['league'] / total_accuracy) * 0.2
            
            # Normalize weights
            total = weights.elo_weight + weights.draw_weight + weights.league_weight
            weights.elo_weight /= total
            weights.draw_weight /= total 
            weights.league_weight /= total
            
        return weights
        
    def predict(self,
                home_team_id: int,
                away_team_id: int,
                league_id: int,
                match_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate ensemble prediction for a match
        
        Args:
            home_team_id: ID of home team
            away_team_id: ID of away team
            league_id: ID of league
            match_context: Additional match context (weather, form, etc)
            
        Returns:
            Dictionary with prediction probabilities and metadata
        """
        # Get base ELO predictions
        elo_prediction = get_elo_ratings_for_match(home_team_id, away_team_id, league_id)
        
        # Get specialized draw prediction
        full_context = {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "league_id": league_id,
            "probabilities": {
                "HOME_WIN": elo_prediction["win_probability"],
                "DRAW": elo_prediction["draw_probability"], 
                "AWAY_WIN": elo_prediction["loss_probability"]
            }        }
        
        if match_context:
            full_context.update(match_context)
            
        draw_prediction = enhance_draw_predictions(full_context, home_team_id, away_team_id)
        
        # Calculate dynamic weights
        weights = self._calculate_dynamic_weights(full_context)
        
        # Combine predictions
        final_probs = {
            "home_win": (
                weights.elo_weight * elo_prediction["win_probability"] +
                (1 - weights.draw_weight) * draw_prediction["probabilities"]["HOME_WIN"]
            ),
            "draw": (
                weights.draw_weight * draw_prediction["probabilities"]["DRAW"]
            ),
            "away_win": (
                weights.elo_weight * elo_prediction["loss_probability"] +
                (1 - weights.draw_weight) * draw_prediction["probabilities"]["AWAY_WIN"]
            )
        }
        
        # Normalize probabilities
        total = sum(final_probs.values())
        final_probs = {k: v/total for k, v in final_probs.items()}
        
        # Add metadata
        prediction = {
            "probabilities": final_probs,
            "models_used": {
                "elo": True,
                "draw": True,
                "league": bool(league_id)
            },
            "model_weights": {
                "elo": weights.elo_weight,
                "draw": weights.draw_weight,
                "league": weights.league_weight
            },
            "metadata": {
                "elo_diff": elo_prediction["elo_diff"],
                "home_elo": elo_prediction["home_elo"],
                "away_elo": elo_prediction["away_elo"]
            }
        }
        
        return prediction
        
    def update_historical_accuracy(self, 
                                 predictions: List[Dict[str, Any]], 
                                 actual_results: List[str]):
        """
        Update historical accuracy metrics for each model component
        
        Args:
            predictions: List of previous predictions
            actual_results: List of actual match results
        """
        if len(predictions) != len(actual_results):
            logger.error("Length mismatch between predictions and results")
            return
            
        elo_correct = 0
        draw_correct = 0
        league_correct = 0
        total = len(predictions)
        
        for pred, result in zip(predictions, actual_results):
            # Check ELO accuracy
            elo_pred = max(pred["models_used"]["elo"], 
                          key=lambda x: pred["probabilities"][x])
            if elo_pred == result:
                elo_correct += 1
                
            # Check draw model accuracy
            if result == "draw" and pred["probabilities"]["draw"] > 0.3:
                draw_correct += 1
            elif result != "draw" and pred["probabilities"]["draw"] < 0.3:
                draw_correct += 1
                
            # Check league-specific accuracy
            if pred["models_used"]["league"]:
                league_pred = max(pred["probabilities"].items(), 
                                key=lambda x: x[1])[0]
                if league_pred == result:
                    league_correct += 1
                    
        # Update historical accuracy
        self.historical_accuracy["elo"] = elo_correct / total if total > 0 else 0
        self.historical_accuracy["draw"] = draw_correct / total if total > 0 else 0
        self.historical_accuracy["league"] = league_correct / total if total > 0 else 0
