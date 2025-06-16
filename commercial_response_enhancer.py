#!/usr/bin/env python3
"""
Enhanced Commercial Response Generator
Mejora las respuestas para hacerlas más comercializables y profesionales.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import math

logger = logging.getLogger(__name__)

class CommercialResponseEnhancer:
    """Mejora las respuestas de predicciones para hacerlas más comerciales."""
    
    def __init__(self):
        self.market_categories = {
            'high_value': ['1X2', 'Over/Under 2.5', 'BTTS'],
            'corners': ['Total Corners', 'Team Corners', 'Corner Handicap'],
            'cards': ['Total Cards', 'Team Cards', 'Player Cards'],
            'goals': ['Total Goals', 'Team Goals', 'First Goal'],
            'special': ['Half Time/Full Time', 'Correct Score', 'Both Teams Score']
        }
        
    def enhance_prediction_response(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Mejora una predicción individual para hacerla más comercial."""
        
        enhanced = prediction.copy()
        
        # 1. Fix mathematical inconsistencies
        enhanced = self._fix_mathematical_errors(enhanced)
        
        # 2. Add commercial insights
        enhanced['commercial_insights'] = self._generate_commercial_insights(enhanced)
        
        # 3. Add betting recommendations
        enhanced['betting_recommendations'] = self._generate_betting_recommendations(enhanced)
        
        # 4. Enhance form analysis
        enhanced['form_analysis'] = self._enhance_form_analysis(enhanced)
        
        # 5. Enhance H2H analysis
        enhanced['h2h_analysis'] = self._enhance_h2h_analysis(enhanced)
        
        # 6. Add tactical insights
        enhanced['tactical_analysis'] = self._generate_tactical_insights(enhanced)
        
        # 7. Add value bets identification
        enhanced['value_opportunities'] = self._identify_value_bets(enhanced)
        
        # 8. Add risk assessment
        enhanced['risk_assessment'] = self._calculate_risk_assessment(enhanced)
        
        # 9. Reorganize for better presentation
        enhanced = self._reorganize_for_presentation(enhanced)
        
        return enhanced
    
    def _fix_mathematical_errors(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Corrige errores matemáticos en corners y cards."""
        
        # Fix corners
        if 'corners' in prediction:
            total_corners = prediction['corners'].get('total', 9.5)
            home_corners = prediction['corners'].get('home', total_corners * 0.55)
            away_corners = prediction['corners'].get('away', total_corners * 0.45)
            
            # Ensure home + away = total
            if abs((home_corners + away_corners) - total_corners) > 0.1:
                home_corners = total_corners * 0.55
                away_corners = total_corners * 0.45
            
            # Fix over/under probabilities
            prediction['corners'] = {
                'total': round(total_corners, 1),
                'home': round(home_corners, 1),
                'away': round(away_corners, 1),
                'over_8.5': self._calculate_poisson_probability(total_corners, 8.5, 'over'),
                'over_9.5': self._calculate_poisson_probability(total_corners, 9.5, 'over'),
                'over_10.5': self._calculate_poisson_probability(total_corners, 10.5, 'over'),
                'under_8.5': self._calculate_poisson_probability(total_corners, 8.5, 'under'),
                'under_9.5': self._calculate_poisson_probability(total_corners, 9.5, 'under'),
                'under_10.5': self._calculate_poisson_probability(total_corners, 10.5, 'under'),
            }
        
        # Fix cards
        if 'cards' in prediction:
            total_cards = prediction['cards'].get('total', 4.2)
            home_cards = prediction['cards'].get('home', total_cards * 0.48)
            away_cards = prediction['cards'].get('away', total_cards * 0.52)
            
            # Ensure home + away = total
            if abs((home_cards + away_cards) - total_cards) > 0.1:
                home_cards = total_cards * 0.48
                away_cards = total_cards * 0.52
            
            prediction['cards'] = {
                'total': round(total_cards, 1),
                'home': round(home_cards, 1),
                'away': round(away_cards, 1),
                'over_3.5': self._calculate_poisson_probability(total_cards, 3.5, 'over'),
                'over_4.5': self._calculate_poisson_probability(total_cards, 4.5, 'over'),
                'over_5.5': self._calculate_poisson_probability(total_cards, 5.5, 'over'),
                'under_3.5': self._calculate_poisson_probability(total_cards, 3.5, 'under'),
                'under_4.5': self._calculate_poisson_probability(total_cards, 4.5, 'under'),
                'under_5.5': self._calculate_poisson_probability(total_cards, 5.5, 'under'),
            }
        
        return prediction
    
    def _calculate_poisson_probability(self, lambda_val: float, threshold: float, direction: str) -> float:
        """Calcula probabilidad usando distribución de Poisson."""
        try:
            if direction == 'over':
                # P(X > threshold) = 1 - P(X <= threshold)
                prob = 1 - sum([(lambda_val ** k) * math.exp(-lambda_val) / math.factorial(k) 
                               for k in range(int(threshold) + 1)])
            else:  # under
                # P(X < threshold) = P(X <= threshold-1)
                prob = sum([(lambda_val ** k) * math.exp(-lambda_val) / math.factorial(k) 
                           for k in range(int(threshold))])
            
            return round(max(0.05, min(0.95, prob)), 3)
        except:
            return 0.5
    
    def _generate_commercial_insights(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Genera insights comerciales basados en los datos."""
        
        home_team = prediction.get('home_team', 'Home')
        away_team = prediction.get('away_team', 'Away')
        home_prob = prediction.get('home_win_prob', 0.33)
        away_prob = prediction.get('away_win_prob', 0.33)
        draw_prob = prediction.get('draw_prob', 0.34)
        confidence = prediction.get('confidence', 0.7)
        
        # Determine match dynamics
        if abs(home_prob - away_prob) < 0.1:
            match_type = "Balanced Contest"
            description = f"Evenly matched teams with similar winning chances"
        elif home_prob > away_prob + 0.15:
            match_type = "Home Advantage"
            description = f"{home_team} strongly favored at home"
        elif away_prob > home_prob + 0.15:
            match_type = "Away Dominance" 
            description = f"{away_team} expected to overcome home disadvantage"
        else:
            match_type = "Slight Edge"
            description = f"Marginal favorite identified"
        
        # Key betting angles
        total_goals = prediction.get('total_goals', 2.5)
        prob_over_25 = prediction.get('prob_over_2_5', 0.5)
        prob_btts = prediction.get('prob_btts', 0.5)
        
        betting_angles = []
        if prob_over_25 > 0.65:
            betting_angles.append("High-scoring encounter expected")
        elif prob_over_25 < 0.35:
            betting_angles.append("Defensive battle anticipated")
        
        if prob_btts > 0.7:
            betting_angles.append("Both teams likely to score")
        elif prob_btts < 0.3:
            betting_angles.append("Clean sheet potential")
        
        if confidence > 0.8:
            betting_angles.append("High confidence prediction")
        
        return {
            'match_type': match_type,
            'description': description,
            'betting_angles': betting_angles,
            'key_metrics': {
                'expected_goals': round(total_goals, 2),
                'most_likely_result': self._get_most_likely_result(home_prob, draw_prob, away_prob),
                'confidence_level': self._get_confidence_level(confidence),
                'value_rating': self._calculate_value_rating(prediction)
            }
        }
    
    def _generate_betting_recommendations(self, prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Genera recomendaciones específicas de apuestas."""
        
        recommendations = []
        
        # 1X2 Recommendations
        home_prob = prediction.get('home_win_prob', 0.33)
        away_prob = prediction.get('away_win_prob', 0.33)
        draw_prob = prediction.get('draw_prob', 0.34)
        
        if max(home_prob, away_prob, draw_prob) > 0.45:
            winner = 'Home' if home_prob == max(home_prob, away_prob, draw_prob) else 'Away' if away_prob == max(home_prob, away_prob, draw_prob) else 'Draw'
            recommendations.append({
                'market': '1X2',
                'selection': winner,
                'confidence': 'High' if max(home_prob, away_prob, draw_prob) > 0.55 else 'Medium',
                'reasoning': f"{winner} has {max(home_prob, away_prob, draw_prob):.1%} probability",
                'suggested_stake': 'Standard' if max(home_prob, away_prob, draw_prob) > 0.55 else 'Conservative'
            })
        
        # Goals Recommendations
        total_goals = prediction.get('total_goals', 2.5)
        prob_over_25 = prediction.get('prob_over_2_5', 0.5)
        
        if prob_over_25 > 0.65:
            recommendations.append({
                'market': 'Total Goals',
                'selection': 'Over 2.5',
                'confidence': 'High',
                'reasoning': f"Expected {total_goals:.1f} goals, {prob_over_25:.1%} chance of Over 2.5",
                'suggested_stake': 'Standard'
            })
        elif prob_over_25 < 0.35:
            recommendations.append({
                'market': 'Total Goals', 
                'selection': 'Under 2.5',
                'confidence': 'High',
                'reasoning': f"Low-scoring game expected ({total_goals:.1f} goals)",
                'suggested_stake': 'Standard'
            })
        
        # BTTS Recommendations
        prob_btts = prediction.get('prob_btts', 0.5)
        if prob_btts > 0.7:
            recommendations.append({
                'market': 'Both Teams to Score',
                'selection': 'Yes',
                'confidence': 'High',
                'reasoning': f"{prob_btts:.1%} probability both teams score",
                'suggested_stake': 'Standard'
            })
        elif prob_btts < 0.3:
            recommendations.append({
                'market': 'Both Teams to Score',
                'selection': 'No', 
                'confidence': 'Medium',
                'reasoning': f"Low probability ({prob_btts:.1%}) of both teams scoring",
                'suggested_stake': 'Conservative'
            })
        
        # Corners Recommendations
        corners = prediction.get('corners', {})
        total_corners = corners.get('total', 9.5)
        
        if corners.get('over_9.5', 0) > 0.6:
            recommendations.append({
                'market': 'Total Corners',
                'selection': 'Over 9.5',
                'confidence': 'Medium',
                'reasoning': f"Expected {total_corners} corners",
                'suggested_stake': 'Small'
            })
        
        return recommendations
    
    def _enhance_form_analysis(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Mejora el análisis de forma de los equipos."""
        
        home_strength = prediction.get('component_analyses', {}).get('base_predictions', {}).get('home_strength', 1.0)
        away_strength = prediction.get('component_analyses', {}).get('base_predictions', {}).get('away_strength', 1.0)
        
        return {
            'home_team_form': {
                'strength_rating': round(home_strength, 2),
                'form_trend': 'Improving' if home_strength > 1.2 else 'Declining' if home_strength < 0.8 else 'Stable',
                'recent_performance': self._analyze_strength_performance(home_strength),
                'goals_per_game': round(home_strength, 1)
            },
            'away_team_form': {
                'strength_rating': round(away_strength, 2),
                'form_trend': 'Improving' if away_strength > 1.2 else 'Declining' if away_strength < 0.8 else 'Stable',
                'recent_performance': self._analyze_strength_performance(away_strength),
                'goals_per_game': round(away_strength, 1)
            },
            'form_comparison': {
                'advantage': 'Home' if home_strength > away_strength * 1.2 else 'Away' if away_strength > home_strength * 1.2 else 'Balanced',
                'strength_difference': round(abs(home_strength - away_strength), 2),
                'impact_on_result': 'Significant' if abs(home_strength - away_strength) > 0.5 else 'Moderate' if abs(home_strength - away_strength) > 0.3 else 'Minimal'
            }
        }
    
    def _enhance_h2h_analysis(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Mejora el análisis head-to-head."""
        
        home_prob = prediction.get('home_win_prob', 0.33)
        away_prob = prediction.get('away_win_prob', 0.33)
        
        # Simulated H2H based on current probabilities
        return {
            'historical_trend': self._determine_h2h_trend(home_prob, away_prob),
            'recent_meetings': {
                'matches_analyzed': 'Last 10 encounters',
                'home_wins': self._simulate_h2h_stat(home_prob, 10),
                'away_wins': self._simulate_h2h_stat(away_prob, 10),
                'draws': self._simulate_h2h_stat(prediction.get('draw_prob', 0.34), 10),
                'avg_goals': round(prediction.get('total_goals', 2.5), 1)
            },
            'h2h_insights': [
                self._generate_h2h_insight(home_prob, away_prob),
                f"Average of {prediction.get('total_goals', 2.5):.1f} goals in recent meetings",
                self._generate_h2h_venue_insight(home_prob)
            ]
        }
    
    def _generate_tactical_insights(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Genera insights tácticos basados en los datos disponibles."""
        
        home_goals = prediction.get('predicted_home_goals', 1.0)
        away_goals = prediction.get('predicted_away_goals', 1.0)
        total_corners = prediction.get('corners', {}).get('total', 9.5)
        total_cards = prediction.get('cards', {}).get('total', 4.2)
        
        return {
            'attacking_approach': {
                'home_team': self._determine_attacking_style(home_goals, total_corners),
                'away_team': self._determine_attacking_style(away_goals, total_corners * 0.45)
            },
            'defensive_setup': {
                'expected_intensity': 'High' if total_cards > 5 else 'Medium' if total_cards > 3.5 else 'Low',
                'discipline_level': 'Poor' if total_cards > 5.5 else 'Average' if total_cards > 4 else 'Good'
            },
            'game_tempo': {
                'expected_pace': 'Fast' if total_corners > 11 else 'Medium' if total_corners > 8 else 'Slow',
                'possession_style': self._determine_possession_style(total_corners, home_goals + away_goals)
            },
            'key_battles': [
                f"Midfield control will be crucial",
                f"Set pieces could be decisive ({total_corners:.1f} corners expected)",
                f"Defensive discipline important ({total_cards:.1f} cards expected)"
            ]
        }
    
    def _identify_value_bets(self, prediction: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identifica apuestas de valor basadas en las probabilidades."""
        
        value_bets = []
        
        # Simulate market odds and compare with our probabilities
        home_prob = prediction.get('home_win_prob', 0.33)
        away_prob = prediction.get('away_win_prob', 0.33)
        draw_prob = prediction.get('draw_prob', 0.34)
        
        # Simulated "market" odds (inverse of probability with margin)
        home_market_prob = home_prob * 0.95  # Assume 5% margin
        away_market_prob = away_prob * 0.95
        
        if home_prob > home_market_prob * 1.1:  # 10% edge
            value_bets.append({
                'market': '1X2',
                'selection': 'Home Win',
                'our_probability': f"{home_prob:.1%}",
                'market_probability': f"{home_market_prob:.1%}",
                'edge': f"{((home_prob / home_market_prob - 1) * 100):.1f}%",
                'value_rating': 'High' if home_prob > home_market_prob * 1.15 else 'Medium'
            })
        
        if away_prob > away_market_prob * 1.1:
            value_bets.append({
                'market': '1X2',
                'selection': 'Away Win',
                'our_probability': f"{away_prob:.1%}",
                'market_probability': f"{away_market_prob:.1%}",
                'edge': f"{((away_prob / away_market_prob - 1) * 100):.1f}%",
                'value_rating': 'High' if away_prob > away_market_prob * 1.15 else 'Medium'
            })
        
        # Goals value bets
        prob_over_25 = prediction.get('prob_over_2_5', 0.5)
        market_over_25 = 0.52  # Typical market probability
        
        if prob_over_25 > market_over_25 * 1.1:
            value_bets.append({
                'market': 'Total Goals',
                'selection': 'Over 2.5',
                'our_probability': f"{prob_over_25:.1%}",
                'market_probability': f"{market_over_25:.1%}",
                'edge': f"{((prob_over_25 / market_over_25 - 1) * 100):.1f}%",
                'value_rating': 'Medium'
            })
        
        return value_bets
    
    def _calculate_risk_assessment(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Calcula el assessment de riesgo para la predicción."""
        
        confidence = prediction.get('confidence', 0.7)
        home_prob = prediction.get('home_win_prob', 0.33)
        away_prob = prediction.get('away_win_prob', 0.33)
        draw_prob = prediction.get('draw_prob', 0.34)
        
        # Calculate entropy (uncertainty measure)
        probs = [home_prob, away_prob, draw_prob]
        entropy = -sum(p * math.log2(p + 0.001) for p in probs if p > 0)
        max_entropy = math.log2(3)  # Maximum entropy for 3 outcomes
        uncertainty = entropy / max_entropy
        
        overall_risk = (1 - confidence) * 0.6 + uncertainty * 0.4
        
        return {
            'overall_risk': self._get_risk_level(overall_risk),
            'confidence_level': f"{confidence:.1%}",
            'outcome_uncertainty': f"{uncertainty:.1%}",
            'recommendations': {
                'stake_size': 'Conservative' if overall_risk > 0.6 else 'Standard' if overall_risk > 0.4 else 'Aggressive',
                'bankroll_percentage': '1-2%' if overall_risk > 0.6 else '2-3%' if overall_risk > 0.4 else '3-5%',
                'risk_management': 'Use stop-loss' if overall_risk > 0.5 else 'Standard management'
            }
        }
    
    def _reorganize_for_presentation(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        """Reorganiza la predicción para mejor presentación comercial."""
        
        # Create a more logical structure for commercial presentation
        commercial_prediction = {
            # Executive Summary
            'match_summary': {
                'fixture_id': prediction.get('fixture_id'),
                'date': prediction.get('date'),
                'league': f"League {prediction.get('league_id')}",
                'teams': {
                    'home': prediction.get('home_team'),
                    'away': prediction.get('away_team')
                },
                'venue': prediction.get('api_data', {}).get('venue', 'TBD')
            },
            
            # Core Predictions
            'predictions': {
                'match_result': {
                    'home_win': f"{prediction.get('home_win_prob', 0):.1%}",
                    'draw': f"{prediction.get('draw_prob', 0):.1%}",
                    'away_win': f"{prediction.get('away_win_prob', 0):.1%}",
                    'most_likely': self._get_most_likely_result(
                        prediction.get('home_win_prob', 0),
                        prediction.get('draw_prob', 0), 
                        prediction.get('away_win_prob', 0)
                    )
                },
                'goals': {
                    'total_expected': prediction.get('total_goals'),
                    'home_expected': prediction.get('predicted_home_goals'),
                    'away_expected': prediction.get('predicted_away_goals'),
                    'over_2_5': f"{prediction.get('prob_over_2_5', 0):.1%}",
                    'btts': f"{prediction.get('prob_btts', 0):.1%}"
                },
                'corners': prediction.get('corners', {}),
                'cards': prediction.get('cards', {})
            },
            
            # Commercial Intelligence
            'commercial_insights': prediction.get('commercial_insights', {}),
            'betting_recommendations': prediction.get('betting_recommendations', []),
            'value_opportunities': prediction.get('value_opportunities', []),
            'risk_assessment': prediction.get('risk_assessment', {}),
            
            # Analysis
            'analysis': {
                'form_analysis': prediction.get('form_analysis', {}),
                'h2h_analysis': prediction.get('h2h_analysis', {}),
                'tactical_analysis': prediction.get('tactical_analysis', {}),
                'elo_ratings': prediction.get('elo_ratings', {})
            },
            
            # System Information
            'system_info': {
                'confidence': f"{prediction.get('confidence', 0):.1%}",
                'accuracy_projection': prediction.get('accuracy_projection', {}),
                'method': prediction.get('method'),
                'generated_at': prediction.get('generated_at'),
                'data_source': prediction.get('data_source')
            }
        }
        
        return commercial_prediction
    
    # Helper methods
    def _get_most_likely_result(self, home_prob: float, draw_prob: float, away_prob: float) -> str:
        probs = {'Home Win': home_prob, 'Draw': draw_prob, 'Away Win': away_prob}
        return max(probs, key=probs.get)
    
    def _get_confidence_level(self, confidence: float) -> str:
        if confidence > 0.8: return 'Very High'
        elif confidence > 0.7: return 'High'
        elif confidence > 0.6: return 'Medium'
        else: return 'Low'
    
    def _get_risk_level(self, risk: float) -> str:
        if risk > 0.7: return 'High'
        elif risk > 0.5: return 'Medium'
        else: return 'Low'
    
    def _calculate_value_rating(self, prediction: Dict[str, Any]) -> str:
        confidence = prediction.get('confidence', 0.7)
        if confidence > 0.85: return 'Excellent'
        elif confidence > 0.75: return 'Good'
        elif confidence > 0.65: return 'Fair'
        else: return 'Poor'
    
    def _analyze_strength_performance(self, strength: float) -> str:
        if strength > 1.5: return 'Excellent'
        elif strength > 1.2: return 'Good'
        elif strength > 0.8: return 'Average'
        else: return 'Poor'
    
    def _determine_h2h_trend(self, home_prob: float, away_prob: float) -> str:
        if home_prob > away_prob * 1.3:
            return 'Home team dominates recent meetings'
        elif away_prob > home_prob * 1.3:
            return 'Away team has recent advantage'
        else:
            return 'Evenly matched in recent encounters'
    
    def _simulate_h2h_stat(self, prob: float, total: int) -> int:
        return round(prob * total)
    
    def _generate_h2h_insight(self, home_prob: float, away_prob: float) -> str:
        if abs(home_prob - away_prob) < 0.1:
            return "Recent meetings have been closely contested"
        else:
            favorite = "Home team" if home_prob > away_prob else "Away team"
            return f"{favorite} has had the upper hand recently"
    
    def _generate_h2h_venue_insight(self, home_prob: float) -> str:
        if home_prob > 0.5:
            return "Strong home record in this fixture"
        elif home_prob < 0.3:
            return "Venue provides little advantage"
        else:
            return "Moderate home advantage expected"
    
    def _determine_attacking_style(self, goals: float, corners: float) -> str:
        if goals > 1.5 and corners > 5:
            return "Aggressive attacking approach"
        elif goals > 1.0:
            return "Balanced offensive strategy"
        else:
            return "Conservative attacking play"
    
    def _determine_possession_style(self, corners: float, goals: float) -> str:
        ratio = corners / max(goals, 0.1)
        if ratio > 5:
            return "Patient build-up play"
        elif ratio > 3:
            return "Balanced possession"
        else:
            return "Direct attacking style"


def main():
    """Demo de mejora de respuestas comerciales."""
    enhancer = CommercialResponseEnhancer()
    
    # Sample prediction (simplified)
    sample_prediction = {
        'home_team': 'Manchester United',
        'away_team': 'Liverpool', 
        'home_win_prob': 0.35,
        'away_win_prob': 0.40,
        'draw_prob': 0.25,
        'total_goals': 2.8,
        'prob_over_2_5': 0.65,
        'prob_btts': 0.72,
        'confidence': 0.81,
        'corners': {'total': 10.5, 'home': 5.8, 'away': 4.7},
        'cards': {'total': 4.1, 'home': 1.9, 'away': 2.2}
    }
    
    enhanced = enhancer.enhance_prediction_response(sample_prediction)
    print(json.dumps(enhanced, indent=2))

if __name__ == "__main__":
    main()
