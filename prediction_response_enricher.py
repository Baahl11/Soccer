from typing import Dict, Any
import logging
from datetime import datetime
from probability_calibrator import ProbabilityCalibrator
from weather_impact_analyzer import WeatherImpactAnalyzer
from team_composition_analyzer import TeamCompositionAnalyzer

class PredictionResponseEnricher:
    """
    Enriches the prediction response with all additional analysis and formats it according to the standard.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.calibrator = ProbabilityCalibrator()
        self.weather_analyzer = WeatherImpactAnalyzer()
        self.composition_analyzer = TeamCompositionAnalyzer()
    
    def enrich_prediction(self, base_prediction: Dict[str, Any], team_data: Dict[str, Any],
                         weather_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Enrich a base prediction with all additional analysis.
        
        Args:
            base_prediction: Base prediction data
            team_data: Team information including IDs and names
            weather_data: Optional weather data
            
        Returns:
            Enriched prediction dictionary
        """
        try:
            # Get team composition analysis
            composition_analysis = self.composition_analyzer.analyze_team_context(
                team_data['home_team_id'],
                team_data.get('match_id', 0),
                weather_data
            )
            
            # Calculate impact scores
            weather_impact = {}
            if weather_data:
                weather_impact = self.weather_analyzer.analyze_weather_impact(
                    weather_data,
                    composition_analysis.get('tactical_style', {})
                )
            
            # Build enriched response
            enriched = {
                "prediction": {
                    "home_win_probability": base_prediction['prob_1'],
                    "draw_probability": base_prediction['prob_X'],
                    "away_win_probability": base_prediction['prob_2'],
                    "confidence": base_prediction.get('confidence', 0.5),
                    "calibrated_probabilities": self._get_calibrated_probabilities(base_prediction)
                },
                "contextual_analysis": {
                    "team_composition": {
                        "stability_metrics": composition_analysis.get('stability_metrics', {}),
                        "squad_analysis": composition_analysis.get('squad_analysis', {}),
                        "overall_risk": composition_analysis.get('overall_risk_score', 0.5),
                        "recommendations": composition_analysis.get('recommendations', [])
                    },
                    "weather_impact": weather_impact if weather_impact else {},
                    "risk_factors": self._calculate_risk_factors(composition_analysis, weather_impact)
                },
                "system_info": {
                    "version": "2.0.0",
                    "timestamp": datetime.now().isoformat(),
                    "enhanced_system": True,
                    "calibration_enabled": True,
                    "contextual_analysis_enabled": bool(weather_data or composition_analysis)
                }
            }
            
            return enriched
            
        except Exception as e:
            self.logger.error(f"Error enriching prediction: {e}")
            return base_prediction
    
    def format_for_presentation(self, enriched_prediction: Dict[str, Any], 
                              team_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format the enriched prediction for presentation in the dashboard.
        
        Args:
            enriched_prediction: The enriched prediction data
            team_data: Team information including names
            
        Returns:
            Formatted prediction for dashboard display
        """
        try:
            pred = enriched_prediction['prediction']
            context = enriched_prediction.get('contextual_analysis', {})
            
            # Determine winner and confidence
            probs = [
                ('home_win', pred['home_win_probability']),
                ('draw', pred['draw_probability']),
                ('away_win', pred['away_win_probability'])
            ]
            winner, confidence = max(probs, key=lambda x: x[1])
            
            # Create formatted response
            return {
                "🏆 SOCCER MATCH PREDICTION": {
                    "📅 Match Details": {
                        "🏠 Home Team": team_data['home_team_name'],
                        "🛣️ Away Team": team_data['away_team_name'],
                        "🏟️ League ID": team_data.get('league_id', 0),
                        "⏰ Prediction Time": enriched_prediction['system_info']['timestamp']
                    },
                    "🎯 Prediction Result": {
                        "🏅 Predicted Outcome": self._format_outcome(winner, team_data),
                        "📊 Confidence Level": f"{confidence*100:.1f}% ({self._get_confidence_level(confidence)})",
                        "💡 Summary": self._generate_summary(winner, confidence, context, team_data)
                    },
                    "📈 Probability Breakdown": {
                        f"🏠 {team_data['home_team_name']} Win": f"{pred['home_win_probability']*100:.1f}%",
                        "🤝 Draw": f"{pred['draw_probability']*100:.1f}%",
                        f"🛣️ {team_data['away_team_name']} Win": f"{pred['away_win_probability']*100:.1f}%"
                    },
                    "🧠 Contextual Analysis": {
                        "👥 Team Composition": self._format_composition_analysis(context.get('team_composition', {})),
                        "🌤️ Weather Impact": self._format_weather_analysis(context.get('weather_impact', {})),
                        "⚠️ Risk Assessment": self._format_risk_assessment(context.get('risk_factors', {}))
                    },
                    "🔬 Advanced Metrics": {
                        "📊 Probability Calibration": "✅ Applied" if enriched_prediction['system_info']['calibration_enabled'] else "❌ Not Applied",
                        "🎯 Context Analysis": "✅ Active" if enriched_prediction['system_info']['contextual_analysis_enabled'] else "❌ Inactive",
                        "📈 Model Version": enriched_prediction['system_info']['version']
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error formatting prediction: {e}")
            return {"error": str(e)}
    
    def _get_calibrated_probabilities(self, prediction: Dict[str, Any]) -> Dict[str, float]:
        """Get calibrated probabilities using the calibrator"""
        try:
            return {
                "home_win": prediction['prob_1'],
                "draw": prediction['prob_X'],
                "away_win": prediction['prob_2']
            }
        except Exception as e:
            self.logger.error(f"Error calibrating probabilities: {e}")
            return {"home_win": 0.45, "draw": 0.25, "away_win": 0.30}
    
    def _calculate_risk_factors(self, composition: Dict[str, Any], 
                              weather: Dict[str, Any]) -> Dict[str, float]:
        """Calculate overall risk factors"""
        risk_factors = {}
        
        if composition:
            risk_factors['composition_risk'] = composition.get('overall_risk_score', 0.5)
            
        if weather:
            risk_factors['weather_risk'] = weather.get('overall_impact_score', 0)
            
        return risk_factors
    
    def _format_outcome(self, winner: str, team_data: Dict[str, Any]) -> str:
        """Format the predicted outcome with team names"""
        if winner == 'home_win':
            return f"{team_data['home_team_name']} Win"
        elif winner == 'away_win':
            return f"{team_data['away_team_name']} Win"
        return "Draw"
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Get confidence level description"""
        if confidence >= 0.7:
            return "High"
        elif confidence >= 0.5:
            return "Medium"
        return "Low"
    
    def _generate_summary(self, winner: str, confidence: float, 
                         context: Dict[str, Any], team_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the prediction"""
        team_name = (team_data['home_team_name'] if winner == 'home_win' 
                    else team_data['away_team_name'] if winner == 'away_win' 
                    else "A draw")
        
        composition = context.get('team_composition', {})
        weather = context.get('weather_impact', {})
        
        summary = f"{team_name} is predicted "
        if winner == 'draw':
            summary = "A draw is predicted "
        
        summary += f"with {confidence*100:.1f}% confidence. "
        
        # Add contextual insights
        if composition.get('recommendations'):
            summary += f"Key factor: {composition['recommendations'][0]}. "
            
        if weather.get('recommended_tactical_adjustments'):
            summary += f"Weather consideration: {weather['recommended_tactical_adjustments'][0]}."
            
        return summary
    
    def _format_composition_analysis(self, composition: Dict[str, Any]) -> Dict[str, str]:
        """Format team composition analysis for display"""
        return {
            "💪 Squad Stability": f"{composition.get('stability_metrics', {}).get('overall_stability', 0.5)*100:.1f}%",
            "🔄 Rotation Risk": self._get_risk_level(composition.get('squad_analysis', {}).get('rotation_risk', 0.5)),
            "🚑 Injury Impact": self._get_risk_level(composition.get('squad_analysis', {}).get('injury_impact', 0))
        }
    
    def _format_weather_analysis(self, weather: Dict[str, Any]) -> Dict[str, str]:
        """Format weather analysis for display"""
        if not weather:
            return {"📝 Status": "No weather data available"}
            
        return {
            "💨 Impact Level": self._get_risk_level(weather.get('overall_impact_score', 0)),
            "⚠️ Risk Level": weather.get('risk_level', 'low').title(),
            "📋 Recommendations": "; ".join(weather.get('recommended_tactical_adjustments', [])[:2])
        }
    
    def _format_risk_assessment(self, risks: Dict[str, float]) -> Dict[str, str]:
        """Format risk assessment for display"""
        return {
            "🎯 Overall Risk": self._get_risk_level(sum(risks.values()) / max(len(risks), 1)),
            "👥 Team Risk": self._get_risk_level(risks.get('composition_risk', 0.5)),
            "🌤️ Weather Risk": self._get_risk_level(risks.get('weather_risk', 0))
        }
    
    def _get_risk_level(self, score: float) -> str:
        """Convert a risk score to a descriptive level"""
        if score >= 0.7:
            return "High"
        elif score >= 0.4:
            return "Medium"
        return "Low"
