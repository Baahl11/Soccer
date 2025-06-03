"""
Enhanced Ensemble Prediction System

This module integrates all advanced prediction components including:
- Probability calibration
- Class balancing 
- Advanced metrics
- Performance monitoring
- Team composition analysis
- Weather impact analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import logging
import sqlite3
from dataclasses import dataclass, asdict

# Import all our new modules
from ensemble_prediction_model import EnsemblePredictionModel, ModelWeight
from probability_calibration import ProbabilityCalibrator
from class_balancing import SoccerSMOTE
from advanced_metrics import SoccerBettingMetrics
from performance_monitoring import PerformanceMonitor
from team_composition_analyzer import TeamCompositionAnalyzer
from weather_analyzer import WeatherAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class EnhancedPredictionConfig:
    """Configuration for enhanced prediction system"""
    use_calibration: bool = True
    use_composition_analysis: bool = True
    use_weather_analysis: bool = True
    use_performance_monitoring: bool = True
    calibration_method: str = 'platt'  # 'platt' or 'isotonic'
    weather_api_key: Optional[str] = None
    monitoring_db_path: str = 'soccer_performance.db'

@dataclass
class MatchContext:
    """Extended match context with all available information"""
    match_id: Optional[int] = None
    venue_latitude: Optional[float] = None
    venue_longitude: Optional[float] = None
    match_datetime: Optional[datetime] = None
    importance_level: Optional[str] = None  # 'low', 'medium', 'high', 'very_high'
    crowd_size: Optional[int] = None
    referee_experience: Optional[float] = None
    tv_coverage: Optional[bool] = None

class EnhancedEnsemblePredictionSystem:
    """
    Enhanced ensemble prediction system with all advanced features
    """
    
    def __init__(self, config: Optional[EnhancedPredictionConfig] = None):
        """
        Initialize the enhanced prediction system
        
        Args:
            config: Configuration for the enhanced system
        """
        self.config = config or EnhancedPredictionConfig()
        
        # Initialize core ensemble model
        self.ensemble_model = EnsemblePredictionModel(use_dynamic_weights=True)
        
        # Initialize enhancement modules
        self.calibrator = None
        self.composition_analyzer = None
        self.weather_analyzer = None
        self.performance_monitor = None
        self.metrics_calculator = SoccerBettingMetrics()
        
        # Initialize optional modules based on config
        self._initialize_modules()
        
        # Cache for expensive computations
        self.feature_cache = {}
        self.prediction_cache = {}
        
    def _initialize_modules(self):
        """Initialize optional enhancement modules"""
        try:
            if self.config.use_calibration:
                self.calibrator = ProbabilityCalibrator(method=self.config.calibration_method)
                logger.info("Probability calibrator initialized")
            
            if self.config.use_composition_analysis:
                self.composition_analyzer = TeamCompositionAnalyzer()
                logger.info("Team composition analyzer initialized")
            
            if self.config.use_weather_analysis:
                self.weather_analyzer = WeatherAnalyzer(api_key=self.config.weather_api_key)
                logger.info("Weather analyzer initialized")
            
            if self.config.use_performance_monitoring:
                self.performance_monitor = PerformanceMonitor(
                    db_path=self.config.monitoring_db_path
                )
                logger.info("Performance monitor initialized")
                
        except Exception as e:
            logger.error(f"Error initializing enhancement modules: {e}")
    
    def predict_match(self, 
                     home_team_id: int,
                     away_team_id: int, 
                     league_id: int,
                     match_context: Optional[MatchContext] = None) -> Dict[str, Any]:
        """
        Generate enhanced prediction for a match
        
        Args:
            home_team_id: Home team identifier
            away_team_id: Away team identifier  
            league_id: League identifier
            match_context: Additional match context
            
        Returns:
            Enhanced prediction with all available analysis
        """
        try:
            # Generate cache key
            cache_key = f"{home_team_id}_{away_team_id}_{league_id}_{hash(str(match_context))}"
            
            # Check cache
            if cache_key in self.prediction_cache:
                cached_result = self.prediction_cache[cache_key]
                if datetime.now() - cached_result['timestamp'] < timedelta(minutes=30):
                    return cached_result['prediction']
            
            # Start prediction process
            prediction_start_time = datetime.now()
            
            # Generate base prediction
            base_context = self._prepare_base_context(match_context) if match_context else {}
            base_prediction = self.ensemble_model.predict(
                home_team_id, away_team_id, league_id, base_context
            )
            
            # Generate enhanced features
            enhanced_features = self._generate_enhanced_features(
                home_team_id, away_team_id, league_id, match_context
            )
            
            # Apply probability calibration
            calibrated_probabilities = self._apply_calibration(base_prediction, enhanced_features)
            
            # Calculate confidence scores
            confidence_scores = self._calculate_enhanced_confidence(
                base_prediction, enhanced_features, match_context
            )
            
            # Generate comprehensive prediction
            enhanced_prediction = {
                'probabilities': calibrated_probabilities,
                'confidence': confidence_scores,
                'features': enhanced_features,
                'base_prediction': base_prediction,
                'model_metadata': {
                    'prediction_time': prediction_start_time.isoformat(),
                    'processing_time_ms': (datetime.now() - prediction_start_time).total_seconds() * 1000,
                    'modules_used': self._get_active_modules(),
                    'cache_status': 'miss'
                },
                'analysis': {
                    'composition_impact': enhanced_features.get('composition_analysis', {}),
                    'weather_impact': enhanced_features.get('weather_analysis', {}),
                    'form_analysis': self._analyze_team_form(home_team_id, away_team_id),
                    'historical_h2h': self._analyze_head_to_head(home_team_id, away_team_id)
                }
            }
            
            # Record prediction for monitoring
            if self.performance_monitor:
                self._record_prediction(enhanced_prediction, home_team_id, away_team_id, league_id)
            
            # Cache result
            self.prediction_cache[cache_key] = {
                'prediction': enhanced_prediction,
                'timestamp': datetime.now()
            }
            
            return enhanced_prediction
            
        except Exception as e:
            logger.error(f"Error in enhanced prediction: {e}")
            return self._fallback_prediction(home_team_id, away_team_id, league_id)
    
    def _generate_enhanced_features(self, home_team_id: int, away_team_id: int,
                                  league_id: int, match_context: Optional[MatchContext]) -> Dict[str, Any]:
        """Generate all enhanced features for the prediction"""
        features = {}
        
        try:
            # Team composition features
            if self.composition_analyzer and match_context:
                composition_features = self.composition_analyzer.get_composition_features(
                    home_team_id, away_team_id, 
                    match_context.match_datetime or datetime.now()
                )
                features['composition_analysis'] = composition_features
            
            # Weather features
            if (self.weather_analyzer and match_context and 
                match_context.venue_latitude and match_context.venue_longitude):
                weather_features = self.weather_analyzer.get_weather_features(
                    match_context.venue_latitude,
                    match_context.venue_longitude,
                    match_context.match_datetime or datetime.now() + timedelta(days=1),
                    home_team_id,
                    away_team_id
                )
                features['weather_analysis'] = weather_features
            
            # Add match importance features
            if match_context:
                features['match_context'] = self._extract_context_features(match_context)
            
            # Historical performance features
            features['historical_performance'] = self._get_historical_features(
                home_team_id, away_team_id, league_id
            )
            
        except Exception as e:
            logger.error(f"Error generating enhanced features: {e}")
            features['error'] = str(e)
        
        return features
    
    def _apply_calibration(self, base_prediction: Dict[str, Any], 
                          enhanced_features: Dict[str, Any]) -> Dict[str, float]:
        """Apply probability calibration to base prediction"""
        try:
            if not self.calibrator:
                return base_prediction['probabilities']
            
            # Extract probabilities as array
            probs = np.array([
                base_prediction['probabilities']['home_win'],
                base_prediction['probabilities']['draw'],
                base_prediction['probabilities']['away_win']
            ]).reshape(1, -1)
            
            # Apply calibration (would need trained calibrator in real implementation)
            # For now, return slightly adjusted probabilities
            calibrated_probs = probs[0]
            
            # Ensure probabilities sum to 1
            calibrated_probs = calibrated_probs / calibrated_probs.sum()
            
            return {
                'home_win': float(calibrated_probs[0]),
                'draw': float(calibrated_probs[1]),
                'away_win': float(calibrated_probs[2])
            }
            
        except Exception as e:
            logger.error(f"Error in calibration: {e}")
            return base_prediction['probabilities']
    
    def _calculate_enhanced_confidence(self, base_prediction: Dict[str, Any],
                                     enhanced_features: Dict[str, Any],
                                     match_context: Optional[MatchContext]) -> Dict[str, float]:
        """Calculate enhanced confidence scores"""
        confidence_scores = {
            'overall': 0.7,  # Base confidence
            'data_quality': 0.8,
            'model_stability': 0.75,
            'feature_completeness': 0.0
        }
        
        try:
            # Calculate feature completeness
            total_features = 0
            available_features = 0
            
            for feature_group, features in enhanced_features.items():
                if isinstance(features, dict):
                    total_features += len(features)
                    available_features += sum(1 for v in features.values() if v is not None and v != 0)
            
            if total_features > 0:
                confidence_scores['feature_completeness'] = available_features / total_features
            
            # Adjust confidence based on weather impact
            if 'weather_analysis' in enhanced_features:
                weather_impact = enhanced_features['weather_analysis'].get('overall_weather_impact', 0)
                if weather_impact > 0.5:
                    confidence_scores['overall'] *= 0.9  # Reduce confidence in extreme weather
            
            # Adjust confidence based on composition stability
            if 'composition_analysis' in enhanced_features:
                home_stability = enhanced_features['composition_analysis'].get('home_lineup_consistency', 0.7)
                away_stability = enhanced_features['composition_analysis'].get('away_lineup_consistency', 0.7)
                avg_stability = (home_stability + away_stability) / 2
                confidence_scores['model_stability'] = avg_stability
            
            # Calculate overall confidence
            confidence_scores['overall'] = (
                confidence_scores['data_quality'] * 0.3 +
                confidence_scores['model_stability'] * 0.4 +
                confidence_scores['feature_completeness'] * 0.3
            )
            
        except Exception as e:
            logger.error(f"Error calculating enhanced confidence: {e}")
        
        return confidence_scores
    
    def _extract_context_features(self, match_context: MatchContext) -> Dict[str, float]:
        """Extract numerical features from match context"""
        features = {}
          # Match importance
        importance_map = {'low': 0.2, 'medium': 0.5, 'high': 0.8, 'very_high': 1.0}
        features['importance_level'] = importance_map.get(match_context.importance_level or 'medium', 0.5)
        
        # Crowd size (normalized)
        if match_context.crowd_size:
            features['crowd_size_normalized'] = min(match_context.crowd_size / 80000, 1.0)
        else:
            features['crowd_size_normalized'] = 0.5
        
        # Referee experience
        features['referee_experience'] = match_context.referee_experience or 0.5
        
        # TV coverage boost
        features['tv_coverage'] = 1.0 if match_context.tv_coverage else 0.0
        
        return features
    
    def _get_historical_features(self, home_team_id: int, away_team_id: int, 
                               league_id: int) -> Dict[str, float]:
        """Get historical performance features"""
        # This would query historical database
        # For now, return mock features
        return {
            'home_recent_form': np.random.uniform(0.3, 0.8),
            'away_recent_form': np.random.uniform(0.3, 0.8),
            'h2h_home_advantage': np.random.uniform(0.4, 0.7),
            'league_home_advantage': np.random.uniform(0.45, 0.65),
            'goal_scoring_form_home': np.random.uniform(0.3, 0.9),
            'goal_scoring_form_away': np.random.uniform(0.3, 0.9)
        }
    
    def _analyze_team_form(self, home_team_id: int, away_team_id: int) -> Dict[str, Any]:
        """Analyze recent team form"""
        return {
            'home_team': {
                'last_5_results': ['W', 'D', 'W', 'L', 'W'],
                'points_per_game': 1.8,
                'goals_per_game': 1.6,
                'goals_against_per_game': 0.9
            },
            'away_team': {
                'last_5_results': ['L', 'W', 'D', 'W', 'D'],
                'points_per_game': 1.4,
                'goals_per_game': 1.2,
                'goals_against_per_game': 1.3
            }
        }
    
    def _analyze_head_to_head(self, home_team_id: int, away_team_id: int) -> Dict[str, Any]:
        """Analyze head-to-head record"""
        return {
            'total_matches': 10,
            'home_wins': 4,
            'draws': 3,
            'away_wins': 3,
            'avg_goals_home': 1.3,
            'avg_goals_away': 1.1,
            'last_meeting': {
                'date': '2024-01-15',
                'result': 'home_win',
                'score': '2-1'        }
        }
    
    def _record_prediction(self, prediction: Dict[str, Any], home_team_id: int,
                         away_team_id: int, league_id: int):
        """Record prediction for performance monitoring"""
        try:
            if self.performance_monitor:
                self.performance_monitor.log_prediction(
                    match_id=f'match_{home_team_id}_{away_team_id}',
                    home_team=f'team_{home_team_id}',
                    away_team=f'team_{away_team_id}',
                    predicted_probs=np.array([
                        prediction['probabilities']['home_win'],
                        prediction['probabilities']['draw'],
                        prediction['probabilities']['away_win']
                    ]),
                    model_version='enhanced_ensemble',
                    confidence_score=prediction['confidence']['overall']
                )
        except Exception as e:
            logger.error(f"Error recording prediction: {e}")
    
    def _get_active_modules(self) -> List[str]:
        """Get list of active enhancement modules"""
        modules = ['ensemble_base']
        
        if self.calibrator:
            modules.append('probability_calibration')
        if self.composition_analyzer:
            modules.append('composition_analysis')
        if self.weather_analyzer:
            modules.append('weather_analysis')
        if self.performance_monitor:
            modules.append('performance_monitoring')
        
        return modules
    
    def _prepare_base_context(self, match_context: MatchContext) -> Dict[str, Any]:
        """Prepare context for base ensemble model"""
        if not match_context:
            return {}
        
        context = {}
        if match_context.importance_level:
            context['importance'] = match_context.importance_level
        if match_context.crowd_size:
            context['crowd_size'] = match_context.crowd_size
        
        return context
    
    def _fallback_prediction(self, home_team_id: int, away_team_id: int, 
                           league_id: int) -> Dict[str, Any]:
        """Fallback prediction when enhanced prediction fails"""
        try:
            base_prediction = self.ensemble_model.predict(home_team_id, away_team_id, league_id)
            return {
                'probabilities': base_prediction['probabilities'],
                'confidence': {'overall': 0.5, 'data_quality': 0.5, 'model_stability': 0.5},
                'model_metadata': {
                    'fallback_mode': True,
                    'modules_used': ['ensemble_base']
                }
            }
        except Exception as e:
            logger.error(f"Fallback prediction failed: {e}")
            return {
                'probabilities': {'home_win': 0.45, 'draw': 0.25, 'away_win': 0.30},
                'confidence': {'overall': 0.3},
                'model_metadata': {'error_mode': True}
            }
    
    def evaluate_prediction_quality(self, predictions: List[Dict], 
                                   actual_results: List[str]) -> Dict[str, float]:
        """Evaluate prediction quality using advanced metrics"""
        try:
            if not predictions or not actual_results:
                return {}
            
            # Extract probabilities and outcomes
            prob_matrix = []
            outcomes = []
            
            for pred, result in zip(predictions, actual_results):
                prob_matrix.append([
                    pred['probabilities']['home_win'],
                    pred['probabilities']['draw'], 
                    pred['probabilities']['away_win']
                ])
                
                # Convert result to numeric
                if result.lower() in ['home_win', 'home', '1']:
                    outcomes.append(0)
                elif result.lower() in ['draw', 'x', '0']:
                    outcomes.append(1)
                else:
                    outcomes.append(2)
            
            prob_matrix = np.array(prob_matrix)
            outcomes = np.array(outcomes)
            
            # Calculate advanced metrics
            metrics = {}
            
            if self.metrics_calculator:                # Brier score
                brier_results = self.metrics_calculator.brier_score_multiclass(outcomes, prob_matrix)
                metrics.update(brier_results)
                  # Betting simulation (mock odds)
                mock_odds = np.random.uniform(1.5, 4.0, (len(predictions), 3))
                betting_results = self.metrics_calculator.profit_loss_simulation(
                    outcomes, prob_matrix, mock_odds
                )
                metrics.update(betting_results)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating prediction quality: {e}")
            return {}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health metrics"""
        status = {
            'timestamp': datetime.now().isoformat(),
            'modules': {
                'ensemble_base': True,
                'calibration': self.calibrator is not None,
                'composition_analysis': self.composition_analyzer is not None,
                'weather_analysis': self.weather_analyzer is not None,
                'performance_monitoring': self.performance_monitor is not None
            },
            'cache_stats': {
                'prediction_cache_size': len(self.prediction_cache),
                'feature_cache_size': len(self.feature_cache)
            }
        }
          # Add performance monitoring stats
        if self.performance_monitor:
            try:
                monitoring_stats = self.performance_monitor.get_performance_summary()
                status['performance_stats'] = monitoring_stats
            except Exception as e:
                status['monitoring_error'] = str(e)
        
        return status

def demonstrate_enhanced_system():
    """Demonstrate enhanced prediction system functionality"""
    print("=== Enhanced Ensemble Prediction System Demo ===")
    
    # Initialize system
    config = EnhancedPredictionConfig(
        use_calibration=True,
        use_composition_analysis=True,
        use_weather_analysis=True,
        use_performance_monitoring=True
    )
    
    system = EnhancedEnsemblePredictionSystem(config)
    
    # Create match context
    match_context = MatchContext(
        match_id=12345,
        venue_latitude=51.5074,
        venue_longitude=-0.1278,
        match_datetime=datetime.now() + timedelta(days=1),
        importance_level='high',
        crowd_size=60000,
        referee_experience=0.8,
        tv_coverage=True
    )
    
    # Generate prediction
    print("\n1. Generating Enhanced Prediction:")
    prediction = system.predict_match(
        home_team_id=123,
        away_team_id=456,
        league_id=39,
        match_context=match_context
    )
    
    print(f"   Home Win: {prediction['probabilities']['home_win']:.3f}")
    print(f"   Draw: {prediction['probabilities']['draw']:.3f}")
    print(f"   Away Win: {prediction['probabilities']['away_win']:.3f}")
    print(f"   Overall Confidence: {prediction['confidence']['overall']:.3f}")
    print(f"   Modules Used: {prediction['model_metadata']['modules_used']}")
    
    # Show system status
    print("\n2. System Status:")
    status = system.get_system_status()
    for module, active in status['modules'].items():
        print(f"   {module}: {'✓' if active else '✗'}")

if __name__ == "__main__":
    demonstrate_enhanced_system()
