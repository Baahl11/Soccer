# fixture_statistics_analyzer.py
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from data import get_fixture_statistics, get_team_statistics

logger = logging.getLogger(__name__)

class FixtureStatisticsAnalyzer:
    """
    Analyzes fixture-level statistics to enhance prediction accuracy.
    Integrates shots, possession, corners, cards, and fouls data.
    """
    
    def __init__(self):
        # Weight factors based on research findings
        self.shot_weight = 0.25
        self.possession_weight = 0.20
        self.corner_weight = 0.15
        self.card_weight = 0.10
        self.foul_weight = 0.05
        
        # League averages for normalization
        self.league_averages = {
            'shots_per_game': 12.5,
            'shots_on_target_per_game': 4.5,
            'possession_percentage': 50.0,
            'corners_per_game': 5.2,
            'cards_per_game': 2.8,
            'fouls_per_game': 11.5
        }
        
    def analyze_fixture_statistics(self, home_team_id: int, away_team_id: int, 
                                 league_id: int, recent_matches: int = 5) -> Dict[str, Any]:
        """
        Analyze fixture-level statistics for both teams to enhance predictions.
        
        Args:
            home_team_id: ID of home team
            away_team_id: ID of away team  
            league_id: ID of league
            recent_matches: Number of recent matches to analyze
            
        Returns:
            Dictionary containing statistical analysis for both teams
        """
        try:
            logger.info(f"Analyzing fixture statistics for teams {home_team_id} vs {away_team_id}")
            
            # Get recent fixture statistics for both teams
            home_stats = self._get_recent_fixture_stats(home_team_id, league_id, recent_matches)
            away_stats = self._get_recent_fixture_stats(away_team_id, league_id, recent_matches)
            
            # Calculate impact metrics
            home_analysis = self._calculate_team_impact_metrics(home_stats, is_home=True)
            away_analysis = self._calculate_team_impact_metrics(away_stats, is_home=False)
            
            # Calculate comparative advantages
            comparative_analysis = self._calculate_comparative_advantages(home_analysis, away_analysis)
            
            return {
                'home': home_analysis,
                'away': away_analysis,
                'comparative': comparative_analysis,
                'confidence_boost': self._calculate_confidence_boost(home_analysis, away_analysis),
                'goal_expectation_modifiers': self._calculate_goal_expectation_modifiers(home_analysis, away_analysis),
                'probability_adjustments': self._calculate_probability_adjustments(comparative_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing fixture statistics: {e}")
            return self._get_default_analysis()
    def _get_recent_fixture_stats(self, team_id: int, league_id: int, recent_matches: int) -> Dict[str, float]:
        """Get recent fixture statistics for a team."""
        try:
            # Get team statistics from data.py with current season
            team_stats = get_team_statistics(team_id, league_id, "2023")  # Use a valid season
            
            # Extract relevant statistics with defaults
            return {
                'shots_per_game': team_stats.get('shots_per_game', self.league_averages['shots_per_game']),
                'shots_on_target_per_game': team_stats.get('shots_on_target_per_game', self.league_averages['shots_on_target_per_game']),
                'possession_percentage': team_stats.get('possession_percentage', self.league_averages['possession_percentage']),
                'corners_per_game': team_stats.get('corners_per_game', self.league_averages['corners_per_game']),
                'cards_per_game': team_stats.get('cards_per_game', self.league_averages['cards_per_game']),
                'fouls_per_game': team_stats.get('fouls_per_game', self.league_averages['fouls_per_game']),
                'goals_per_game': team_stats.get('goals_per_game', 1.2),
                'goals_conceded_per_game': team_stats.get('goals_conceded_per_game', 1.1),
                'passes_completed_per_game': team_stats.get('passes_completed_per_game', 400),
                'passes_attempted_per_game': team_stats.get('passes_attempted_per_game', 500)
            }
            
        except Exception as e:
            logger.warning(f"Error getting fixture stats for team {team_id}: {e}")
            return self._get_default_team_stats()
    
    def _calculate_team_impact_metrics(self, stats: Dict[str, float], is_home: bool = False) -> Dict[str, float]:
        """Calculate impact metrics for a team based on fixture statistics."""
        try:
            # Shot-based metrics
            shot_conversion_rate = stats['goals_per_game'] / max(1, stats['shots_per_game'])
            shot_accuracy = stats['shots_on_target_per_game'] / max(1, stats['shots_per_game'])
            attacking_threat = shot_conversion_rate * shot_accuracy
            
            # Possession-based metrics
            possession_control = stats['possession_percentage'] / 50.0  # Normalize to 50%
            passing_accuracy = stats['passes_completed_per_game'] / max(1, stats['passes_attempted_per_game'])
            possession_efficiency = stats['goals_per_game'] / max(1, stats['possession_percentage'] / 100)
            
            # Defensive metrics
            defensive_solidity = 1 / max(0.5, stats['goals_conceded_per_game'])
            
            # Disciplinary metrics
            card_tendency = stats['cards_per_game'] / self.league_averages['cards_per_game']
            foul_rate = stats['fouls_per_game'] / self.league_averages['fouls_per_game']
            disciplinary_risk = min(2.0, card_tendency)
            
            # Corner generation ability
            corner_generation = stats['corners_per_game'] / self.league_averages['corners_per_game']
            
            # Home advantage adjustment
            home_boost = 1.1 if is_home else 1.0
            
            return {
                'attacking_threat': min(2.0, attacking_threat * home_boost),
                'shot_volume': stats['shots_per_game'] / self.league_averages['shots_per_game'],
                'shot_accuracy': shot_accuracy,
                'possession_control': possession_control,
                'possession_efficiency': possession_efficiency,
                'passing_quality': passing_accuracy,
                'defensive_solidity': defensive_solidity,
                'disciplinary_risk': disciplinary_risk,
                'aggression_level': foul_rate,
                'corner_generation': corner_generation,
                'overall_quality': self._calculate_overall_quality(attacking_threat, possession_control, defensive_solidity)
            }
            
        except Exception as e:
            logger.error(f"Error calculating team impact metrics: {e}")
            return self._get_default_team_metrics()
    
    def _calculate_comparative_advantages(self, home_analysis: Dict[str, float], 
                                       away_analysis: Dict[str, float]) -> Dict[str, Any]:
        """Calculate comparative advantages between teams."""
        try:
            return {
                'shooting_advantage': self._calculate_advantage(
                    home_analysis['attacking_threat'], away_analysis['attacking_threat']
                ),
                'possession_advantage': self._calculate_advantage(
                    home_analysis['possession_control'], away_analysis['possession_control']
                ),
                'defensive_advantage': self._calculate_advantage(
                    home_analysis['defensive_solidity'], away_analysis['defensive_solidity']
                ),
                'discipline_advantage': self._calculate_advantage(
                    1 / max(0.1, home_analysis['disciplinary_risk']), 
                    1 / max(0.1, away_analysis['disciplinary_risk'])
                ),
                'corner_advantage': self._calculate_advantage(
                    home_analysis['corner_generation'], away_analysis['corner_generation']
                ),
                'overall_advantage': self._calculate_advantage(
                    home_analysis['overall_quality'], away_analysis['overall_quality']
                )
            }
            
        except Exception as e:
            logger.error(f"Error calculating comparative advantages: {e}")
            return {'shooting_advantage': 0, 'possession_advantage': 0, 'defensive_advantage': 0,
                   'discipline_advantage': 0, 'corner_advantage': 0, 'overall_advantage': 0}
    
    def _calculate_advantage(self, home_value: float, away_value: float) -> float:
        """Calculate advantage score (-1 to 1) where positive favors home team."""
        if home_value == 0 and away_value == 0:
            return 0
        total = home_value + away_value
        if total == 0:
            return 0
        return (home_value - away_value) / total
    
    def _calculate_confidence_boost(self, home_analysis: Dict[str, float], 
                                  away_analysis: Dict[str, float]) -> float:
        """Calculate confidence boost based on statistical clarity."""
        try:
            # Calculate statistical separation
            key_metrics = ['attacking_threat', 'possession_control', 'defensive_solidity']
            
            separations = []
            for metric in key_metrics:
                home_val = home_analysis.get(metric, 1.0)
                away_val = away_analysis.get(metric, 1.0)
                if home_val + away_val > 0:
                    separation = abs(home_val - away_val) / (home_val + away_val)
                    separations.append(separation)
            
            avg_separation = np.mean(separations) if separations else 0
            
            # Convert to confidence boost (0 to 0.1)
            return min(0.1, avg_separation * 0.2)
            
        except Exception as e:
            logger.error(f"Error calculating confidence boost: {e}")
            return 0.02
    
    def _calculate_goal_expectation_modifiers(self, home_analysis: Dict[str, float], 
                                           away_analysis: Dict[str, float]) -> Dict[str, float]:
        """Calculate goal expectation modifiers based on fixture statistics."""
        try:
            # Home team goal expectation modifier
            home_attacking_boost = 1 + (home_analysis['attacking_threat'] - 1) * 0.2
            home_possession_boost = 1 + (home_analysis['possession_control'] - 1) * 0.1
            away_defensive_impact = 1 / max(0.5, away_analysis['defensive_solidity'])
            
            home_modifier = home_attacking_boost * home_possession_boost * away_defensive_impact
            
            # Away team goal expectation modifier  
            away_attacking_boost = 1 + (away_analysis['attacking_threat'] - 1) * 0.2
            away_possession_boost = 1 + (away_analysis['possession_control'] - 1) * 0.1
            home_defensive_impact = 1 / max(0.5, home_analysis['defensive_solidity'])
            
            away_modifier = away_attacking_boost * away_possession_boost * home_defensive_impact
            
            # Bound the modifiers
            return {
                'home': max(0.5, min(2.0, home_modifier)),
                'away': max(0.5, min(2.0, away_modifier))
            }
            
        except Exception as e:
            logger.error(f"Error calculating goal expectation modifiers: {e}")
            return {'home': 1.0, 'away': 1.0}
    
    def _calculate_probability_adjustments(self, comparative: Dict[str, float]) -> Dict[str, float]:
        """Calculate probability adjustments based on comparative analysis."""
        try:
            # Calculate overall team advantage
            overall_advantage = (
                comparative['shooting_advantage'] * 0.3 +
                comparative['possession_advantage'] * 0.2 +
                comparative['defensive_advantage'] * 0.2 +
                comparative['discipline_advantage'] * 0.15 +
                comparative['corner_advantage'] * 0.15
            )
            
            # Convert to probability adjustments
            home_adjustment = overall_advantage * 0.1  # Max 10% adjustment
            away_adjustment = -overall_advantage * 0.1
            draw_adjustment = -abs(overall_advantage) * 0.05  # Reduce draw probability with clear advantage
            
            return {
                'home': max(-0.1, min(0.1, home_adjustment)),
                'away': max(-0.1, min(0.1, away_adjustment)), 
                'draw': max(-0.05, min(0.05, draw_adjustment))
            }
            
        except Exception as e:
            logger.error(f"Error calculating probability adjustments: {e}")
            return {'home': 0, 'away': 0, 'draw': 0}
    
    def _calculate_overall_quality(self, attacking_threat: float, possession_control: float, 
                                 defensive_solidity: float) -> float:
        """Calculate overall team quality metric."""
        return (attacking_threat * 0.4 + possession_control * 0.3 + defensive_solidity * 0.3)
    
    def _get_default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when calculation fails."""
        default_team_metrics = self._get_default_team_metrics()
        
        return {
            'home': default_team_metrics,
            'away': default_team_metrics,
            'comparative': {
                'shooting_advantage': 0, 'possession_advantage': 0, 'defensive_advantage': 0,
                'discipline_advantage': 0, 'corner_advantage': 0, 'overall_advantage': 0
            },
            'confidence_boost': 0.02,
            'goal_expectation_modifiers': {'home': 1.0, 'away': 1.0},
            'probability_adjustments': {'home': 0, 'away': 0, 'draw': 0}
        }
    
    def _get_default_team_stats(self) -> Dict[str, float]:
        """Return default team statistics."""
        return {
            'shots_per_game': self.league_averages['shots_per_game'],
            'shots_on_target_per_game': self.league_averages['shots_on_target_per_game'],
            'possession_percentage': self.league_averages['possession_percentage'],
            'corners_per_game': self.league_averages['corners_per_game'],
            'cards_per_game': self.league_averages['cards_per_game'],
            'fouls_per_game': self.league_averages['fouls_per_game'],
            'goals_per_game': 1.2,
            'goals_conceded_per_game': 1.1,
            'passes_completed_per_game': 400,
            'passes_attempted_per_game': 500
        }
    
    def _get_default_team_metrics(self) -> Dict[str, float]:
        """Return default team impact metrics."""
        return {
            'attacking_threat': 1.0,
            'shot_volume': 1.0,
            'shot_accuracy': 0.36,
            'possession_control': 1.0,
            'possession_efficiency': 2.4,
            'passing_quality': 0.8,
            'defensive_solidity': 0.9,
            'disciplinary_risk': 1.0,
            'aggression_level': 1.0,
            'corner_generation': 1.0,
            'overall_quality': 1.0
        }

def enhance_goal_predictions(base_home_goals: float, base_away_goals: float, 
                           fixture_stats: Dict[str, Any]) -> tuple[float, float]:
    """
    Enhance goal predictions using fixture statistics.
    
    Args:
        base_home_goals: Base home team goal prediction
        base_away_goals: Base away team goal prediction
        fixture_stats: Fixture statistics analysis
        
    Returns:
        Tuple of enhanced (home_goals, away_goals)
    """
    try:
        modifiers = fixture_stats['goal_expectation_modifiers']
        
        enhanced_home_goals = base_home_goals * modifiers['home']
        enhanced_away_goals = base_away_goals * modifiers['away']
        
        # Ensure reasonable bounds
        enhanced_home_goals = max(0.1, min(6.0, enhanced_home_goals))
        enhanced_away_goals = max(0.1, min(6.0, enhanced_away_goals))
        
        return enhanced_home_goals, enhanced_away_goals
        
    except Exception as e:
        logger.error(f"Error enhancing goal predictions: {e}")
        return base_home_goals, base_away_goals

def enhance_match_probabilities(base_probs: Dict[str, float], 
                              fixture_stats: Dict[str, Any]) -> Dict[str, float]:
    """
    Enhance match outcome probabilities using fixture statistics.
    
    Args:
        base_probs: Base probabilities {'home': float, 'away': float, 'draw': float}
        fixture_stats: Fixture statistics analysis
        
    Returns:
        Enhanced probabilities dictionary
    """
    try:
        adjustments = fixture_stats['probability_adjustments']
        
        # Apply adjustments
        enhanced_home_prob = base_probs['home'] + adjustments['home']
        enhanced_away_prob = base_probs['away'] + adjustments['away']
        enhanced_draw_prob = base_probs['draw'] + adjustments['draw']
        
        # Ensure all probabilities are positive
        enhanced_home_prob = max(0.05, enhanced_home_prob)
        enhanced_away_prob = max(0.05, enhanced_away_prob)
        enhanced_draw_prob = max(0.05, enhanced_draw_prob)
        
        # Normalize to sum to 1
        total = enhanced_home_prob + enhanced_away_prob + enhanced_draw_prob
        
        return {
            'home': enhanced_home_prob / total,
            'away': enhanced_away_prob / total,
            'draw': enhanced_draw_prob / total
        }
        
    except Exception as e:
        logger.error(f"Error enhancing match probabilities: {e}")
        return base_probs
