import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MomentumFeatures:
    """
    Advanced momentum features with multiple time windows to capture trends
    """
    def __init__(self):
        self.windows = [3, 5, 10]  # Multiple time windows for momentum analysis
        self.trend_smoothing = 0.8  # Smoothing factor for trend calculations
        self.logger = logging.getLogger(__name__)
        
    def calculate_momentum_features(self, matches: List, 
                                  team_id: int, 
                                  match_date: str,
                                  is_home: bool = True) -> Dict[str, float]:
        """
        Calculate advanced momentum features using multiple time windows
        
        Args:
            matches: List of previous matches for the team
            team_id: Team ID
            match_date: Date of the match to predict
            is_home: Whether the team plays at home
            
        Returns:
            Dictionary containing momentum features
        """
        prefix = 'home' if is_home else 'away'
        features = {}
        
        try:
            # Filter matches before the current match date
            current_dt = datetime.strptime(match_date, "%Y-%m-%d")
            recent_matches = sorted(
                [m for m in matches if 
                 datetime.strptime(m.get('date', '1900-01-01'), "%Y-%m-%d") < current_dt and
                 (m.get('home_team_id') == team_id or m.get('away_team_id') == team_id)],
                key=lambda x: x['date'],
                reverse=True
            )
            
            if not recent_matches:
                return self._get_default_momentum_features(prefix)
            
            # Calculate momentum across different windows
            for window in self.windows:
                window_matches = recent_matches[:min(window, len(recent_matches))]
                
                # Basic momentum metrics
                win_ratio = self._calculate_win_ratio(window_matches, team_id)
                scoring_ratio = self._calculate_scoring_ratio(window_matches, team_id)
                
                # Weighted form trend that gives more importance to recent matches
                form_trend = self._calculate_weighted_form_trend(window_matches, team_id)
                
                # Performance against expectation
                vs_expectation = self._calculate_vs_expectation(window_matches, team_id)
                
                # Goal difference trend
                goal_diff_trend = self._calculate_goal_diff_trend(window_matches, team_id)
                
                # Home/Away specific performance (depending on where they'll play next)
                venue_performance = self._calculate_venue_performance(window_matches, team_id, is_home)
                
                # Store features with window size in name
                features.update({
                    f"{prefix}_win_ratio_{window}": win_ratio,
                    f"{prefix}_scoring_ratio_{window}": scoring_ratio,
                    f"{prefix}_form_trend_{window}": form_trend,
                    f"{prefix}_vs_expectation_{window}": vs_expectation, 
                    f"{prefix}_goal_diff_trend_{window}": goal_diff_trend,
                    f"{prefix}_venue_performance_{window}": venue_performance
                })
            
            # Calculate aggregate momentum metrics
            features.update(self._calculate_aggregate_metrics(recent_matches, team_id, prefix))
            
            # Calculate momentum acceleration (rate of change in form)
            if len(recent_matches) >= max(self.windows):
                features[f"{prefix}_momentum_acceleration"] = self._calculate_momentum_acceleration(recent_matches, team_id)
            else:
                features[f"{prefix}_momentum_acceleration"] = 0.0
                
            # Calculate consistency metric
            features[f"{prefix}_result_consistency"] = self._calculate_result_consistency(recent_matches, team_id)
                
        except Exception as e:
            self.logger.error(f"Error calculating momentum features: {e}")
            return self._get_default_momentum_features(prefix)
            
        return features
        
    def _get_default_momentum_features(self, prefix: str) -> Dict[str, float]:
        """Get default momentum features when data is unavailable"""
        features = {}
        
        # Default values for each window
        for window in self.windows:
            features.update({
                f"{prefix}_win_ratio_{window}": 0.5,
                f"{prefix}_scoring_ratio_{window}": 0.5,
                f"{prefix}_form_trend_{window}": 0.0,
                f"{prefix}_vs_expectation_{window}": 0.0,
                f"{prefix}_goal_diff_trend_{window}": 0.0,
                f"{prefix}_venue_performance_{window}": 0.5
            })
        
        # Default aggregate metrics
        features.update({
            f"{prefix}_momentum_index": 0.5,
            f"{prefix}_momentum_acceleration": 0.0,
            f"{prefix}_result_consistency": 0.5,
            f"{prefix}_momentum_strength": 0.5,
            f"{prefix}_momentum_quality": 0.5,
            f"{prefix}_momentum_volatility": 0.5
        })
        
        return features
    
    def _calculate_win_ratio(self, matches: List[Dict[str, Any]], team_id: int) -> float:
        """Calculate win ratio from match list"""
        if not matches:
            return 0.5
        
        wins = 0
        for match in matches:
            if self._is_win(match, team_id):
                wins += 1
        
        return wins / len(matches)
    
    def _calculate_scoring_ratio(self, matches: List[Dict[str, Any]], team_id: int) -> float:
        """Calculate ratio of matches where team scored at least one goal"""
        if not matches:
            return 0.5
        
        scoring_matches = 0
        for match in matches:
            goals = self._get_team_goals(match, team_id)
            if goals > 0:
                scoring_matches += 1
        
        return scoring_matches / len(matches)
    
    def _calculate_weighted_form_trend(self, matches: List[Dict[str, Any]], team_id: int) -> float:
        """
        Calculate form trend with exponential weighting for recency
        
        Returns:
            Float: Trend value between -1 and 1, where positive values indicate improving form
        """
        if not matches:
            return 0.0
        
        # Convert results to numeric values (win=1, draw=0.5, loss=0)
        results = []
        for match in matches:
            if self._is_win(match, team_id):
                results.append(1.0)
            elif self._is_draw(match, team_id):
                results.append(0.5)
            else:
                results.append(0.0)
                
        # Apply exponential weights to emphasize recent matches
        weights = np.exp(np.linspace(0, 1, len(results))) - 1
        weights = weights / np.sum(weights)
        
        # Calculate weighted average
        trend = np.sum(weights * np.array(results)) - 0.5
        return float(min(1.0, max(-1.0, trend * 2)))  # Normalize to [-1, 1]
    
    def _calculate_vs_expectation(self, matches: List[Dict[str, Any]], team_id: int) -> float:
        """
        Calculate how team is performing vs expectations (based on pre-match odds or xG)
        
        Returns:
            Float: Value between -1 and 1, where positive values indicate overperformance
        """
        if not matches:
            return 0.0
        
        performances = []
        for match in matches:
            # Check if we have expected goals data
            if 'xg_for' in match and 'xg_against' in match:
                actual_goals = self._get_team_goals(match, team_id)
                conceded_goals = self._get_opponent_goals(match, team_id)
                
                # Get expected goals values
                team_is_home = match.get('home_team_id') == team_id
                if team_is_home:
                    xg_for = float(match.get('home_xg', match.get('xg_for', 0)))
                    xg_against = float(match.get('away_xg', match.get('xg_against', 0)))
                else:
                    xg_for = float(match.get('away_xg', match.get('xg_for', 0)))
                    xg_against = float(match.get('home_xg', match.get('xg_against', 0)))
                
                # Calculate performance vs expectation
                goal_diff = actual_goals - conceded_goals
                expected_diff = xg_for - xg_against
                performances.append(goal_diff - expected_diff)
            elif 'pre_match_odds' in match:
                # Alternative: use pre-match odds
                odds = match.get('pre_match_odds', {})
                if team_is_home and 'home_win' in odds:
                    win_prob = 1 / float(odds['home_win'])
                elif not team_is_home and 'away_win' in odds:
                    win_prob = 1 / float(odds['away_win'])
                else:
                    continue
                
                # Calculate overperformance (1 for win when unlikely, -1 for loss when likely)
                if self._is_win(match, team_id):
                    performances.append(1 - win_prob)
                else:
                    performances.append(-win_prob)
        
        if not performances:
            return 0.0
            
        # Calculate weighted average with more weight on recent matches
        weights = np.exp(np.linspace(0, 1, len(performances))) - 1
        weights = weights / np.sum(weights)
        
        return float(min(1.0, max(-1.0, np.sum(weights * np.array(performances)))))
    
    def _calculate_goal_diff_trend(self, matches: List[Dict[str, Any]], team_id: int) -> float:
        """
        Calculate trend in goal difference
        
        Returns:
            Float: Trend value between -1 and 1
        """
        if not matches:
            return 0.0
        
        goal_diffs = []
        for match in matches:
            goal_diff = self._get_team_goals(match, team_id) - self._get_opponent_goals(match, team_id)
            goal_diffs.append(goal_diff)
        
        # Apply exponential weights
        weights = np.exp(np.linspace(0, 1, len(goal_diffs))) - 1
        weights = weights / np.sum(weights)
        
        # Calculate weighted average and normalize
        avg_diff = np.sum(weights * np.array(goal_diffs))
        return float(min(1.0, max(-1.0, avg_diff / 3.0)))  # Normalize to [-1, 1]
    
    def _calculate_venue_performance(self, matches: List[Dict[str, Any]], team_id: int, is_home: bool) -> float:
        """
        Calculate performance specific to venue type (home/away)
        
        Returns:
            Float: Performance ratio from 0 to 1
        """
        venue_matches = []
        for match in matches:
            match_is_home = match.get('home_team_id') == team_id
            if match_is_home == is_home:
                venue_matches.append(match)
        
        if not venue_matches:
            return 0.5
        
        # Calculate win ratio at this venue type
        venue_wins = sum(1 for m in venue_matches if self._is_win(m, team_id))
        return venue_wins / len(venue_matches)
    
    def _calculate_momentum_acceleration(self, matches: List[Dict[str, Any]], team_id: int) -> float:
        """
        Calculate momentum acceleration (whether form is improving or declining)
        by comparing shorter and longer term form windows
        
        Returns:
            Float: Value between -1 (declining) and 1 (improving)
        """
        if len(matches) < max(self.windows):
            return 0.0
        
        # Get form from different windows
        short_window = min(self.windows)
        long_window = max(self.windows)
        
        short_term_matches = matches[:short_window]
        long_term_matches = matches[:long_window]
        
        # Calculate weighted form for each window
        short_term_form = self._calculate_weighted_form_trend(short_term_matches, team_id)
        long_term_form = self._calculate_weighted_form_trend(long_term_matches, team_id)
        
        # Calculate acceleration
        acceleration = short_term_form - long_term_form
        return float(min(1.0, max(-1.0, acceleration * 2)))  # Scale to [-1, 1]
    
    def _calculate_result_consistency(self, matches: List[Dict[str, Any]], team_id: int) -> float:
        """
        Calculate result consistency 
        
        Returns:
            Float: Value between 0 (inconsistent) and 1 (consistent)
        """
        if len(matches) < 3:
            return 0.5
        
        # Convert results to numeric values
        results = []
        for match in matches:
            if self._is_win(match, team_id):
                results.append(1.0)
            elif self._is_draw(match, team_id):
                results.append(0.5)
            else:
                results.append(0.0)
          # Calculate standard deviation and convert to consistency score
        std_dev = float(np.std(results)) if len(results) > 1 else 0.0
        consistency = 1.0 - min(1.0, std_dev * 2)
        return float(consistency)
    
    def _calculate_aggregate_metrics(self, matches: List[Dict[str, Any]], team_id: int, prefix: str) -> Dict[str, float]:
        """
        Calculate aggregate momentum metrics from raw results
        
        Returns:
            Dictionary of momentum metrics
        """
        # Extract basic results
        results = []
        goals_for = []
        goals_against = []
        
        for match in matches:
            # Add result
            if self._is_win(match, team_id):
                results.append(1.0)
            elif self._is_draw(match, team_id):
                results.append(0.5)
            else:
                results.append(0.0)
            
            # Add goals
            goals_for.append(self._get_team_goals(match, team_id))
            goals_against.append(self._get_opponent_goals(match, team_id))
        
        # Calculate point trend with smoothing
        window = min(5, len(results))
        result_trends = self._calculate_trends(results, window)
        
        # Calculate goal scoring trend
        goals_for_trends = self._calculate_trends(goals_for, window)
        
        # Calculate defensive trend (inverted)
        goals_against_trends = [-x for x in self._calculate_trends(goals_against, window)]
          # Overall momentum strength (weighted average of trends)
        result_trends_clean = [x for x in result_trends if not np.isnan(x)]
        goals_for_trends_clean = [x for x in goals_for_trends if not np.isnan(x)]
        goals_against_trends_clean = [x for x in goals_against_trends if not np.isnan(x)]
        
        momentum_strength = (
            (0.5 * np.mean(result_trends_clean) if result_trends_clean else 0.0) +
            (0.3 * np.mean(goals_for_trends_clean) if goals_for_trends_clean else 0.0) +
            (0.2 * np.mean(goals_against_trends_clean) if goals_against_trends_clean else 0.0)
        )
        
        # Momentum quality - higher for consistent good results
        momentum_quality = self._calculate_momentum_quality(results, goals_for, goals_against)
        
        # Momentum volatility - how rapidly form changes
        momentum_volatility = self._calculate_momentum_volatility(results)
        
        # Combined momentum index (overall measure)
        momentum_index = (
            (0.5 * min(1.0, max(0.0, momentum_strength + 0.5))) +  # Transform from [-1,1] to [0,1]
            (0.3 * momentum_quality) + 
            (0.2 * (1.0 - momentum_volatility))  # Lower volatility is better
        )
        
        return {
            f"{prefix}_momentum_index": float(momentum_index),
            f"{prefix}_momentum_strength": float(min(1.0, max(0.0, momentum_strength + 0.5))),
            f"{prefix}_momentum_quality": float(momentum_quality),
            f"{prefix}_momentum_volatility": float(momentum_volatility)
        }
    
    def _calculate_trends(self, values: List[float], window: int) -> List[float]:
        """
        Calculate local trends in a sequence using rolling differences
        
        Returns:
            List of local trend values
        """
        if len(values) < window:
            return [0.0]
        
        trends = []
        for i in range(len(values) - window + 1):
            window_values = values[i:i+window]
            
            # Simple linear trend
            x = np.arange(len(window_values))
            y = np.array(window_values)
            
            # Calculate trend (linear regression slope)
            if len(np.unique(y)) == 1:  # All values are the same
                trends.append(0.0)
            else:
                # Calculate slope using linear regression
                try:
                    slope = np.polyfit(x, y, 1)[0]
                    trends.append(slope)
                except:
                    trends.append(0.0)
        
        # Normalize trends
        if trends:
            max_abs_trend = max(abs(min(trends)), abs(max(trends)))
            if max_abs_trend > 0:
                trends = [t / max_abs_trend for t in trends]
        
        return trends
    
    def _calculate_momentum_quality(self, results: List[float], 
                                   goals_for: List[float], 
                                   goals_against: List[float]) -> float:
        """
        Calculate momentum quality based on wins and goal margins
        
        Returns:
            Float: Quality score from 0 to 1
        """
        if not results:
            return 0.5
        
        quality_scores = []
        for i in range(len(results)):
            result = results[i]
            
            # Base quality from result
            if result == 1.0:  # Win
                base_quality = 0.7
            elif result == 0.5:  # Draw
                base_quality = 0.5
            else:  # Loss
                base_quality = 0.3
            
            # Adjust for goal margin
            margin = goals_for[i] - goals_against[i]
            
            # Higher quality for big wins, lower for big losses
            margin_factor = min(0.3, max(-0.3, margin * 0.1))
            
            quality_scores.append(min(1.0, max(0.0, base_quality + margin_factor)))
        
        # Apply exponential weights (more weight to recent matches)
        weights = np.exp(np.linspace(0, 1, len(quality_scores))) - 1
        weights = weights / np.sum(weights)
        
        return float(np.sum(weights * np.array(quality_scores)))
    
    def _calculate_momentum_volatility(self, results: List[float]) -> float:
        """
        Calculate momentum volatility based on result changes
        
        Returns:
            Float: Volatility score from 0 (stable) to 1 (volatile)
        """
        if len(results) < 3:
            return 0.5
        
        # Calculate changes between consecutive results
        changes = [abs(results[i] - results[i-1]) for i in range(1, len(results))]
        
        # Average change (normalized)
        avg_change = np.mean(changes) * 2  # *2 as max difference is 0.5
        
        return float(min(1.0, avg_change))
    
    def _is_win(self, match: Dict[str, Any], team_id: Union[int, None]) -> bool:
        """Determine if the team won the match"""
        if team_id is None:
            return False
            
        is_home = match.get('home_team_id') == team_id
        
        if is_home:
            return match.get('home_goals', 0) > match.get('away_goals', 0)
        else:
            return match.get('away_goals', 0) > match.get('home_goals', 0)
    
    def _is_draw(self, match: Dict[str, Any], team_id: Union[int, None]) -> bool:
        """Determine if the match was a draw"""
        # team_id is not used in this method, but keeping the parameter for consistency
        return match.get('home_goals', 0) == match.get('away_goals', 0)
    
    def _get_team_goals(self, match: Dict[str, Any], team_id: Union[int, None]) -> float:
        """Get goals scored by the team"""
        if team_id is None:
            return 0.0
            
        is_home = match.get('home_team_id') == team_id
        
        if is_home:
            return float(match.get('home_goals', 0))
        else:
            return float(match.get('away_goals', 0))
    
    def _get_opponent_goals(self, match: Dict[str, Any], team_id: Union[int, None]) -> float:
        """Get goals conceded by the team"""
        if team_id is None:
            return 0.0
            
        is_home = match.get('home_team_id') == team_id
        
        if is_home:
            return float(match.get('away_goals', 0))
        else:
            return float(match.get('home_goals', 0))


class TeamStrengthIndex:
    """
    Calculates composite team strength indices considering multiple factors
    """
    def __init__(self):
        self.form_weight = 0.3
        self.historical_weight = 0.2
        self.goals_weight = 0.15
        self.defense_weight = 0.15
        self.xg_weight = 0.1
        self.tactical_weight = 0.1
        self.logger = logging.getLogger(__name__)
    
    def calculate_team_strength(self, 
                              team_data: Dict[str, Any], 
                              matches: List[Dict[str, Any]],
                              league_data: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
        """
        Calculate comprehensive team strength index
        
        Args:
            team_data: Team information including league position, ratings, etc.
            matches: List of recent matches
            league_data: Optional league-level data for context
            
        Returns:
            Dictionary with various strength indices
        """
        try:
            if not team_data or not matches:
                return self._get_default_strength_indices()
            
            # Calculate various components of team strength
            offensive_strength = self._calculate_offensive_strength(team_data, matches)
            defensive_strength = self._calculate_defensive_strength(team_data, matches)
            form_strength = self._calculate_form_strength(matches)
            tactical_strength = self._calculate_tactical_strength(team_data, matches)
            historical_strength = self._calculate_historical_strength(team_data)
            xg_strength = self._calculate_xg_strength(matches)
            
            # Combine into overall strength index
            overall_strength = (
                self.form_weight * form_strength +
                self.historical_weight * historical_strength +
                self.goals_weight * offensive_strength +
                self.defense_weight * defensive_strength +
                self.xg_weight * xg_strength +
                self.tactical_weight * tactical_strength
            )
            
            # Calculate home/away specific strength indices
            home_strength = self._calculate_venue_specific_strength(matches, True)
            away_strength = self._calculate_venue_specific_strength(matches, False)
            
            # Calculate contextual strength against different opponent levels
            top_team_strength = self._calculate_contextual_strength(matches, 'top')
            mid_team_strength = self._calculate_contextual_strength(matches, 'mid')
            bottom_team_strength = self._calculate_contextual_strength(matches, 'bottom')
            
            return {
                'overall_strength': float(min(1.0, max(0.0, overall_strength))),
                'offensive_strength': float(min(1.0, max(0.0, offensive_strength))),
                'defensive_strength': float(min(1.0, max(0.0, defensive_strength))),
                'form_strength': float(min(1.0, max(0.0, form_strength))),
                'tactical_strength': float(min(1.0, max(0.0, tactical_strength))),
                'home_strength': float(min(1.0, max(0.0, home_strength))),
                'away_strength': float(min(1.0, max(0.0, away_strength))),
                'vs_top_teams': float(min(1.0, max(0.0, top_team_strength))),
                'vs_mid_teams': float(min(1.0, max(0.0, mid_team_strength))),
                'vs_bottom_teams': float(min(1.0, max(0.0, bottom_team_strength))),
                'consistency_index': float(min(1.0, max(0.0, self._calculate_consistency(matches))))
            }
        
        except Exception as e:
            self.logger.error(f"Error calculating team strength indices: {e}")
            return self._get_default_strength_indices()
    
    def _calculate_offensive_strength(self, team_data: Dict[str, Any], 
                                    matches: List[Dict[str, Any]]) -> float:
        """Calculate offensive strength based on goals and attacking metrics"""
        # Extract goal scoring data
        goals_scored = [self._get_team_goals(m, team_data['team_id']) for m in matches]
        
        if not goals_scored:
            return 0.5
        
        # Calculate average goals and normalize (2 goals per game is average)
        avg_goals = sum(goals_scored) / len(goals_scored)
        norm_goals = min(1.0, avg_goals / 2.0)
        
        # Get attacking metrics if available
        shots_on_target_ratio = team_data.get('shots_on_target_ratio', 0.33)
        shot_conversion = team_data.get('shot_conversion', 0.1)
        
        # Combine metrics
        return float(0.5 * norm_goals + 0.25 * shots_on_target_ratio + 0.25 * min(1.0, shot_conversion * 5.0))
    
    def _calculate_defensive_strength(self, team_data: Dict[str, Any], 
                                    matches: List[Dict[str, Any]]) -> float:
        """Calculate defensive strength based on goals conceded and defensive metrics"""
        # Extract goals conceded data
        goals_conceded = [self._get_opponent_goals(m, team_data['team_id']) for m in matches]
        
        if not goals_conceded:
            return 0.5
        
        # Calculate average conceded and normalize (lower is better)
        avg_conceded = sum(goals_conceded) / len(goals_conceded)
        norm_conceded = 1.0 - min(1.0, avg_conceded / 2.0)
        
        # Calculate clean sheet ratio
        clean_sheets = sum(1 for g in goals_conceded if g == 0)
        clean_sheet_ratio = clean_sheets / len(goals_conceded)
        
        # Get defensive metrics if available
        defensive_actions = team_data.get('defensive_actions', 30) / 50  # Normalize
        
        # Combine metrics
        return float(0.4 * norm_conceded + 0.4 * clean_sheet_ratio + 0.2 * min(1.0, defensive_actions))
    
    def _calculate_form_strength(self, matches: List[Dict[str, Any]]) -> float:
        """Calculate form strength based on recent results"""
        if not matches:
            return 0.5
        
        # Focus on most recent matches
        recent_matches = matches[:min(10, len(matches))]
        
        # Calculate points from results (3 for win, 1 for draw, 0 for loss)
        points = []
        for match in recent_matches:
            result = self._get_match_result(match, match.get('team_id'))
            if result == 'W':
                points.append(3.0)
            elif result == 'D':
                points.append(1.0)
            else:
                points.append(0.0)
        
        # Apply exponential weights
        weights = np.exp(np.linspace(0, 1, len(points))) - 1
        weights = weights / np.sum(weights)
        
        # Calculate weighted PPG (points per game)
        weighted_ppg = np.sum(weights * np.array(points))
        
        # Normalize to [0,1] scale (3 ppg = 1.0, 0 ppg = 0.0)
        return float(weighted_ppg / 3.0)
    
    def _calculate_tactical_strength(self, team_data: Dict[str, Any], 
                                    matches: List[Dict[str, Any]]) -> float:
        """Calculate tactical strength based on style and adaptability"""
        # Default to mid-level if no tactical data
        if not team_data.get('tactical_data'):
            return 0.5
        
        tactical_data = team_data.get('tactical_data', {})
        
        # Get tactical metrics
        pressing_success = tactical_data.get('pressing_success', 0.5)
        possession_control = tactical_data.get('possession_control', 0.5)
        buildup_success = tactical_data.get('buildup_success', 0.5)
        tactical_flexibility = tactical_data.get('tactical_flexibility', 0.5)
        
        # Combine metrics
        return float(0.3 * pressing_success + 
                    0.3 * possession_control + 
                    0.2 * buildup_success +
                    0.2 * tactical_flexibility)
    
    def _calculate_historical_strength(self, team_data: Dict[str, Any]) -> float:
        """Calculate historical strength based on league position and achievements"""
        # Default strength based on normalized league position
        position = team_data.get('league_position', 10)
        league_size = team_data.get('league_size', 20)
        
        # Normalize position (closer to 1 is better)
        norm_position = 1.0 - ((position - 1) / (league_size - 1)) if league_size > 1 else 0.5
        
        # Factor in trophies and historical success if available
        trophy_score = min(1.0, team_data.get('trophy_score', 0) / 10)
        
        # Factor in league prestige
        league_prestige = team_data.get('league_prestige', 0.5)
        
        # Combine metrics
        return float(0.6 * norm_position + 0.2 * trophy_score + 0.2 * league_prestige)
    
    def _calculate_xg_strength(self, matches: List[Dict[str, Any]]) -> float:
        """Calculate strength based on xG performance"""
        # Check if xG data is available
        xg_matches = [m for m in matches if 'xg' in m or 'xg_for' in m]
        
        if not xg_matches:
            return 0.5
        
        # Calculate xG differential
        xg_diffs = []
        for match in xg_matches:
            team_id = match.get('team_id')
            if 'xg_for' in match and 'xg_against' in match:
                xg_for = match.get('xg_for', 1.0)
                xg_against = match.get('xg_against', 1.0)
            else:
                # Handle different xG formats
                is_home = match.get('home_team_id') == team_id
                if is_home:
                    xg_for = match.get('home_xg', 1.0)
                    xg_against = match.get('away_xg', 1.0)
                else:
                    xg_for = match.get('away_xg', 1.0)
                    xg_against = match.get('home_xg', 1.0)
            
            xg_diffs.append(xg_for - xg_against)
        
        # Calculate average xG differential
        avg_xg_diff = sum(xg_diffs) / len(xg_diffs)
        
        # Normalize to [0,1] (range from -2 to +2 xG difference)
        return float(min(1.0, max(0.0, (avg_xg_diff + 2) / 4.0)))
    
    def _calculate_venue_specific_strength(self, matches: List[Dict[str, Any]], is_home: bool) -> float:
        """Calculate strength specific to home/away venues"""
        # Filter matches by venue
        venue_matches = [m for m in matches if (m.get('home_team_id') == m.get('team_id')) == is_home]
        
        if not venue_matches:
            return 0.5
        
        # Calculate points from these matches
        points = []
        for match in venue_matches:
            result = self._get_match_result(match, match.get('team_id'))
            if result == 'W':
                points.append(3.0)
            elif result == 'D':
                points.append(1.0)
            else:
                points.append(0.0)
        
        # Calculate PPG (points per game)
        ppg = sum(points) / len(points)
        
        # Normalize to [0,1]
        return float(ppg / 3.0)
    
    def _calculate_contextual_strength(self, matches: List[Dict[str, Any]], opponent_level: str) -> float:
        """Calculate strength against different levels of opposition"""
        # Filter matches by opponent level
        level_matches = []
        for match in matches:
            opp_strength = match.get('opponent_strength', 0.5)
            
            # Categorize opponent
            if opponent_level == 'top' and opp_strength >= 0.7:
                level_matches.append(match)
            elif opponent_level == 'mid' and 0.3 < opp_strength < 0.7:
                level_matches.append(match)
            elif opponent_level == 'bottom' and opp_strength <= 0.3:
                level_matches.append(match)
        
        if not level_matches:
            return 0.5
        
        # Calculate points from these matches
        points = []
        for match in level_matches:
            result = self._get_match_result(match, match.get('team_id'))
            if result == 'W':
                points.append(3.0)
            elif result == 'D':
                points.append(1.0)
            else:
                points.append(0.0)
        
        # Calculate PPG
        ppg = sum(points) / len(points)
        
        # Normalize to [0,1]
        return float(ppg / 3.0)
    
    def _calculate_consistency(self, matches: List[Dict[str, Any]]) -> float:
        """Calculate team consistency index"""
        if len(matches) < 5:
            return 0.5
        
        # Extract results and goal data
        results = []
        goal_diffs = []
        
        for match in matches:
            team_id = match.get('team_id')
            result = self._get_match_result(match, team_id)
            
            # Convert result to numeric
            if result == 'W':
                results.append(1.0)
            elif result == 'D':
                results.append(0.5)
            else:
                results.append(0.0)
                
            # Calculate goal difference - handle None team_id safely
            if team_id is not None:
                goal_diff = self._get_team_goals(match, team_id) - self._get_opponent_goals(match, team_id)
                goal_diffs.append(goal_diff)
            else:
                goal_diffs.append(0.0)  # Default to 0 goal difference when team_id is missing
          # Calculate variance in results
        result_std = np.std(results) if len(results) > 1 else 0.0
        
        # Calculate variance in goal differences
        goal_diff_std = np.std(goal_diffs) if len(goal_diffs) > 1 else 0.0
        
        # Combine into consistency score (lower variance = higher consistency)
        consistency = 1.0 - min(1.0, (result_std + goal_diff_std/3) / 1.5)
        
        return float(consistency)
    
    def _get_default_strength_indices(self) -> Dict[str, float]:
        """Get default strength indices when data is unavailable"""
        return {
            'overall_strength': 0.5,
            'offensive_strength': 0.5,
            'defensive_strength': 0.5,
            'form_strength': 0.5,
            'tactical_strength': 0.5,
            'home_strength': 0.5,
            'away_strength': 0.5,
            'vs_top_teams': 0.5,
            'vs_mid_teams': 0.5,
            'vs_bottom_teams': 0.5,
            'consistency_index': 0.5
        }
    
    def _get_match_result(self, match: Dict[str, Any], team_id: Union[int, None]) -> str:
        """Get match result (W/D/L) for the team"""
        if team_id is None:
            return 'D'  # Default to draw when team_id is missing
        
        team_goals = self._get_team_goals(match, team_id)
        opponent_goals = self._get_opponent_goals(match, team_id)
        
        if team_goals > opponent_goals:
            return 'W'
        elif team_goals < opponent_goals:
            return 'L'
        else:            return 'D'
    
    def _get_team_goals(self, match: Dict[str, Any], team_id: Union[int, None]) -> float:
        """Get goals scored by the team"""
        if team_id is None:
            return 0.0
            
        is_home = match.get('home_team_id') == team_id
        
        if is_home:
            return float(match.get('home_goals', 0))
        else:
            return float(match.get('away_goals', 0))
    
    def _get_opponent_goals(self, match: Dict[str, Any], team_id: Union[int, None]) -> float:
        """Get goals conceded by the team"""
        if team_id is None:
            return 0.0
            
        is_home = match.get('home_team_id') == team_id
        
        if is_home:
            return float(match.get('away_goals', 0))
        else:
            return float(match.get('home_goals', 0))

# Helper function to integrate the advanced features
def add_advanced_features(matches_df: pd.DataFrame, 
                        team_data: Dict[int, Dict[str, Any]],
                        league_data: Optional[Dict[int, Dict[str, Any]]] = None) -> pd.DataFrame:
    """
    Add advanced momentum and strength features to match dataframe
    
    Args:
        matches_df: DataFrame of matches
        team_data: Dictionary of team data keyed by team_id
        league_data: Optional dictionary of league data
        
    Returns:
        DataFrame with added features
    """
    df = matches_df.copy()
    
    # Initialize feature calculators
    momentum_features = MomentumFeatures()
    strength_index = TeamStrengthIndex()
    
    # Group matches by team
    team_matches = {}
    for _, match in df.iterrows():
        home_id = match['home_team_id']
        away_id = match['away_team_id']
        
        if home_id not in team_matches:
            team_matches[home_id] = []
        if away_id not in team_matches:
            team_matches[away_id] = []
        
        # Add match to both teams' lists
        match_dict = match.to_dict()
        team_matches[home_id].append(match_dict)
        team_matches[away_id].append(match_dict)
    
    # Process each match
    for idx, match in df.iterrows():
        date = match['date']
        home_id = match['home_team_id']
        away_id = match['away_team_id']
        
        # Add momentum features
        home_momentum = momentum_features.calculate_momentum_features(
            team_matches[home_id], home_id, date, is_home=True
        )
        away_momentum = momentum_features.calculate_momentum_features(
            team_matches[away_id], away_id, date, is_home=False
        )
        
        # Add team strength indices
        home_league = league_data.get(match['league_id'], {}) if league_data else None
        away_league = home_league  # Same league
        
        home_strength = strength_index.calculate_team_strength(
            team_data.get(home_id, {'team_id': home_id}),
            team_matches[home_id],
            home_league
        )
        away_strength = strength_index.calculate_team_strength(
            team_data.get(away_id, {'team_id': away_id}),
            team_matches[away_id],
            away_league
        )
        
        # Add all features to dataframe
        for feature, value in home_momentum.items():
            df.at[idx, feature] = value
            
        for feature, value in away_momentum.items():
            df.at[idx, feature] = value
            
        for feature, value in home_strength.items():
            df.at[idx, f'home_{feature}'] = value
            
        for feature, value in away_strength.items():
            df.at[idx, f'away_{feature}'] = value
    
    return df
