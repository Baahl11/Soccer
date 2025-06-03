"""
Team Composition Changes Analysis Module

This module analyzes team composition changes and their impact on match outcomes.
It tracks lineup changes, player availability, formation shifts, and squad stability.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler
import json

logger = logging.getLogger(__name__)

@dataclass
class PlayerInfo:
    """Information about a player"""
    player_id: int
    name: str
    position: str
    market_value: float
    importance_score: float  # 1-10 scale
    minutes_played_season: int
    goals_season: int
    assists_season: int

@dataclass
class LineupChange:
    """Represents a lineup change between matches"""
    match_id: int
    team_id: int
    changes_count: int
    key_players_out: List[PlayerInfo]
    key_players_in: List[PlayerInfo]
    formation_change: bool
    total_value_change: float
    importance_impact: float

class TeamCompositionAnalyzer:
    """
    Analyzes team composition changes and their impact on performance
    """
    
    def __init__(self, db_connection=None):
        self.db_connection = db_connection
        self.scaler = StandardScaler()
        self.composition_cache = {}
        self.player_importance_weights = {
            'goalkeeper': 1.5,
            'defender': 1.0,
            'midfielder': 1.2,
            'forward': 1.3
        }
        
    def analyze_lineup_stability(self, team_id: int, matches_window: int = 10) -> Dict[str, float]:
        """
        Analyze lineup stability over recent matches
        
        Args:
            team_id: Team identifier
            matches_window: Number of recent matches to analyze
            
        Returns:
            Dict containing stability metrics
        """
        try:
            # Get recent lineups
            recent_lineups = self._get_recent_lineups(team_id, matches_window)
            
            if len(recent_lineups) < 2:
                return self._default_stability_metrics()
            
            stability_metrics = {
                'lineup_consistency': self._calculate_lineup_consistency(recent_lineups),
                'formation_stability': self._calculate_formation_stability(recent_lineups),
                'key_player_availability': self._calculate_key_player_availability(recent_lineups),
                'rotation_intensity': self._calculate_rotation_intensity(recent_lineups),
                'injury_impact': self._calculate_injury_impact(team_id, recent_lineups),
                'tactical_consistency': self._calculate_tactical_consistency(recent_lineups)
            }
            
            return stability_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing lineup stability for team {team_id}: {e}")
            return self._default_stability_metrics()
    
    def analyze_composition_changes(self, team_id: int, upcoming_match_id: int) -> Dict[str, Any]:
        """
        Analyze expected composition changes for upcoming match
        
        Args:
            team_id: Team identifier
            upcoming_match_id: Upcoming match identifier
            
        Returns:
            Dict containing composition change analysis
        """
        try:
            # Get last lineup and expected changes
            last_lineup = self._get_last_lineup(team_id)
            expected_changes = self._predict_lineup_changes(team_id, upcoming_match_id)
            
            analysis = {
                'expected_changes_count': expected_changes['changes_count'],
                'key_players_impact': expected_changes['importance_impact'],
                'formation_change_probability': expected_changes['formation_change_prob'],
                'injury_list_impact': self._analyze_injury_list_impact(team_id),
                'suspension_impact': self._analyze_suspension_impact(team_id),
                'rotation_probability': self._calculate_rotation_probability(team_id),
                'squad_depth_score': self._calculate_squad_depth_score(team_id),
                'composition_risk_score': self._calculate_composition_risk_score(expected_changes)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing composition changes for team {team_id}: {e}")
            return self._default_composition_analysis()
    
    def get_composition_features(self, home_team_id: int, away_team_id: int, 
                               match_date: datetime) -> Dict[str, float]:
        """
        Generate composition-related features for match prediction
        
        Args:
            home_team_id: Home team identifier
            away_team_id: Away team identifier
            match_date: Match date
            
        Returns:
            Dict containing composition features
        """
        try:
            # Analyze both teams
            home_stability = self.analyze_lineup_stability(home_team_id)
            away_stability = self.analyze_lineup_stability(away_team_id)
            
            home_changes = self.analyze_composition_changes(home_team_id, 0)  # 0 for upcoming
            away_changes = self.analyze_composition_changes(away_team_id, 0)
            
            features = {
                # Stability features
                'home_lineup_consistency': home_stability['lineup_consistency'],
                'away_lineup_consistency': away_stability['lineup_consistency'],
                'home_formation_stability': home_stability['formation_stability'],
                'away_formation_stability': away_stability['formation_stability'],
                'home_key_player_availability': home_stability['key_player_availability'],
                'away_key_player_availability': away_stability['key_player_availability'],
                
                # Change impact features
                'home_expected_changes': home_changes['expected_changes_count'],
                'away_expected_changes': away_changes['expected_changes_count'],
                'home_key_players_impact': home_changes['key_players_impact'],
                'away_key_players_impact': away_changes['key_players_impact'],
                
                # Risk and depth features
                'home_composition_risk': home_changes['composition_risk_score'],
                'away_composition_risk': away_changes['composition_risk_score'],
                'home_squad_depth': home_changes['squad_depth_score'],
                'away_squad_depth': away_changes['squad_depth_score'],
                
                # Relative features
                'stability_advantage': home_stability['lineup_consistency'] - away_stability['lineup_consistency'],
                'depth_advantage': home_changes['squad_depth_score'] - away_changes['squad_depth_score'],
                'composition_advantage': away_changes['composition_risk_score'] - home_changes['composition_risk_score']
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error generating composition features: {e}")
            return self._default_composition_features()
    
    def _get_recent_lineups(self, team_id: int, matches_count: int) -> List[Dict]:
        """Get recent team lineups"""
        if self.db_connection:
            query = """
            SELECT match_id, formation, players, match_date
            FROM team_lineups 
            WHERE team_id = ? 
            ORDER BY match_date DESC 
            LIMIT ?
            """
            return self._execute_query(query, (team_id, matches_count))
        else:
            # Mock data for testing
            return self._generate_mock_lineups(team_id, matches_count)
    
    def _get_last_lineup(self, team_id: int) -> Optional[Dict]:
        """Get team's last lineup"""
        recent = self._get_recent_lineups(team_id, 1)
        return recent[0] if recent else None
    
    def _calculate_lineup_consistency(self, lineups: List[Dict]) -> float:
        """Calculate lineup consistency score (0-1)"""
        if len(lineups) < 2:
            return 1.0
        
        consistency_scores = []
        for i in range(1, len(lineups)):
            current_players = set(lineups[i].get('players', []))
            previous_players = set(lineups[i-1].get('players', []))
            
            if len(current_players) == 0 or len(previous_players) == 0:
                continue
                  # Calculate player overlap
            overlap = len(current_players.intersection(previous_players))
            total_unique = len(current_players.union(previous_players))
            
            if total_unique > 0:
                consistency_scores.append(overlap / min(len(current_players), len(previous_players)))
        
        return float(np.mean(consistency_scores)) if consistency_scores else 0.5
    
    def _calculate_formation_stability(self, lineups: List[Dict]) -> float:
        """Calculate formation stability score (0-1)"""
        if len(lineups) < 2:
            return 1.0
        
        formations = [lineup.get('formation', '4-4-2') for lineup in lineups]
        unique_formations = len(set(formations))
        
        # Higher stability = fewer formation changes
        return max(0.0, 1.0 - (unique_formations - 1) / len(lineups))
    
    def _calculate_key_player_availability(self, lineups: List[Dict]) -> float:
        """Calculate key player availability score (0-1)"""
        if not lineups:
            return 0.5
        
        # Identify key players from recent appearances
        player_appearances = {}
        total_matches = len(lineups)
        
        for lineup in lineups:
            players = lineup.get('players', [])
            for player in players:
                player_appearances[player] = player_appearances.get(player, 0) + 1
        
        # Players appearing in >70% of matches are considered key
        key_players = [p for p, count in player_appearances.items() 
                      if count / total_matches > 0.7]
        
        if not key_players:
            return 0.5
        
        # Check availability in most recent match
        recent_players = set(lineups[0].get('players', []))
        available_key_players = len([p for p in key_players if p in recent_players])
        
        return available_key_players / len(key_players) if key_players else 0.5
    
    def _calculate_rotation_intensity(self, lineups: List[Dict]) -> float:
        """Calculate rotation intensity (0-1, higher = more rotation)"""
        if len(lineups) < 2:
            return 0.0
        changes_per_match = []
        for i in range(1, len(lineups)):
            current_players = set(lineups[i].get('players', []))
            previous_players = set(lineups[i-1].get('players', []))
            
            if len(previous_players) > 0:
                changes = len(previous_players.symmetric_difference(current_players))
                changes_per_match.append(changes / 11)  # Normalize by starting XI size
        
        return float(np.mean(changes_per_match)) if changes_per_match else 0.0
    
    def _calculate_injury_impact(self, team_id: int, lineups: List[Dict]) -> float:
        """Calculate injury impact on team composition (0-1)"""
        # This would typically integrate with injury/medical data
        # For now, simulate based on player consistency patterns
        
        if not lineups:
            return 0.0
        
        # Look for sudden player absences (potential injuries)
        recent_absences = 0
        if len(lineups) >= 3:
            # Check for players who were regular but suddenly absent
            regular_lineup = set(lineups[-1].get('players', []))
            recent_lineup = set(lineups[0].get('players', []))
            
            missing_regulars = regular_lineup - recent_lineup
            recent_absences = len(missing_regulars)
        
        # Normalize impact (assuming max 4 key injuries significantly impact team)
        return min(1.0, recent_absences / 4.0)
    
    def _calculate_tactical_consistency(self, lineups: List[Dict]) -> float:
        """Calculate tactical consistency based on formations and player positions"""
        if len(lineups) < 2:
            return 1.0
        
        formation_consistency = self._calculate_formation_stability(lineups)
        
        # Could be extended to analyze positional consistency
        # For now, use formation consistency as proxy
        return formation_consistency
    
    def _predict_lineup_changes(self, team_id: int, match_id: int) -> Dict[str, Any]:
        """Predict likely lineup changes for upcoming match"""
        # This would integrate with injury reports, suspensions, rotation patterns
        # For now, provide reasonable estimates
        
        recent_lineups = self._get_recent_lineups(team_id, 5)
        rotation_intensity = self._calculate_rotation_intensity(recent_lineups)
        
        # Estimate changes based on rotation pattern
        expected_changes = int(rotation_intensity * 11 * 1.2)  # Slight increase for upcoming match
        
        return {
            'changes_count': min(expected_changes, 6),  # Max 6 changes typically
            'importance_impact': rotation_intensity * 0.8,  # Assume some key players affected
            'formation_change_prob': 0.2 if len(recent_lineups) > 0 else 0.1
        }
    
    def _analyze_injury_list_impact(self, team_id: int) -> float:
        """Analyze impact of current injury list"""
        # This would integrate with real injury data
        # For now, simulate reasonable values
        return np.random.uniform(0.0, 0.3)  # 0-30% impact
    
    def _analyze_suspension_impact(self, team_id: int) -> float:
        """Analyze impact of current suspensions"""
        # This would integrate with real suspension data
        # For now, simulate reasonable values
        return np.random.uniform(0.0, 0.2)  # 0-20% impact
    
    def _calculate_rotation_probability(self, team_id: int) -> float:
        """Calculate probability of significant rotation"""
        recent_lineups = self._get_recent_lineups(team_id, 3)
        rotation_intensity = self._calculate_rotation_intensity(recent_lineups)
        
        # Higher recent rotation suggests higher probability of continued rotation
        return min(1.0, rotation_intensity * 1.5)
    
    def _calculate_squad_depth_score(self, team_id: int) -> float:
        """Calculate squad depth score (0-1)"""
        # This would analyze full squad, market values, player ratings
        # For now, provide reasonable estimates based on team tier
        
        # Could be enhanced with actual squad analysis
        return np.random.uniform(0.4, 0.9)  # Most teams have decent depth
    
    def _calculate_composition_risk_score(self, expected_changes: Dict[str, Any]) -> float:
        """Calculate overall composition risk score (0-1)"""
        changes_risk = min(1.0, expected_changes['changes_count'] / 6.0)
        importance_risk = expected_changes['importance_impact']
        formation_risk = expected_changes['formation_change_prob']
        
        # Weighted combination
        return (changes_risk * 0.4 + importance_risk * 0.4 + formation_risk * 0.2)
    
    def _generate_mock_lineups(self, team_id: int, matches_count: int) -> List[Dict]:
        """Generate mock lineup data for testing"""
        lineups = []
        base_players = [f"player_{i}" for i in range(1, 16)]  # 15 players pool
        
        for i in range(matches_count):
            # Simulate realistic lineup changes
            if i == 0:
                players = base_players[:11]  # Starting XI
            else:
                # Make 1-3 changes from previous lineup
                changes = np.random.randint(1, 4)
                players = lineups[-1]['players'].copy()
                
                # Replace some players
                for _ in range(changes):
                    if len(players) >= 11:
                        out_idx = np.random.randint(0, len(players))
                        available_players = [p for p in base_players if p not in players]
                        if available_players:
                            players[out_idx] = np.random.choice(available_players)
            
            formations = ['4-4-2', '4-3-3', '3-5-2', '4-2-3-1']
            formation = np.random.choice(formations, p=[0.4, 0.3, 0.2, 0.1])
            lineups.append({
                'match_id': f"match_{i}",
                'formation': formation,
                'players': players[:11],
                'match_date': datetime.now() - timedelta(days=i*7)
            })
        
        return lineups
    
    def _execute_query(self, query: str, params: tuple) -> List[Dict]:
        """Execute database query and return results"""
        if self.db_connection is None:
            logger.warning("No database connection available")
            return []
            
        try:
            cursor = self.db_connection.cursor()
            cursor.execute(query, params)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Database query error: {e}")
            return []
    
    def _default_stability_metrics(self) -> Dict[str, float]:
        """Default stability metrics when data is unavailable"""
        return {
            'lineup_consistency': 0.7,
            'formation_stability': 0.8,
            'key_player_availability': 0.8,
            'rotation_intensity': 0.3,
            'injury_impact': 0.1,
            'tactical_consistency': 0.8
        }
    
    def _default_composition_analysis(self) -> Dict[str, Any]:
        """Default composition analysis when data is unavailable"""
        return {
            'expected_changes_count': 2,
            'key_players_impact': 0.2,
            'formation_change_probability': 0.1,
            'injury_list_impact': 0.1,
            'suspension_impact': 0.05,
            'rotation_probability': 0.3,
            'squad_depth_score': 0.6,
            'composition_risk_score': 0.3
        }
    
    def _default_composition_features(self) -> Dict[str, float]:
        """Default composition features when analysis fails"""
        return {
            'home_lineup_consistency': 0.7,
            'away_lineup_consistency': 0.7,
            'home_formation_stability': 0.8,
            'away_formation_stability': 0.8,
            'home_key_player_availability': 0.8,
            'away_key_player_availability': 0.8,
            'home_expected_changes': 2,
            'away_expected_changes': 2,
            'home_key_players_impact': 0.2,
            'away_key_players_impact': 0.2,
            'home_composition_risk': 0.3,
            'away_composition_risk': 0.3,
            'home_squad_depth': 0.6,
            'away_squad_depth': 0.6,
            'stability_advantage': 0.0,
            'depth_advantage': 0.0,
            'composition_advantage': 0.0
        }

def demonstrate_composition_analysis():
    """Demonstrate team composition analysis functionality"""
    print("=== Team Composition Analysis Demo ===")
    
    analyzer = TeamCompositionAnalyzer()
    
    # Analyze lineup stability
    print("\n1. Lineup Stability Analysis:")
    stability = analyzer.analyze_lineup_stability(team_id=123)
    for metric, value in stability.items():
        print(f"   {metric}: {value:.3f}")
    
    # Analyze composition changes
    print("\n2. Composition Changes Analysis:")
    changes = analyzer.analyze_composition_changes(team_id=123, upcoming_match_id=456)
    for metric, value in changes.items():
        print(f"   {metric}: {value}")
    
    # Generate features for match prediction
    print("\n3. Match Prediction Features:")
    features = analyzer.get_composition_features(
        home_team_id=123, 
        away_team_id=456, 
        match_date=datetime.now()
    )
    for feature, value in features.items():
        print(f"   {feature}: {value:.3f}")

if __name__ == "__main__":
    demonstrate_composition_analysis()
