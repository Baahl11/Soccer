#!/usr/bin/env python3
"""
Real-Time Injury Analyzer

Sistema avanzado de análisis de lesiones en tiempo real para maximizar
la precisión de predicciones considerando el impacto de jugadores ausentes.

Este módulo proporciona:
1. Análisis detallado de lesiones de jugadores clave
2. Evaluación del impacto táctico de las ausencias
3. Calidad de reemplazos disponibles
4. Performance histórica sin jugadores específicos
5. Predicción de fechas de regreso
"""

import logging
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class InjuredPlayer:
    """Información de jugador lesionado"""
    player_id: int
    name: str
    position: str
    importance_rating: float  # 0-1 scale
    injury_type: str
    severity: str  # "minor", "moderate", "severe"
    expected_return: Optional[datetime]
    replacement_quality: float  # 0-1 scale

@dataclass
class InjuryImpact:
    """Impacto de lesiones en el equipo"""
    attacking_impact: float
    defensive_impact: float
    midfield_creativity: float
    set_piece_threat: float
    leadership_impact: float
    overall_team_strength: float

class RealTimeInjuryAnalyzer:
    """
    Analizador de lesiones en tiempo real para predicciones de fútbol
    """
    
    def __init__(self):
        """Inicializar el analizador de lesiones"""
        self.injury_sources = {
            "transfermarkt": "https://api.transfermarkt.com/injuries",
            "football_api": "https://api.football-api.com/injuries",
            "team_news": "https://api.team-news.com/injuries"
        }
        
        # Factores de importancia por posición
        self.position_importance = {
            "GK": 0.95,   # Portero muy importante
            "CB": 0.85,   # Central importante
            "LB": 0.70,   # Lateral menos crítico
            "RB": 0.70,   # Lateral menos crítico
            "CDM": 0.80,  # Pivote importante
            "CM": 0.75,   # Centrocampista
            "CAM": 0.85,  # Mediapunta muy importante
            "LW": 0.75,   # Extremo
            "RW": 0.75,   # Extremo
            "ST": 0.90,   # Delantero muy importante
            "CF": 0.90    # Delantero centro
        }
        
        # Cache para evitar llamadas repetidas
        self.injury_cache = {}
        self.cache_duration = timedelta(hours=2)
    
    def get_comprehensive_injury_report(self, team_id: int, fixture_id: int) -> Dict[str, Any]:
        """
        Genera reporte completo de lesiones para un equipo
        
        Args:
            team_id: ID del equipo
            fixture_id: ID del partido
            
        Returns:
            Reporte completo de impacto de lesiones
        """
        try:
            # Verificar cache
            cache_key = f"injury_report_{team_id}_{fixture_id}"
            if cache_key in self.injury_cache:
                cached_data = self.injury_cache[cache_key]
                if datetime.now() - cached_data["timestamp"] < self.cache_duration:
                    return cached_data["data"]
            
            # Obtener datos de lesiones
            injured_players = self._get_current_injuries(team_id)
            
            # Analizar impacto
            impact_analysis = self._calculate_comprehensive_impact(injured_players, team_id)
            
            # Generar reporte
            report = {
                "team_id": team_id,
                "fixture_id": fixture_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "injured_players": [self._player_to_dict(player) for player in injured_players],
                "impact_analysis": impact_analysis,
                "tactical_adjustments": self._predict_tactical_adjustments(injured_players, team_id),
                "replacement_analysis": self._analyze_replacements(injured_players, team_id),
                "historical_performance": self._get_performance_without_players(injured_players, team_id),
                "match_readiness": self._assess_match_readiness(injured_players, fixture_id),
                "confidence_impact": self._calculate_confidence_impact(impact_analysis)
            }
            
            # Guardar en cache
            self.injury_cache[cache_key] = {
                "data": report,
                "timestamp": datetime.now()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating injury report for team {team_id}: {e}")
            return self._get_default_injury_report(team_id, fixture_id)
    
    def _get_current_injuries(self, team_id: int) -> List[InjuredPlayer]:
        """Obtiene lista actual de jugadores lesionados"""
        try:
            injured_players = []
            
            # En un entorno real, esto haría llamadas a APIs
            # Por ahora, simulamos con datos realistas
            mock_injuries = self._get_mock_injury_data(team_id)
            
            for injury_data in mock_injuries:
                player = InjuredPlayer(
                    player_id=injury_data["player_id"],
                    name=injury_data["name"],
                    position=injury_data["position"],
                    importance_rating=self._calculate_player_importance(
                        injury_data["position"], 
                        injury_data.get("market_value", 0),
                        injury_data.get("minutes_played", 0)
                    ),
                    injury_type=injury_data["injury_type"],
                    severity=injury_data["severity"],
                    expected_return=injury_data.get("expected_return"),
                    replacement_quality=injury_data.get("replacement_quality", 0.7)
                )
                injured_players.append(player)
            
            return injured_players
            
        except Exception as e:
            logger.error(f"Error getting current injuries for team {team_id}: {e}")
            return []
    
    def _calculate_comprehensive_impact(self, injured_players: List[InjuredPlayer], team_id: int) -> Dict[str, Any]:
        """Calcula impacto comprehensivo de las lesiones"""
        try:
            if not injured_players:
                return self._get_zero_impact()
            
            # Calcular impactos por área
            attacking_impact = self._calculate_attacking_impact(injured_players)
            defensive_impact = self._calculate_defensive_impact(injured_players)
            midfield_impact = self._calculate_midfield_impact(injured_players)
            set_piece_impact = self._calculate_set_piece_impact(injured_players)
            leadership_impact = self._calculate_leadership_impact(injured_players)
            
            # Calcular impacto general del equipo
            overall_impact = self._calculate_overall_team_impact(
                attacking_impact, defensive_impact, midfield_impact, 
                set_piece_impact, leadership_impact
            )
            
            return {
                "attacking_impact": attacking_impact,
                "defensive_impact": defensive_impact,
                "midfield_creativity": midfield_impact,
                "set_piece_threat": set_piece_impact,
                "leadership_impact": leadership_impact,
                "overall_team_strength": overall_impact,
                "total_players_injured": len(injured_players),
                "key_players_injured": len([p for p in injured_players if p.importance_rating > 0.8]),
                "severity_breakdown": self._get_severity_breakdown(injured_players),
                "positional_coverage": self._analyze_positional_coverage(injured_players, team_id)
            }
            
        except Exception as e:
            logger.error(f"Error calculating comprehensive impact: {e}")
            return self._get_zero_impact()
    
    def _calculate_attacking_impact(self, injured_players: List[InjuredPlayer]) -> float:
        """Calcula impacto en capacidad ofensiva"""
        attacking_positions = ["ST", "CF", "LW", "RW", "CAM"]
        attacking_injuries = [p for p in injured_players if p.position in attacking_positions]
        
        if not attacking_injuries:
            return 0.0
        
        impact = 0.0
        for player in attacking_injuries:
            position_weight = self.position_importance.get(player.position, 0.5)
            severity_multiplier = self._get_severity_multiplier(player.severity)
            replacement_factor = 1 - player.replacement_quality
            
            player_impact = player.importance_rating * position_weight * severity_multiplier * replacement_factor
            impact += player_impact
        
        # Normalizar y limitar impacto máximo
        return min(impact * 0.5, 0.4)  # Máximo 40% de reducción
    
    def _calculate_defensive_impact(self, injured_players: List[InjuredPlayer]) -> float:
        """Calcula impacto en capacidad defensiva"""
        defensive_positions = ["GK", "CB", "LB", "RB", "CDM"]
        defensive_injuries = [p for p in injured_players if p.position in defensive_positions]
        
        if not defensive_injuries:
            return 0.0
        
        impact = 0.0
        for player in defensive_injuries:
            position_weight = self.position_importance.get(player.position, 0.5)
            severity_multiplier = self._get_severity_multiplier(player.severity)
            replacement_factor = 1 - player.replacement_quality
            
            player_impact = player.importance_rating * position_weight * severity_multiplier * replacement_factor
            impact += player_impact
        
        return min(impact * 0.6, 0.35)  # Máximo 35% de reducción
    
    def _calculate_midfield_impact(self, injured_players: List[InjuredPlayer]) -> float:
        """Calcula impacto en creatividad del mediocampo"""
        midfield_positions = ["CM", "CAM", "CDM"]
        midfield_injuries = [p for p in injured_players if p.position in midfield_positions]
        
        if not midfield_injuries:
            return 0.0
        
        impact = 0.0
        for player in midfield_injuries:
            position_weight = self.position_importance.get(player.position, 0.5)
            severity_multiplier = self._get_severity_multiplier(player.severity)
            replacement_factor = 1 - player.replacement_quality
            
            player_impact = player.importance_rating * position_weight * severity_multiplier * replacement_factor
            impact += player_impact
        
        return min(impact * 0.55, 0.3)  # Máximo 30% de reducción
    
    def _calculate_set_piece_impact(self, injured_players: List[InjuredPlayer]) -> float:
        """Calcula impacto en amenaza de balón parado"""
        # Jugadores clave en balón parado suelen ser CAM, ST, CB (corners)
        set_piece_positions = ["CAM", "ST", "CB", "LW", "RW"]
        set_piece_injuries = [p for p in injured_players 
                            if p.position in set_piece_positions and p.importance_rating > 0.7]
        
        if not set_piece_injuries:
            return 0.0
        
        # Impacto significativo porque especialistas son únicos
        impact = len(set_piece_injuries) * 0.15
        return min(impact, 0.4)  # Máximo 40% de reducción
    
    def _calculate_leadership_impact(self, injured_players: List[InjuredPlayer]) -> float:
        """Calcula impacto en liderazgo del equipo"""
        # Asumir que jugadores con alta importancia son líderes
        leaders_injured = [p for p in injured_players if p.importance_rating > 0.85]
        
        if not leaders_injured:
            return 0.0
        
        # Cada líder lesionado tiene impacto significativo
        impact = len(leaders_injured) * 0.12
        return min(impact, 0.25)  # Máximo 25% de reducción
    
    def _calculate_overall_team_impact(self, attacking: float, defensive: float, 
                                     midfield: float, set_piece: float, leadership: float) -> float:
        """Calcula impacto general en la fortaleza del equipo"""
        # Pesos para diferentes aspectos
        weights = {
            "attacking": 0.25,
            "defensive": 0.25,
            "midfield": 0.20,
            "set_piece": 0.15,
            "leadership": 0.15
        }
        
        overall_impact = (
            attacking * weights["attacking"] +
            defensive * weights["defensive"] +
            midfield * weights["midfield"] +
            set_piece * weights["set_piece"] +
            leadership * weights["leadership"]
        )
        
        return min(overall_impact, 0.35)  # Máximo 35% de reducción general
    
    def _predict_tactical_adjustments(self, injured_players: List[InjuredPlayer], team_id: int) -> Dict[str, Any]:
        """Predice ajustes tácticos necesarios por las lesiones"""
        if not injured_players:
            return {"formation_change": False, "tactical_adjustments": []}
        
        adjustments = []
        formation_change = False
        
        # Analizar por posiciones afectadas
        positions_affected = [p.position for p in injured_players]
        
        if "GK" in positions_affected:
            adjustments.append("Backup goalkeeper may affect distribution and communication")
        
        if any(pos in positions_affected for pos in ["CB", "LB", "RB"]):
            adjustments.append("Defensive line may lack cohesion with replacement players")
            if len([p for p in injured_players if p.position in ["CB", "LB", "RB"]]) >= 2:
                formation_change = True
                adjustments.append("Formation change likely to accommodate defensive weaknesses")
        
        if any(pos in positions_affected for pos in ["ST", "CF"]):
            adjustments.append("Attacking patterns may change with different striker profile")
        
        if "CAM" in positions_affected:
            adjustments.append("Creative playmaking may suffer, more direct play expected")
        
        return {
            "formation_change": formation_change,
            "tactical_adjustments": adjustments,
            "expected_style_change": self._predict_style_change(injured_players),
            "vulnerability_areas": self._identify_vulnerability_areas(injured_players)
        }
    
    def _analyze_replacements(self, injured_players: List[InjuredPlayer], team_id: int) -> Dict[str, Any]:
        """Analiza calidad de jugadores de reemplazo"""
        if not injured_players:
            return {"replacement_quality_avg": 1.0, "depth_concerns": []}
        
        replacement_analysis = {
            "replacement_quality_avg": np.mean([p.replacement_quality for p in injured_players]),
            "depth_concerns": [],
            "positional_depth": {},
            "quality_drop_by_position": {}
        }
        
        for player in injured_players:
            quality_drop = 1 - player.replacement_quality
            replacement_analysis["quality_drop_by_position"][player.position] = quality_drop
            
            if quality_drop > 0.3:
                replacement_analysis["depth_concerns"].append(
                    f"Significant quality drop at {player.position} ({player.name})"
                )
        
        return replacement_analysis
    
    def _get_performance_without_players(self, injured_players: List[InjuredPlayer], team_id: int) -> Dict[str, Any]:
        """Obtiene performance histórica sin jugadores específicos"""
        # En un entorno real, esto consultaría una base de datos
        # Por ahora, simulamos basado en la importancia de los jugadores
        
        if not injured_players:
            return {"matches_without": 0, "win_rate": 1.0, "goals_avg": 1.5}
        
        importance_loss = sum(p.importance_rating for p in injured_players) / len(injured_players)
        
        # Simular impacto en performance
        base_win_rate = 0.50
        base_goals_avg = 1.5
        
        win_rate_impact = importance_loss * 0.3
        goals_impact = importance_loss * 0.4
        
        return {
            "matches_without": 10,  # Simulado
            "win_rate": max(0.1, base_win_rate - win_rate_impact),
            "goals_avg": max(0.5, base_goals_avg - goals_impact),
            "goals_conceded_avg": base_goals_avg + (importance_loss * 0.2),
            "estimated_impact": importance_loss
        }
    
    def _assess_match_readiness(self, injured_players: List[InjuredPlayer], fixture_id: int) -> Dict[str, Any]:
        """Evalúa preparación del equipo para el partido específico"""
        if not injured_players:
            return {"readiness_score": 1.0, "concerns": []}
        
        concerns = []
        readiness_impact = 0.0
        
        for player in injured_players:
            if player.importance_rating > 0.8:
                concerns.append(f"Key player {player.name} unavailable")
                readiness_impact += 0.1
            
            if player.severity == "severe":
                readiness_impact += 0.05
        
        readiness_score = max(0.3, 1.0 - readiness_impact)
        
        return {
            "readiness_score": readiness_score,
            "concerns": concerns,
            "adaptation_time": "minimal" if readiness_score > 0.8 else "significant",
            "risk_level": "low" if readiness_score > 0.7 else "high"
        }
    
    def _calculate_confidence_impact(self, impact_analysis: Dict[str, Any]) -> float:
        """Calcula impacto en la confianza de la predicción"""
        overall_impact = impact_analysis.get("overall_team_strength", 0.0)
        key_players = impact_analysis.get("key_players_injured", 0)
        
        # Mayor impacto de lesiones = menor confianza en predicción
        confidence_reduction = overall_impact + (key_players * 0.05)
        
        return min(confidence_reduction, 0.3)  # Máximo 30% de reducción en confianza
    
    # Métodos auxiliares
    
    def _get_mock_injury_data(self, team_id: int) -> List[Dict[str, Any]]:
        """Genera datos de lesiones simulados (reemplazar con API real)"""
        # Simular lesiones basado en estadísticas realistas
        import random
        
        injuries = []
        num_injuries = random.randint(0, 4)  # 0-4 lesiones por equipo
        
        mock_players = [
            {"player_id": 1001, "name": "Star Striker", "position": "ST", "market_value": 50000000, "minutes_played": 2500},
            {"player_id": 1002, "name": "Key Midfielder", "position": "CAM", "market_value": 35000000, "minutes_played": 2200},
            {"player_id": 1003, "name": "Main Defender", "position": "CB", "market_value": 25000000, "minutes_played": 2400},
            {"player_id": 1004, "name": "Backup Winger", "position": "RW", "market_value": 15000000, "minutes_played": 1200},
        ]
        
        for i in range(num_injuries):
            if i < len(mock_players):
                player = mock_players[i].copy()
                player.update({
                    "injury_type": random.choice(["muscle", "knee", "ankle", "hamstring"]),
                    "severity": random.choice(["minor", "moderate", "severe"]),
                    "expected_return": datetime.now() + timedelta(days=random.randint(7, 60)),
                    "replacement_quality": random.uniform(0.4, 0.9)
                })
                injuries.append(player)
        
        return injuries
    
    def _calculate_player_importance(self, position: str, market_value: int, minutes_played: int) -> float:
        """Calcula importancia del jugador"""
        base_importance = self.position_importance.get(position, 0.5)
        
        # Ajustar por valor de mercado
        value_factor = min(market_value / 50000000, 1.0)  # Normalizar a 50M máximo
        
        # Ajustar por minutos jugados
        minutes_factor = min(minutes_played / 3000, 1.0)  # Normalizar a 3000 min máximo
        
        importance = base_importance * (0.5 + value_factor * 0.3 + minutes_factor * 0.2)
        
        return min(importance, 1.0)
    
    def _get_severity_multiplier(self, severity: str) -> float:
        """Obtiene multiplicador por severidad de lesión"""
        multipliers = {
            "minor": 0.5,
            "moderate": 0.8,
            "severe": 1.0
        }
        return multipliers.get(severity, 0.8)
    
    def _get_severity_breakdown(self, injured_players: List[InjuredPlayer]) -> Dict[str, int]:
        """Obtiene breakdown por severidad"""
        breakdown = {"minor": 0, "moderate": 0, "severe": 0}
        for player in injured_players:
            breakdown[player.severity] += 1
        return breakdown
    
    def _analyze_positional_coverage(self, injured_players: List[InjuredPlayer], team_id: int) -> Dict[str, str]:
        """Analiza cobertura posicional"""
        coverage = {}
        for player in injured_players:
            coverage[player.position] = "weak" if player.replacement_quality < 0.6 else "adequate"
        return coverage
    
    def _predict_style_change(self, injured_players: List[InjuredPlayer]) -> str:
        """Predice cambio de estilo de juego"""
        if any(p.position == "CAM" and p.importance_rating > 0.8 for p in injured_players):
            return "more_direct"
        elif any(p.position in ["ST", "CF"] for p in injured_players):
            return "less_offensive"
        elif any(p.position in ["CB", "CDM"] for p in injured_players):
            return "more_defensive"
        else:
            return "minimal_change"
    
    def _identify_vulnerability_areas(self, injured_players: List[InjuredPlayer]) -> List[str]:
        """Identifica áreas de vulnerabilidad"""
        vulnerabilities = []
        positions = [p.position for p in injured_players]
        
        if "GK" in positions:
            vulnerabilities.append("goalkeeper_inexperience")
        if len([p for p in positions if p in ["CB", "LB", "RB"]]) >= 2:
            vulnerabilities.append("defensive_instability")
        if "CAM" in positions:
            vulnerabilities.append("lack_of_creativity")
        if "ST" in positions:
            vulnerabilities.append("reduced_goal_threat")
        
        return vulnerabilities
    
    def _player_to_dict(self, player: InjuredPlayer) -> Dict[str, Any]:
        """Convierte jugador a diccionario"""
        return {
            "player_id": player.player_id,
            "name": player.name,
            "position": player.position,
            "importance_rating": player.importance_rating,
            "injury_type": player.injury_type,
            "severity": player.severity,
            "expected_return": player.expected_return.isoformat() if player.expected_return else None,
            "replacement_quality": player.replacement_quality
        }
    
    def _get_zero_impact(self) -> Dict[str, Any]:
        """Retorna impacto cero cuando no hay lesiones"""
        return {
            "attacking_impact": 0.0,
            "defensive_impact": 0.0,
            "midfield_creativity": 0.0,
            "set_piece_threat": 0.0,
            "leadership_impact": 0.0,
            "overall_team_strength": 0.0,
            "total_players_injured": 0,
            "key_players_injured": 0,
            "severity_breakdown": {"minor": 0, "moderate": 0, "severe": 0},
            "positional_coverage": {}
        }
    
    def _get_default_injury_report(self, team_id: int, fixture_id: int) -> Dict[str, Any]:
        """Retorna reporte por defecto en caso de error"""
        return {
            "team_id": team_id,
            "fixture_id": fixture_id,
            "analysis_timestamp": datetime.now().isoformat(),
            "injured_players": [],
            "impact_analysis": self._get_zero_impact(),
            "tactical_adjustments": {"formation_change": False, "tactical_adjustments": []},
            "replacement_analysis": {"replacement_quality_avg": 1.0, "depth_concerns": []},
            "historical_performance": {"matches_without": 0, "win_rate": 0.5, "goals_avg": 1.5},
            "match_readiness": {"readiness_score": 1.0, "concerns": []},
            "confidence_impact": 0.0,
            "error": "Could not fetch injury data, using defaults"
        }

# Función de utilidad para integración fácil
def get_injury_impact_for_prediction(team_id: int, fixture_id: int) -> Dict[str, Any]:
    """
    Función de utilidad para obtener impacto de lesiones en predicciones
    
    Args:
        team_id: ID del equipo
        fixture_id: ID del partido
        
    Returns:
        Diccionario con factores de ajuste para aplicar a predicciones
    """
    analyzer = RealTimeInjuryAnalyzer()
    report = analyzer.get_comprehensive_injury_report(team_id, fixture_id)
    
    impact = report["impact_analysis"]
    
    return {
        "xg_multiplier": 1.0 - impact["attacking_impact"],
        "defensive_multiplier": 1.0 + impact["defensive_impact"], 
        "confidence_adjustment": -report["confidence_impact"],
        "tactical_disruption": impact["overall_team_strength"],
        "key_adjustments": {
            "goals_expected": -impact["attacking_impact"],
            "goals_conceded_expected": impact["defensive_impact"],
            "corners_impact": -impact["set_piece_threat"],
            "cards_impact": impact["leadership_impact"]  # Menos liderazgo = más tarjetas
        }
    }

if __name__ == "__main__":
    # Ejemplo de uso
    analyzer = RealTimeInjuryAnalyzer()
    
    # Analizar lesiones para un equipo específico
    team_id = 33  # Manchester United
    fixture_id = 12345
    
    report = analyzer.get_comprehensive_injury_report(team_id, fixture_id)
    
    print("=== INJURY IMPACT REPORT ===")
    print(f"Team ID: {report['team_id']}")
    print(f"Injured Players: {len(report['injured_players'])}")
    print(f"Overall Impact: {report['impact_analysis']['overall_team_strength']:.2%}")
    print(f"Confidence Impact: {report['confidence_impact']:.2%}")
    
    if report['injured_players']:
        print("\nKey Injuries:")
        for player in report['injured_players']:
            print(f"  - {player['name']} ({player['position']}) - {player['severity']}")
    
    print(f"\nMatch Readiness: {report['match_readiness']['readiness_score']:.2%}")
