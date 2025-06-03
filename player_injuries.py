# player_injuries.py
import logging
import pandas as pd
from data import FootballAPI
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class InjuryAnalyzer:
    """
    Analiza y procesa datos de lesiones de jugadores para incorporarlos
    en las predicciones.
    """
    
    def __init__(self):
        self.api = FootballAPI()
        
    def get_team_injuries(self, team_id: int, fixture_id: Optional[int] = None) -> Dict:
        """
        Obtiene las lesiones actuales de un equipo para un partido específico o en general.
        
        Args:
            team_id: ID del equipo
            fixture_id: ID del partido (opcional)
            
        Returns:
            Diccionario con métricas de lesiones
        """
        try:
            # Determinar los parámetros de consulta según si hay fixture_id
            if fixture_id:
                params = {'fixture': fixture_id, 'team': team_id}
            else:
                params = {'team': team_id}
                
            # Obtener datos de lesiones desde la API
            injuries_data = self.api._make_request('injuries', params)
            injuries = injuries_data.get('response', [])
            
            if not injuries:
                return self._empty_injury_metrics()
            
            # Procesar datos de lesiones
            return self._process_injuries(injuries)
            
        except Exception as e:
            logger.error(f"Error obteniendo lesiones para equipo {team_id}: {e}")
            return self._empty_injury_metrics()
    
    def _process_injuries(self, injuries: List[Dict]) -> Dict[str, Union[int, float]]:
        """
        Procesa lista de lesiones para extraer métricas relevantes.
        
        Args:
            injuries: Lista de datos de lesiones
            
        Returns:
            Diccionario con métricas procesadas
        """
        metrics = {
            'total_injured': len(injuries),
            'key_players_injured': 0,
            'days_out_avg': 0.0,
            'defensive_injuries': 0,
            'offensive_injuries': 0,
            'severity_score': 0.0
        }
        
        if not injuries:
            return metrics
        
        # Posiciones defensivas
        defensive_positions = ['Goalkeeper', 'Defender']
        # Posiciones ofensivas
        offensive_positions = ['Attacker', 'Midfielder']
        
        days_out_total = 0
        severity_total = 0.0
        
        for injury in injuries:
            player = injury.get('player', {})
            position = player.get('position', '')
            
            # Contar lesiones por posición
            if position in defensive_positions:
                metrics['defensive_injuries'] += 1
            elif position in offensive_positions:
                metrics['offensive_injuries'] += 1
            
            # Determinar si es jugador clave (asumimos que jugadores con
            # más minutos son más importantes)
            # Esta información tendría que venir de otra fuente o ser estimada
            if player.get('minutes', 0) > 1000:  # Por ejemplo, más de 1000 minutos jugados
                metrics['key_players_injured'] += 1
            
            # Calcular días fuera aproximados
            injury_type = injury.get('type', '')
            injury_start = injury.get('start', '')
            
            # Estimar severidad basada en el tipo de lesión
            severity = self._estimate_severity(injury_type)
            severity_total += severity
            
            # Añadir a días fuera totales
            if injury_start:
                try:
                    start_date = datetime.fromisoformat(injury_start)
                    days_out = (datetime.now() - start_date).days
                    days_out_total += days_out
                except (ValueError, TypeError):
                    pass
        
        # Calcular promedios
        if len(injuries) > 0:
            metrics['days_out_avg'] = days_out_total / len(injuries)
            metrics['severity_score'] = severity_total / len(injuries)
        
        return metrics
    
    def _estimate_severity(self, injury_type: str) -> float:
        """
        Estima la severidad de una lesión basada en su tipo.
        
        Args:
            injury_type: Tipo de lesión
            
        Returns:
            Puntuación de severidad (0-1)
        """
        # Mapeo simple de tipos de lesión a severidad
        severity_map = {
            'Knock': 0.2,
            'Bruised': 0.3,
            'Strain': 0.4,
            'Sprain': 0.5,
            'Muscle': 0.6,
            'Thigh': 0.5,
            'Hamstring': 0.7,
            'Calf': 0.6,
            'Ankle': 0.6,
            'Knee': 0.8,
            'ACL': 0.9,
            'MCL': 0.8,
            'Broken': 0.85,
            'Fracture': 0.8,
            'Surgery': 0.9,
            'Suspended': 0.6,
            'Illness': 0.4,
            'Corona': 0.5,
            'COVID-19': 0.5
        }
        
        # Buscar coincidencias parciales en el tipo de lesión
        severity = 0.3  # Valor por defecto para lesiones desconocidas
        for key, value in severity_map.items():
            if key.lower() in injury_type.lower():
                severity = max(severity, value)  # Tomar la severidad más alta si hay múltiples coincidencias
        
        return severity
    
    def _empty_injury_metrics(self) -> Dict[str, Union[int, float]]:
        """Retorna un diccionario vacío con las métricas de lesiones"""
        return {
            'total_injured': 0,
            'key_players_injured': 0,
            'days_out_avg': 0.0,
            'defensive_injuries': 0,
            'offensive_injuries': 0,
            'severity_score': 0.0
        }
    
    def get_team_injury_impact(self, team_id: int, fixture_id: Optional[int] = None) -> float:
        """
        Calcula el impacto de las lesiones en un equipo en una escala de 0 a 1.
        
        Args:
            team_id: ID del equipo
            fixture_id: ID del partido (opcional)
            
        Returns:
            Puntuación de impacto (0-1) donde 0 = sin impacto, 1 = impacto extremo
        """
        metrics = self.get_team_injuries(team_id, fixture_id)
        
        # Ponderaciones para cada métrica
        weights = {
            'total_injured': 0.15,
            'key_players_injured': 0.35,
            'defensive_injuries': 0.2,
            'offensive_injuries': 0.15,
            'severity_score': 0.15
        }
        
        # Valores normalizados como variables individuales en lugar de un diccionario
        norm_total_injured = min(metrics['total_injured'] / 5, 1.0)
        norm_key_players = min(metrics['key_players_injured'] / 3, 1.0)
        norm_defensive = min(metrics['defensive_injuries'] / 3, 1.0)
        norm_offensive = min(metrics['offensive_injuries'] / 3, 1.0)
        norm_severity = metrics['severity_score']  # Ya está normalizado (0-1)
        
        # Calcular puntuación ponderada
        impact_score = (
            norm_total_injured * weights['total_injured'] +
            norm_key_players * weights['key_players_injured'] +
            norm_defensive * weights['defensive_injuries'] +
            norm_offensive * weights['offensive_injuries'] +
            norm_severity * weights['severity_score']
        )
        
        return impact_score
