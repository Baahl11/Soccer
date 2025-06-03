from typing import Dict, Any, Optional, List
import numpy as np

class FormationAnalyzer:
    """
    Analiza las formaciones de los equipos y su impacto táctico.
    """
    
    COMMON_FORMATIONS = {
        '4-4-2': {
            'strengths': ['organizacion defensiva', 'transiciones rapidas'],
            'weaknesses': ['dificultad para atacar', 'posesión limitada'],
            'counter_formations': ['3-4-3', '4-2-3-1', '4-3-3']
        },
        '4-3-3': {
            'strengths': ['presión alta', 'ataque fluido', 'amplitud'],
            'weaknesses': ['exposicion defensiva', 'espacios entre líneas'],
            'counter_formations': ['4-5-1', '3-5-2', '4-4-2']
        },
        '3-5-2': {
            'strengths': ['solidez defensiva', 'control del medio campo'],
            'weaknesses': ['vulnerabilidad por las bandas', 'transición defensiva'],
            'counter_formations': ['4-3-3', '4-2-3-1', '3-4-3']
        },
        '4-2-3-1': {
            'strengths': ['balance táctico', 'presión alta', 'creatividad'],
            'weaknesses': ['dependencia del mediapunta', 'espacios entre líneas'],
            'counter_formations': ['4-3-3', '3-5-2', '4-4-2']
        }
    }

    def __init__(self):
        """Inicializa el analizador de formaciones."""
        pass

    def analyze_formation_matchup(self, home_formation: str, away_formation: str) -> Dict[str, Any]:
        """
        Analiza el enfrentamiento táctico entre dos formaciones.
        
        Args:
            home_formation: Formación del equipo local
            away_formation: Formación del equipo visitante
            
        Returns:
            Diccionario con análisis del matchup
        """
        try:
            # Normalizar formaciones
            if home_formation not in self.COMMON_FORMATIONS:
                home_formation = '4-4-2'
            if away_formation not in self.COMMON_FORMATIONS:
                away_formation = '4-4-2'
            
            # Verificar ventajas directas
            home_advantage = away_formation in self.COMMON_FORMATIONS[home_formation]['counter_formations']
            away_advantage = home_formation in self.COMMON_FORMATIONS[away_formation]['counter_formations']
            
            # Analizar zonas de ventaja
            zones_analysis = self._analyze_formation_zones(home_formation, away_formation)
            
            # Preparar resultado
            result = {
                'home_formation': home_formation,
                'away_formation': away_formation,
                'home_formation_details': self.COMMON_FORMATIONS[home_formation],
                'away_formation_details': self.COMMON_FORMATIONS[away_formation],
                'tactical_advantage': 'home' if home_advantage and not away_advantage else 
                                     'away' if away_advantage and not home_advantage else 'neutral',
                'advantage_score': 0.2 if home_advantage and not away_advantage else 
                                  -0.2 if away_advantage and not home_advantage else 0.0,
                'zones_analysis': zones_analysis
            }
            
            return result
            
        except Exception as e:
            print(f"Error en analyze_formation_matchup: {str(e)}")
            return {
                'home_formation': home_formation,
                'away_formation': away_formation,
                'tactical_advantage': 'neutral',
                'advantage_score': 0.0,
                'zones_analysis': {}
            }

    def _analyze_formation_zones(self, home_formation: str, away_formation: str) -> Dict[str, Any]:
        """
        Analiza las ventajas por zonas del campo según las formaciones.
        """
        # Simplificado para el MVP
        zones = {
            'wings': 0.0,
            'central': 0.0,
            'defensive': 0.0,
            'offensive': 0.0
        }
        
        # Análisis básico por formación
        home_nums = list(map(int, home_formation.split('-')))
        away_nums = list(map(int, away_formation.split('-')))
        
        # Ventaja en las bandas
        zones['wings'] = 0.1 if sum(home_nums[:-1]) > sum(away_nums[:-1]) else -0.1
        
        # Ventaja en el centro
        zones['central'] = 0.1 if home_nums[1] > away_nums[1] else -0.1
        
        return zones
    
    def predict_corner_impact(self, formation: str) -> float:
        """
        Predice el impacto de una formación en la generación de corners.
        
        Args:
            formation: Formación a analizar (ej: '4-3-3')
            
        Returns:
            Float entre -1 y 1 indicando el impacto en corners
        """
        # Normalizar formación
        if formation not in self.COMMON_FORMATIONS:
            formation = '4-4-2'
            
        # Factores que aumentan corners
        corner_factors = {
            'wing-play': 0.3,  # Juego por bandas genera más corners
            'high-press': 0.2,  # Presión alta recupera balón en zonas peligrosas
            'attacking': 0.25,  # Estilo ofensivo genera más corners
            'possession': 0.15  # Más posesión correlaciona con más corners
        }
        
        formation_details = self.COMMON_FORMATIONS[formation]
        total_impact = 0.0
        
        # Sumar impactos positivos
        for factor, weight in corner_factors.items():
            if factor in formation_details.get('strengths', []):
                total_impact += weight
                
        # Restar impactos negativos
        if 'defensive' in formation_details.get('style', []):
            total_impact -= 0.2
        if 'counter-attack' in formation_details.get('style', []):
            total_impact -= 0.1
            
        # Normalizar a rango [-1, 1]
        return max(-1.0, min(1.0, total_impact))