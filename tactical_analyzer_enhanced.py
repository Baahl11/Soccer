"""
Sistema de análisis táctico mejorado con base de datos de equipos

Este módulo implementa un análisis táctico mejorado que utiliza una base de datos
específica por equipo y un sistema de fallback gradual.

Autor: Equipo de Desarrollo
Fecha: Mayo 22, 2025
"""

import logging
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
import random  # Solo para simulación en casos de falta de datos parciales

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='tactical_analysis.log',
    filemode='a'
)

logger = logging.getLogger('tactical_analyzer')

# Cargar base de datos táctica de equipos
TEAM_DB_PATH = Path(__file__).parent / "data" / "tactical_teams.json"
TEAM_DB: Dict[str, Dict[str, Any]] = {}

if TEAM_DB_PATH.exists():
    try:
        with open(TEAM_DB_PATH, 'r', encoding='utf-8') as f:
            TEAM_DB = json.load(f)
        logger.info(f"Cargada base de datos táctica con {len(TEAM_DB)} equipos")
    except Exception as e:
        logger.error(f"Error cargando base de datos táctica: {str(e)}")


class TacticalAnalyzerEnhanced:
    """Analizador táctico mejorado con base de datos de equipos."""
    
    def __init__(self):
        """Inicializa el analizador táctico."""
        self.team_db = TEAM_DB
        
        # Estadísticas de uso
        self.stats = {
            "total_analyses": 0,
            "full_data_analyses": 0,
            "partial_data_analyses": 0,
            "default_analyses": 0
        }
        
        logger.info("TacticalAnalyzerEnhanced inicializado")
    
    def get_tactical_analysis(self, home_team_id: Union[int, str], away_team_id: Union[int, str], 
                              home_name: Optional[str] = None, away_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Obtiene análisis táctico para un partido, con sistema de fallback en cascada.
        
        Args:
            home_team_id: ID del equipo local
            away_team_id: ID del equipo visitante
            home_name: Nombre del equipo local (opcional)
            away_name: Nombre del equipo visitante (opcional)
            
        Returns:
            Análisis táctico detallado
        """
        # Convertir IDs a strings para buscar en la BD
        home_id_str = str(home_team_id)
        away_id_str = str(away_team_id)
        
        self.stats["total_analyses"] += 1
        
        # 1. Verificar si tenemos análisis completo en la API externa
        api_analysis = self._fetch_from_tactical_api(home_team_id, away_team_id)
        if api_analysis:
            self.stats["full_data_analyses"] += 1
            logger.info(f"Usando análisis táctico de API para {home_team_id} vs {away_team_id}")
            return api_analysis
        
        # 2. Verificar si tenemos ambos equipos en la base de datos
        home_data = self.team_db.get(home_id_str, {})
        away_data = self.team_db.get(away_id_str, {})
        
        if home_data and away_data:
            self.stats["full_data_analyses"] += 1
            logger.info(f"Usando datos tácticos de BD local para {home_team_id} vs {away_team_id}")
            return self._create_analysis_from_db_data(home_data, away_data)
        
        # 3. Verificar si tenemos al menos un equipo en la base de datos
        if home_data or away_data:
            self.stats["partial_data_analyses"] += 1
            logger.info(f"Usando datos tácticos parciales para {home_team_id} vs {away_team_id}")
            return self._create_analysis_from_partial_data(home_data, away_data, home_id_str, away_id_str, home_name, away_name)
        
        # 4. Intentar inferir de estadísticas históricas
        inferred_analysis = self._infer_tactical_analysis(home_team_id, away_team_id)
        if inferred_analysis:
            self.stats["partial_data_analyses"] += 1
            logger.info(f"Usando análisis inferido para {home_team_id} vs {away_team_id}")
            return inferred_analysis
        
        # 5. Como último recurso, usar análisis predeterminado
        self.stats["default_analyses"] += 1
        logger.warning(f"Usando análisis táctico genérico para {home_team_id} vs {away_team_id}")
        return self._create_default_analysis(home_id_str, away_id_str, home_name, away_name)
    
    def _fetch_from_tactical_api(self, home_team_id: Union[int, str], away_team_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Intenta obtener análisis táctico de la API externa.
        En una implementación real, esto conectaría con un servicio de análisis táctico.
        
        Args:
            home_team_id: ID del equipo local
            away_team_id: ID del equipo visitante
            
        Returns:
            Análisis táctico o None si no está disponible
        """
        # En una implementación real, aquí se conectaría con la API
        # Por ahora, simplemente devolvemos None para simular que no hay datos
        return None
    
    def _create_analysis_from_db_data(self, home_data: Dict[str, Any], away_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea análisis táctico completo basado en datos de la BD para ambos equipos.
        
        Args:
            home_data: Datos del equipo local
            away_data: Datos del equipo visitante
            
        Returns:
            Análisis táctico detallado
        """
        # Extraer estilos tácticos
        home_style = home_data.get("style", {})
        away_style = away_data.get("style", {})
        
        # Extraer formaciones
        home_formation = home_data.get("formation", "4-4-2")
        away_formation = away_data.get("formation", "4-4-2")
        
        # Determinar ventajas tácticas
        advantages = self._analyze_tactical_matchup(home_style, away_style, home_formation, away_formation)
        
        # Crear análisis completo
        analysis = {
            "tactical_style": {
                "home": home_style,
                "away": away_style
            },
            "expected_formations": {
                "home": home_formation,
                "away": away_formation
            },
            "key_battles": advantages["key_battles"],
            "strengths": {
                "home": home_data.get("strengths", ["No hay datos detallados disponibles"]),
                "away": away_data.get("strengths", ["No hay datos detallados disponibles"])
            },
            "weaknesses": {
                "home": home_data.get("weaknesses", ["No hay datos detallados disponibles"]),
                "away": away_data.get("weaknesses", ["No hay datos detallados disponibles"])
            },
            "tactical_recommendation": advantages["recommendation"],
            "matchup_analysis": {
                "description": advantages["description"],
                "advantage": advantages["advantage"]
            },
            "data_quality": "complete"  # Metadato para seguimiento
        }
        
        return analysis
    
    def _create_analysis_from_partial_data(self, home_data: Dict[str, Any], away_data: Dict[str, Any], 
                                           home_id: str, away_id: str, home_name: Optional[str], away_name: Optional[str]) -> Dict[str, Any]:
        """
        Crea análisis táctico basado en datos parciales (al menos un equipo con datos).
        
        Args:
            home_data: Datos del equipo local (puede estar vacío)
            away_data: Datos del equipo visitante (puede estar vacío)
            home_id: ID del equipo local
            away_id: ID del equipo visitante
            home_name: Nombre del equipo local
            away_name: Nombre del equipo visitante
            
        Returns:
            Análisis táctico con datos parciales
        """
        # Determinar qué equipo tiene datos
        has_home_data = bool(home_data)
        has_away_data = bool(away_data)
        
        # Crear valores por defecto para datos faltantes, pero un poco más variados
        default_styles = [
            {"possession": "medio-bajo", "pressing": "medio", "counter_attacking": "alto"},
            {"possession": "medio", "pressing": "medio-alto", "counter_attacking": "medio"},
            {"possession": "alto", "pressing": "alto", "counter_attacking": "bajo"},
            {"possession": "medio-alto", "pressing": "bajo", "counter_attacking": "medio-alto"}
        ]
        
        default_formations = ["4-4-2", "4-2-3-1", "3-5-2", "4-3-3", "5-3-2"]
        
        # Obtener o generar datos tácticos
        home_style = home_data.get("style", random.choice(default_styles))
        away_style = away_data.get("style", random.choice(default_styles))
        
        home_formation = home_data.get("formation", random.choice(default_formations))
        away_formation = away_data.get("formation", random.choice(default_formations))
        
        # Analizar matchup con los datos disponibles
        advantages = self._analyze_tactical_matchup(home_style, away_style, home_formation, away_formation)
        
        # Crear descripción según los datos disponibles
        if has_home_data and not has_away_data:
            data_quality = "partial_home"
            description = f"Análisis basado en datos conocidos de {home_data.get('name', home_name or home_id)}"
        elif has_away_data and not has_home_data:
            data_quality = "partial_away"
            description = f"Análisis basado en datos conocidos de {away_data.get('name', away_name or away_id)}"
        else:
            data_quality = "inferred"
            description = "Análisis basado en datos inferidos"
        
        # Crear análisis completo
        analysis = {
            "tactical_style": {
                "home": home_style,
                "away": away_style
            },
            "expected_formations": {
                "home": home_formation,
                "away": away_formation
            },
            "key_battles": advantages["key_battles"],
            "strengths": {
                "home": home_data.get("strengths", ["Datos limitados"]),
                "away": away_data.get("strengths", ["Datos limitados"])
            },
            "weaknesses": {
                "home": home_data.get("weaknesses", ["Análisis limitado"]),
                "away": away_data.get("weaknesses", ["Análisis limitado"])
            },
            "tactical_recommendation": advantages["recommendation"],
            "matchup_analysis": {
                "description": advantages["description"],
                "advantage": advantages["advantage"]
            },
            "data_quality": data_quality  # Metadato para seguimiento
        }
        
        return analysis
    
    def _infer_tactical_analysis(self, home_team_id: Union[int, str], away_team_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Intenta inferir análisis táctico basado en estadísticas históricas.
        En una implementación real, esto analizaría datos históricos de partidos.
        
        Args:
            home_team_id: ID del equipo local
            away_team_id: ID del equipo visitante
            
        Returns:
            Análisis táctico inferido o None si no es posible
        """
        # En una implementación real, aquí se analizarían estadísticas históricas
        # Por ahora, devolvemos None para simular que no es posible
        return None
    
    def _create_default_analysis(self, home_id: str, away_id: str, home_name: Optional[str], away_name: Optional[str]) -> Dict[str, Any]:
        """
        Crea análisis táctico predeterminado cuando no hay datos disponibles.
        
        Args:
            home_id: ID del equipo local
            away_id: ID del equipo visitante
            home_name: Nombre del equipo local
            away_name: Nombre del equipo visitante
            
        Returns:
            Análisis táctico predeterminado
        """
        # Al menos variamos un poco los valores por defecto
        home_form = random.choice(["4-4-2", "4-3-3", "4-2-3-1"])
        away_form = random.choice(["4-3-3", "4-2-3-1", "3-5-2"])
        
        # Evitar exactamente la misma formación
        while away_form == home_form:
            away_form = random.choice(["4-3-3", "4-2-3-1", "3-5-2"])
        
        analysis = {
            "tactical_style": {
                "home": {"possession": "medio", "pressing": "medio", "counter_attacking": "medio"},
                "away": {"possession": "medio", "pressing": "medio", "counter_attacking": "medio"}
            },
            "expected_formations": {
                "home": home_form,
                "away": away_form
            },
            "key_battles": [
                {
                    "zone": "Medio campo",
                    "importance": "Alta",
                    "advantage": "Neutral",
                    "key_factors": ["Control de posesión", "Presión en mediocampo"]
                }
            ],
            "strengths": {
                "home": ["Datos tácticos no disponibles"],
                "away": ["Datos tácticos no disponibles"]
            },
            "weaknesses": {
                "home": ["Datos tácticos no disponibles"],
                "away": ["Datos tácticos no disponibles"]
            },
            "tactical_recommendation": "Sistema táctico en desarrollo",
            "matchup_analysis": {
                "description": "Análisis basado en datos limitados",
                "advantage": "Neutral"
            },
            "data_quality": "default"  # Metadato para seguimiento
        }
        
        return analysis
    
    def _analyze_tactical_matchup(self, home_style: Dict[str, str], away_style: Dict[str, str], 
                                 home_formation: str, away_formation: str) -> Dict[str, Any]:
        """
        Analiza el enfrentamiento táctico entre dos equipos.
        
        Args:
            home_style: Estilo táctico del equipo local
            away_style: Estilo táctico del equipo visitante
            home_formation: Formación del equipo local
            away_formation: Formación del equipo visitante
            
        Returns:
            Análisis del enfrentamiento táctico
        """
        # Determinar ventajas tácticas
        home_possession = self._style_to_numeric(home_style.get("possession", "medio"))
        away_possession = self._style_to_numeric(away_style.get("possession", "medio"))
        
        home_pressing = self._style_to_numeric(home_style.get("pressing", "medio"))
        away_pressing = self._style_to_numeric(away_style.get("pressing", "medio"))
        
        home_counter = self._style_to_numeric(home_style.get("counter_attacking", "medio"))
        away_counter = self._style_to_numeric(away_style.get("counter_attacking", "medio"))
        
        # Calcular diferencias
        possession_diff = home_possession - away_possession
        pressing_diff = home_pressing - away_pressing
        counter_diff = home_counter - away_counter
        
        # Determinar ventajas y crear key battles
        key_battles = []
        advantage = "Neutral"
        advantage_score = 0
        
        if abs(possession_diff) >= 0.5:
            team = "Local" if possession_diff > 0 else "Visitante"
            key_battles.append({
                "zone": "Posesión general",
                "importance": "Alta",
                "advantage": team,
                "key_factors": ["Control de balón", "Creación de juego"]
            })
            advantage_score += (1 if team == "Local" else -1)
        
        if abs(pressing_diff) >= 0.5:
            team = "Local" if pressing_diff > 0 else "Visitante"
            key_battles.append({
                "zone": "Presión y recuperación",
                "importance": "Alta",
                "advantage": team,
                "key_factors": ["Intensidad", "Recuperaciones en campo contrario"]
            })
            advantage_score += (1 if team == "Local" else -1)
        
        if abs(counter_diff) >= 0.5:
            team = "Local" if counter_diff > 0 else "Visitante"
            key_battles.append({
                "zone": "Contragolpes",
                "importance": "Media",
                "advantage": team,
                "key_factors": ["Velocidad en transiciones", "Efectividad"]
            })
            advantage_score += (0.5 if team == "Local" else -0.5)
        
        # Analizar formaciones
        formation_advantage = self._analyze_formations(home_formation, away_formation)
        if formation_advantage:
            key_battles.append(formation_advantage)
            advantage_score += (0.5 if formation_advantage["advantage"] == "Local" else -0.5 if formation_advantage["advantage"] == "Visitante" else 0)
        
        # Determinar ventaja general
        if advantage_score > 1:
            advantage = "Local"
            description = "Ventaja táctica para el equipo local"
            recommendation = "El equipo local debería explotar su ventaja en posesión y presión"
        elif advantage_score < -1:
            advantage = "Visitante"
            description = "Ventaja táctica para el equipo visitante"
            recommendation = "El equipo visitante debería imponer su estilo de juego"
        else:
            advantage = "Neutral"
            description = "Enfrentamiento táctico equilibrado"
            recommendation = "Partido muy parejo tácticamente, los detalles serán cruciales"
        
        # Si no hay key battles, añadir una genérica
        if not key_battles:
            key_battles.append({
                "zone": "Medio campo",
                "importance": "Alta",
                "advantage": "Neutral",
                "key_factors": ["Control del ritmo", "Duelos individuales"]
            })
        
        return {
            "advantage": advantage,
            "description": description,
            "recommendation": recommendation,
            "key_battles": key_battles
        }
    
    def _style_to_numeric(self, style: str) -> float:
        """Convierte un estilo táctico textual en un valor numérico."""
        style_values = {
            "muy bajo": 0.0,
            "bajo": 0.25,
            "medio-bajo": 0.4,
            "medio": 0.5,
            "medio-alto": 0.6,
            "alto": 0.75,
            "muy alto": 1.0
        }
        return style_values.get(style.lower(), 0.5)
    
    def _analyze_formations(self, home_formation: str, away_formation: str) -> Optional[Dict[str, Any]]:
        """
        Analiza la relación entre formaciones para identificar ventajas.
        
        Args:
            home_formation: Formación del equipo local
            away_formation: Formación del equipo visitante
            
        Returns:
            Información sobre las ventajas de formación o None
        """
        # Análisis simplificado de ventajas entre formaciones comunes
        formation_matchups = {
            # 4-4-2 vs otros
            ("4-4-2", "4-3-3"): ("Visitante", "Superioridad en el medio campo", "Media"),
            ("4-4-2", "4-2-3-1"): ("Visitante", "Mayor control en zonas de creación", "Media"),
            ("4-4-2", "3-5-2"): ("Local", "Mayor presencia en bandas", "Media"),
            
            # 4-3-3 vs otros
            ("4-3-3", "4-4-2"): ("Local", "Superioridad en el medio campo", "Media"),
            ("4-3-3", "4-2-3-1"): ("Neutral", "Sistemas equilibrados", "Baja"),
            ("4-3-3", "3-5-2"): ("Visitante", "Mayor control del mediocampo", "Media"),
            
            # 4-2-3-1 vs otros
            ("4-2-3-1", "4-4-2"): ("Local", "Mayor control en zonas de creación", "Media"),
            ("4-2-3-1", "4-3-3"): ("Neutral", "Sistemas equilibrados", "Baja"),
            ("4-2-3-1", "3-5-2"): ("Visitante", "Mayor presencia en mediocampo", "Media"),
            
            # 3-5-2 vs otros
            ("3-5-2", "4-4-2"): ("Visitante", "Mayor presencia en bandas", "Media"),
            ("3-5-2", "4-3-3"): ("Local", "Mayor control del mediocampo", "Media"),
            ("3-5-2", "4-2-3-1"): ("Local", "Mayor presencia en mediocampo", "Media")
        }
        
        # Buscar matchup directo
        key = (home_formation, away_formation)
        if key in formation_matchups:
            team, reason, importance = formation_matchups[key]
            return {
                "zone": "Formaciones tácticas",
                "importance": importance,
                "advantage": team,
                "key_factors": [reason, "Disposición táctica"]
            }
        
        # Buscar matchup inverso
        key = (away_formation, home_formation)
        if key in formation_matchups:
            team, reason, importance = formation_matchups[key]
            # Invertir el equipo con ventaja
            inverted_team = "Local" if team == "Visitante" else "Visitante" if team == "Local" else "Neutral"
            return {
                "zone": "Formaciones tácticas",
                "importance": importance,
                "advantage": inverted_team,
                "key_factors": [reason, "Disposición táctica"]
            }
        
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """Devuelve estadísticas de uso del analizador táctico."""
        return self.stats


# Función para integrar en el sistema existente
def get_tactical_analysis(home_team_id: Union[int, str], away_team_id: Union[int, str], 
                          home_name: Optional[str] = None, away_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Función de conveniencia para obtener análisis táctico.
    
    Args:
        home_team_id: ID del equipo local
        away_team_id: ID del equipo visitante
        home_name: Nombre del equipo local (opcional)
        away_name: Nombre del equipo visitante (opcional)
        
    Returns:
        Análisis táctico para el enfrentamiento
    """
    analyzer = TacticalAnalyzerEnhanced()
    return analyzer.get_tactical_analysis(home_team_id, away_team_id, home_name, away_name)


# Ejemplo de uso
if __name__ == "__main__":
    # Probar con equipos conocidos
    home_id = 541  # Real Madrid
    away_id = 529  # Barcelona
    
    analysis = get_tactical_analysis(home_id, away_id, "Real Madrid", "Barcelona")
    
    print(f"Análisis táctico para Real Madrid vs Barcelona:")
    print(f"Calidad de datos: {analysis.get('data_quality', 'desconocida')}")
    print("\nEstilos tácticos:")
    print(f"Real Madrid: {analysis['tactical_style']['home']}")
    print(f"Barcelona: {analysis['tactical_style']['away']}")
    print("\nFormaciones esperadas:")
    print(f"Real Madrid: {analysis['expected_formations']['home']}")
    print(f"Barcelona: {analysis['expected_formations']['away']}")
    
    print("\nVentajas tácticas:")
    for battle in analysis.get("key_battles", []):
        print(f"- {battle['zone']}: Ventaja {battle['advantage']} ({battle['importance']})")
        print(f"  Factores: {', '.join(battle['key_factors'])}")
    
    print(f"\nDescripción: {analysis['matchup_analysis']['description']}")
    print(f"Recomendación: {analysis['tactical_recommendation']}")
    
    # Probar con equipos desconocidos
    print("\n\nProbando con equipos no en la base de datos:")
    unknown_analysis = get_tactical_analysis(99999, 88888, "Equipo Desconocido", "Otro Desconocido")
    print(f"Calidad de datos: {unknown_analysis.get('data_quality', 'desconocida')}")
    print(f"Formaciones: {unknown_analysis['expected_formations']['home']} vs {unknown_analysis['expected_formations']['away']}")
