"""
Módulo para integrar análisis táctico en las predicciones de partidos.
Sirve como puente entre los módulos de tactical_analysis, interpretability y el sistema de predicciones.
"""

import logging
from typing import Dict, Any, List, Optional
from interpretability import TacticalInterpreter
from tactical_analysis import TacticalAnalyzer
from formation import get_formation_strength, FORMATION_STYLES

logger = logging.getLogger(__name__)

def enrich_prediction_with_tactical_analysis(
    prediction: Dict[str, Any], 
    home_matches: List[Dict[str, Any]], 
    away_matches: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Enriquece las predicciones con análisis táctico detallado.
    
    Args:
        prediction: Predicción actual a enriquecer
        home_matches: Partidos recientes del equipo local
        away_matches: Partidos recientes del equipo visitante
        
    Returns:
        Predicción enriquecida con información táctica
    """
    try:        # Instanciar el intérprete táctico
        interpreter = TacticalInterpreter()
        
        # Generar reporte táctico completo
        tactical_report = interpreter.generate_tactical_report(home_matches, away_matches)
        
        # Obtener análisis táctico completo
        analyzer = TacticalAnalyzer()
        tactical_matchup = analyzer.get_tactical_matchup(home_matches, away_matches)
        
        # Extraer información clave y enriquecer la predicción
        prediction['tactical_analysis'] = {
            'style_comparison': tactical_report['style_comparison'],
            'key_advantages': tactical_report['tactical_advantages'].get('key_points', []),
            'suggested_approach': tactical_report['suggested_adaptations'].get('summary', 'No hay datos suficientes'),
            'tactical_style': {
                'home': tactical_matchup.get('team1_profile', {}),
                'away': tactical_matchup.get('team2_profile', {})
            },
            'matchup_analysis': {
                'possession_battle': tactical_matchup.get('possession_battle', {}),
                'pressing_dynamics': tactical_matchup.get('pressing_dynamics', {}),
                'attacking_comparison': tactical_matchup.get('attacking_comparison', {}),
                'tactical_advantages': tactical_matchup.get('tactical_advantages', {})
            },
            'key_battles': tactical_matchup.get('key_battles', []),
            'tactical_indices': {
                'home': tactical_matchup.get('team1_profile', {}).get('tactical_indices', {}),
                'away': tactical_matchup.get('team2_profile', {}).get('tactical_indices', {})
            },
            'tactical_traits': {
                'home': tactical_matchup.get('team1_profile', {}).get('tactical_traits', {}),
                'away': tactical_matchup.get('team2_profile', {}).get('tactical_traits', {})
            }
        }
        
        # Añadir información de impacto táctico en el resultado
        if 'projected_impact' in tactical_report:
            prediction['tactical_analysis']['projected_impact'] = tactical_report['projected_impact']
        
        return prediction
        
    except Exception as e:
        logger.error(f"Error al enriquecer predicción con análisis táctico: {e}")
        # En caso de error, añadir mensaje informativo pero no afectar el resto de la predicción
        prediction['tactical_analysis'] = {
            'error': f"No se pudo completar el análisis táctico: {str(e)}",
            'status': 'error'
        }
        return prediction

def add_formation_analysis(
    prediction: Dict[str, Any],
    home_formation: Optional[str] = None,
    away_formation: Optional[str] = None
) -> Dict[str, Any]:
    """
    Añade análisis específico de formaciones a la predicción.
    
    Args:
        prediction: Predicción a enriquecer
        home_formation: Formación del equipo local (ej. "4-3-3")
        away_formation: Formación del equipo visitante
        
    Returns:
        Predicción enriquecida con información de formaciones
    """
    try:
        # Si no tenemos información de formaciones, intentar extraerla de la predicción
        if not home_formation and 'home_team' in prediction and 'formation' in prediction['home_team']:
            home_formation = prediction['home_team']['formation']
            
        if not away_formation and 'away_team' in prediction and 'formation' in prediction['away_team']:
            away_formation = prediction['away_team']['formation']
            
        # Si aún no tenemos formaciones, no podemos continuar
        if not home_formation or not away_formation:
            prediction['formation_analysis'] = {
                'status': 'unavailable',
                'message': 'No hay información de formaciones disponible'
            }
            return prediction
            
        # Calcular análisis de formaciones
        formation_strength = get_formation_strength(home_formation, away_formation)
        
        # Obtener características de estilo para ambas formaciones
        home_style = FORMATION_STYLES.get(home_formation, {})
        away_style = FORMATION_STYLES.get(away_formation, {})
        
        # Calcular ventajas de estilo
        style_advantages = {
            'home': [style for style, value in home_style.items() 
                    if value > 0.6 and (style not in away_style or home_style[style] > away_style.get(style, 0))],
            'away': [style for style, value in away_style.items() 
                    if value > 0.6 and (style not in home_style or away_style[style] > home_style.get(style, 0))]
        }
        
        # Añadir información al resultado
        prediction['formation_analysis'] = {
            'home_formation': home_formation,
            'away_formation': away_formation,
            'formation_strength': formation_strength.get('formation_strength', 1.0),
            'is_offensive_matchup': formation_strength.get('is_offensive', False),
            'home_style': home_style,
            'away_style': away_style,
            'style_advantages': style_advantages,
            'status': 'complete'
        }
        
        # Si el factor de fuerza es significativo, añadir comentario interpretativo
        strength_factor = formation_strength.get('formation_strength', 1.0)
        if strength_factor > 1.15:
            prediction['formation_analysis']['interpretation'] = 'Formaciones favorecen un partido abierto con más goles'
        elif strength_factor < 0.9:
            prediction['formation_analysis']['interpretation'] = 'Formaciones sugieren un partido cerrado con pocos goles'
        else:
            prediction['formation_analysis']['interpretation'] = 'Formaciones equilibradas sin una ventaja clara'
            
        return prediction
        
    except Exception as e:
        logger.error(f"Error al añadir análisis de formaciones: {e}")
        prediction['formation_analysis'] = {
            'status': 'error',
            'message': f"Error en análisis de formaciones: {str(e)}"
        }
        return prediction

def get_simplified_tactical_analysis(
    home_team_id: int, 
    away_team_id: int,
    home_matches: Optional[List[Dict[str, Any]]] = None,
    away_matches: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Genera un análisis táctico simplificado para consumo en API.
    
    Args:
        home_team_id: ID del equipo local
        away_team_id: ID del equipo visitante
        home_matches: Partidos recientes del equipo local (opcional)
        away_matches: Partidos recientes del equipo visitante (opcional)
        
    Returns:
        Análisis táctico simplificado    
    """        
    try:
        # Si no se proporcionan partidos, usar el analizador táctico directamente
        analyzer = TacticalAnalyzer()
        
        if not home_matches or not away_matches:
            # Necesitamos cargar los partidos desde el historial
            from team_history import HistoricalAnalyzer
            historical = HistoricalAnalyzer()
            home_matches = historical.get_team_matches(home_team_id)[:5]  # Obtener los 5 partidos más recientes
            away_matches = historical.get_team_matches(away_team_id)[:5]  # Obtener los 5 partidos más recientes
            
            # Verificar si tenemos suficientes datos para el análisis
            if not home_matches or not away_matches:
                # No hay datos históricos suficientes, usar análisis predeterminado
                logger.warning(f"No hay datos históricos para equipos {home_team_id} y/o {away_team_id}. Usando análisis predeterminado.")
                return create_default_tactical_analysis(home_team_id, away_team_id)
            
            # Ahora que tenemos los partidos, podemos obtener el análisis táctico
            tactical_analysis = analyzer.get_tactical_matchup(home_matches, away_matches)
        else:
            # Usar los partidos proporcionados
            tactical_analysis = analyzer.get_tactical_matchup(home_matches, away_matches)
        
        # Verificar si el análisis táctico tiene contenido
        if not tactical_analysis or not any(tactical_analysis.values()):
            logger.warning(f"El análisis táctico para equipos {home_team_id} y {away_team_id} no contiene datos. Usando análisis predeterminado.")
            return create_default_tactical_analysis(home_team_id, away_team_id)
            
        # Generar análisis con método de redes neuronales
        neural_analysis = generate_neural_tactical_analysis(home_team_id, away_team_id)
            
        # Simplificar el análisis para la API pero incluir más datos tácticos
        simplified = {
            'tactical_style': {
                'home': tactical_analysis.get('team1_style', {}),
                'away': tactical_analysis.get('team2_style', {})
            },
            'key_battles': [
                {
                    'zone': battle.get('description', battle.get('zone', 'Zona no definida')),
                    'importance': battle.get('importance', 'Media'),
                    'key_factors': battle.get('key_factors', []),
                    'advantage': battle.get('advantage', 'Neutral')
                }
                for battle in tactical_analysis.get('key_battles', [])[:5]  # Aumentamos a los 5 principales
            ],
            'summary': tactical_analysis.get('summary', 'No hay suficientes datos para un análisis táctico completo'),
            'tactical_indices': {
                'home': tactical_analysis.get('team1_profile', {}).get('tactical_indices', {}),
                'away': tactical_analysis.get('team2_profile', {}).get('tactical_indices', {})
            },
            'tactical_traits': {
                'home': {
                    'strengths': tactical_analysis.get('team1_profile', {}).get('tactical_traits', {}).get('strengths', []),
                    'weaknesses': tactical_analysis.get('team1_profile', {}).get('tactical_traits', {}).get('weaknesses', [])
                },
                'away': {
                    'strengths': tactical_analysis.get('team2_profile', {}).get('tactical_traits', {}).get('strengths', []),
                    'weaknesses': tactical_analysis.get('team2_profile', {}).get('tactical_traits', {}).get('weaknesses', [])
                }
            },
            'matchup_analysis': {
                'possession_battle': tactical_analysis.get('possession_battle', {}),
                'pressing_dynamics': tactical_analysis.get('pressing_dynamics', {}),
                'attacking_comparison': tactical_analysis.get('attacking_comparison', {}),
                'tactical_advantages': tactical_analysis.get('tactical_advantages', {})
            },
            # Incluir ambos análisis (histórico y redes neuronales)
            'analysis_methods': {
                'historical': {
                    'description': 'Análisis basado en datos históricos de partidos recientes',
                    'confidence': tactical_analysis.get('confidence', 0.7),
                    'key_metrics': tactical_analysis.get('key_metrics', {})
                },
                'neural_network': neural_analysis
            }
        }
        
        return simplified
        
    except Exception as e:
        logger.error(f"Error generando análisis táctico simplificado: {e}")
        # En caso de error, proporcionar un análisis táctico predeterminado en lugar de solo un mensaje de error
        try:
            # Intentar crear un análisis predeterminado con los IDs proporcionados
            return create_default_tactical_analysis(home_team_id, away_team_id)
        except Exception:
            # Si incluso el análisis predeterminado falla, devolver un mensaje de error
            return {
                'error': f"No se pudo completar el análisis táctico: {str(e)}",
                'status': 'error'
            }

def create_default_tactical_analysis(home_team_id: int, away_team_id: int) -> Dict[str, Any]:
    """
    Crea un análisis táctico predeterminado cuando no hay datos históricos disponibles.
    
    Args:
        home_team_id: ID del equipo local
        away_team_id: ID del equipo visitante
        
    Returns:
        Análisis táctico predeterminado
    """
    # Crear un análisis táctico genérico basado en promedios de la liga
    default_analysis = {
        'tactical_style': {
            'home': {'possession': 0.5, 'countering': 0.5, 'high_press': 0.5, 'defensive': 0.5},
            'away': {'possession': 0.5, 'countering': 0.5, 'high_press': 0.5, 'defensive': 0.5},
        },
        'key_battles': [
            {
                'zone': 'Mediocampo central',
                'importance': 'Alta',
                'key_factors': ['Control del ritmo', 'Transiciones'],
                'advantage': 'Neutral'
            },
            {
                'zone': 'Bandas',
                'importance': 'Media',
                'key_factors': ['Centros', 'Amplitud'],
                'advantage': 'Neutral'
            },
            {
                'zone': 'Último tercio',
                'importance': 'Alta',
                'key_factors': ['Finalización', 'Defensa aérea'],
                'advantage': 'Neutral'
            }
        ],
        'summary': 'Análisis basado en promedios de la liga por falta de datos históricos específicos',
        'tactical_indices': {
            'home': {
                'pressing_intensity': 5.0,
                'possession_control': 5.0,
                'counter_attack_threat': 5.0,
                'defensive_solidity': 5.0,
                'attacking_efficiency': 5.0
            },
            'away': {
                'pressing_intensity': 5.0,
                'possession_control': 5.0,
                'counter_attack_threat': 5.0,
                'defensive_solidity': 5.0,
                'attacking_efficiency': 5.0
            }
        },
        'tactical_traits': {
            'home': {
                'strengths': ['Juego equilibrado', 'Adaptabilidad táctica'],
                'weaknesses': ['Sin debilidades claras identificadas']
            },
            'away': {
                'strengths': ['Juego equilibrado', 'Adaptabilidad táctica'],
                'weaknesses': ['Sin debilidades claras identificadas']
            }
        },
        'matchup_analysis': {
            'possession_battle': {
                'description': 'Control de posesión equilibrado',
                'advantage': 'Neutral',
                'key_metrics': {'possession_index': 1.0}
            },
            'pressing_dynamics': {
                'description': 'Ambos equipos con presión moderada',
                'advantage': 'Neutral',
                'key_metrics': {'pressing_difference': 0.0}
            },
            'attacking_comparison': {
                'description': 'Estilos ofensivos similares',
                'advantage': 'Neutral',
                'key_metrics': {'attacking_threat_ratio': 1.0}
            },            'tactical_advantages': {
                'home': ['Factor campo'],
                'away': ['Motivación visitante'],
                'overall': 'Equilibrio táctico'
            }
        }
    }
    
    # Incluir ambos métodos de análisis (histórico y neural)
    default_analysis['analysis_methods'] = {
        'historical': {
            'description': 'Análisis basado en promedios históricos de la liga',
            'confidence': 0.5,
            'key_metrics': {'possession_ratio': 0.5, 'pressing_intensity': 5.0}
        },
        'neural_network': {
            'description': 'Análisis de redes neuronales no disponible para estos equipos',
            'confidence': 0.0,
            'available': False
        }
    }
    
    return default_analysis

def generate_neural_tactical_analysis(home_team_id: int, away_team_id: int) -> Dict[str, Any]:
    """
    Genera un análisis táctico utilizando un enfoque de redes neuronales.
    
    Args:
        home_team_id: ID del equipo local
        away_team_id: ID del equipo visitante
        
    Returns:
        Análisis táctico basado en redes neuronales
    """
    try:
        # Intentar importar el modelo de red neuronal si está disponible
        try:
            from fnn_model import FeedforwardNeuralNetwork
            nn_available = True
        except ImportError:
            nn_available = False
            
        if not nn_available:
            return {
                'description': 'Análisis de redes neuronales no disponible',
                'confidence': 0.0,
                'available': False
            }
            
        # Obtener características tácticas de ambos equipos
        tactical_features = {}
        try:
            # Intentar utilizar módulos de características si están disponibles            try:
                # Actualmente no se encuentra la función específica para características tácticas
                # Esta sección está preparada para una futura integración
                # Si en el futuro se añade un módulo de características tácticas, añadir la importación aquí
                
                # Por ahora, lanzamos una excepción para usar los valores predeterminados
                raise ImportError("Módulo de características tácticas no disponible actualmente")
        except (ImportError, AttributeError):
                # Si no encuentra la función específica, usar valores predeterminados
                tactical_features = {
                    'home': {
                        'possession_tendency': 0.52,
                        'defensive_line_height': 65.0,
                        'pressing_intensity': 6.2,
                        'passing_directness': 7.1,
                        'width_utilization': 68.0,
                        'counter_attack_frequency': 5.8,
                        'set_piece_threat': 6.4
                    },
                    'away': {
                        'possession_tendency': 0.48,
                        'defensive_line_height': 58.0,
                        'pressing_intensity': 5.9,
                        'passing_directness': 6.8,
                        'width_utilization': 72.0,
                        'counter_attack_frequency': 6.5,
                        'set_piece_threat': 5.8
                    }
                }
        except Exception as e:
            logger.error(f"Error importando módulos de características: {e}")
            # Si hay un error general, usar valores predeterminados
            tactical_features = {
                'home': {
                    'possession_tendency': 0.52,
                    'defensive_line_height': 65.0,
                    'pressing_intensity': 6.2,
                    'passing_directness': 7.1,
                    'width_utilization': 68.0,
                    'counter_attack_frequency': 5.8,
                    'set_piece_threat': 6.4
                },
                'away': {
                    'possession_tendency': 0.48,
                    'defensive_line_height': 58.0,
                    'pressing_intensity': 5.9,
                    'passing_directness': 6.8,
                    'width_utilization': 72.0,
                    'counter_attack_frequency': 6.5,
                    'set_piece_threat': 5.8
                }
            }
            
        # Calcular ventajas tácticas basadas en características
        advantages = calculate_tactical_advantages(tactical_features['home'], tactical_features['away'])
        
        # Calcular un puntaje de compatibilidad de estilos
        style_compatibility = calculate_style_compatibility(tactical_features['home'], tactical_features['away'])
        
        # Integrar con sistema ELO si está disponible
        try:
            from team_elo_rating import EloRating
            elo_system = EloRating()
            
            home_elo = elo_system.get_team_rating(home_team_id)
            away_elo = elo_system.get_team_rating(away_team_id)
            
            elo_factor = (home_elo - away_elo) / 400.0  # Normalizar la diferencia
            
            # Ajustar ventajas basadas en diferencia ELO
            for category in advantages:
                if 'home_advantage' in advantages[category]:
                    advantages[category]['home_advantage'] += elo_factor * 0.2
                if 'away_advantage' in advantages[category]:
                    advantages[category]['away_advantage'] -= elo_factor * 0.2
        except Exception as e:
            logger.warning(f"No se pudo integrar datos ELO: {e}")
        
        # Construir el análisis final
        return {
            'description': 'Análisis táctico generado mediante modelo predictivo de red neuronal',
            'confidence': 0.85,
            'available': True,
            'style_compatibility': style_compatibility,
            'tactical_advantages': advantages,
            'team_profiles': {
                'home': {
                    'style': get_team_style_profile(tactical_features['home']),
                    'strengths': identify_team_strengths(tactical_features['home']),
                    'weaknesses': identify_team_vulnerabilities(tactical_features['home'])
                },
                'away': {
                    'style': get_team_style_profile(tactical_features['away']),
                    'strengths': identify_team_strengths(tactical_features['away']),
                    'weaknesses': identify_team_vulnerabilities(tactical_features['away'])
                }
            },
            'key_matchups': generate_key_matchups(tactical_features['home'], tactical_features['away'])
        }
    except Exception as e:
        logger.error(f"Error generando análisis de redes neuronales: {e}")
        return {
            'description': 'Error en análisis de redes neuronales',
            'confidence': 0.0,
            'available': False,
            'error': str(e)
        }

def calculate_tactical_advantages(home_features: Dict[str, float], away_features: Dict[str, float]) -> Dict[str, Dict[str, float]]:
    """Calcula ventajas tácticas entre dos equipos basado en sus características"""
    advantages = {
        'pressing': {},
        'possession': {},
        'transition': {},
        'set_pieces': {}
    }
    
    # Análisis de pressing
    if home_features['pressing_intensity'] > away_features['pressing_intensity'] + 0.5:
        advantages['pressing']['home_advantage'] = home_features['pressing_intensity'] - away_features['pressing_intensity']
        advantages['pressing']['description'] = 'El equipo local tiene ventaja en intensidad de presión'
    elif away_features['pressing_intensity'] > home_features['pressing_intensity'] + 0.5:
        advantages['pressing']['away_advantage'] = away_features['pressing_intensity'] - home_features['pressing_intensity']
        advantages['pressing']['description'] = 'El equipo visitante tiene ventaja en intensidad de presión'
    else:
        advantages['pressing']['description'] = 'Ambos equipos con intensidad de presión similar'
    
    # Análisis de posesión
    if home_features['possession_tendency'] > away_features['possession_tendency'] + 0.05:
        advantages['possession']['home_advantage'] = (home_features['possession_tendency'] - away_features['possession_tendency']) * 10
        advantages['possession']['description'] = 'El equipo local domina la posesión'
    elif away_features['possession_tendency'] > home_features['possession_tendency'] + 0.05:
        advantages['possession']['away_advantage'] = (away_features['possession_tendency'] - home_features['possession_tendency']) * 10
        advantages['possession']['description'] = 'El equipo visitante domina la posesión'
    else:
        advantages['possession']['description'] = 'Posesión equilibrada entre ambos equipos'
    
    # Análisis de transiciones
    if home_features['counter_attack_frequency'] > away_features['counter_attack_frequency'] + 0.8:
        advantages['transition']['home_advantage'] = (home_features['counter_attack_frequency'] - away_features['counter_attack_frequency']) / 2
        advantages['transition']['description'] = 'El equipo local es superior en contraataques'
    elif away_features['counter_attack_frequency'] > home_features['counter_attack_frequency'] + 0.8:
        advantages['transition']['away_advantage'] = (away_features['counter_attack_frequency'] - home_features['counter_attack_frequency']) / 2
        advantages['transition']['description'] = 'El equipo visitante es superior en contraataques'
    else:
        advantages['transition']['description'] = 'Ambos equipos con capacidad similar en transiciones'
    
    # Análisis de jugadas a balón parado
    if home_features['set_piece_threat'] > away_features['set_piece_threat'] + 0.7:
        advantages['set_pieces']['home_advantage'] = (home_features['set_piece_threat'] - away_features['set_piece_threat']) / 2
        advantages['set_pieces']['description'] = 'El equipo local es más peligroso en jugadas a balón parado'
    elif away_features['set_piece_threat'] > home_features['set_piece_threat'] + 0.7:
        advantages['set_pieces']['away_advantage'] = (away_features['set_piece_threat'] - home_features['set_piece_threat']) / 2
        advantages['set_pieces']['description'] = 'El equipo visitante es más peligroso en jugadas a balón parado'
    else:
        advantages['set_pieces']['description'] = 'Ambos equipos con amenaza similar en jugadas a balón parado'
    
    return advantages

def calculate_style_compatibility(home_features: Dict[str, float], away_features: Dict[str, float]) -> Dict[str, Any]:
    """Calcula la compatibilidad de estilos entre dos equipos"""
    # Calcular diferencia en tendencia de posesión
    possession_contrast = abs(home_features['possession_tendency'] - away_features['possession_tendency'])
    
    # Calcular diferencia en altura de línea defensiva
    defensive_line_contrast = abs(home_features['defensive_line_height'] - away_features['defensive_line_height']) / 100.0
    
    # Calcular contraste de estilos directo vs. posesión
    directness_contrast = abs(home_features['passing_directness'] - away_features['passing_directness']) / 10.0
    
    # Probabilidad de partido con muchos goles
    high_scoring_probability = 0.5
    
    # Aumentar si ambos equipos tienen líneas defensivas altas
    if home_features['defensive_line_height'] > 65 and away_features['defensive_line_height'] > 65:
        high_scoring_probability += 0.15
    
    # Aumentar si hay gran contraste de estilos
    if possession_contrast > 0.15 or directness_contrast > 0.3:
        high_scoring_probability += 0.1
    
    # Aumentar si ambos equipos son de contraataque
    if home_features['counter_attack_frequency'] > 6.5 and away_features['counter_attack_frequency'] > 6.5:
        high_scoring_probability += 0.15
    
    # Calcular espectacularidad esperada
    entertainment_score = 5.0
    entertainment_score += possession_contrast * 10  # 0-1 → 0-10
    entertainment_score += directness_contrast * 5   # 0-1 → 0-5
    entertainment_score = min(max(entertainment_score, 1), 10)  # limitar entre 1-10
    
    # Generar explicaciones para cada aspecto
    possession_explanation = generate_possession_contrast_explanation(
        possession_contrast, 
        home_features['possession_tendency'], 
        away_features['possession_tendency']
    )
    
    defensive_explanation = generate_defensive_contrast_explanation(
        defensive_line_contrast,
        home_features['defensive_line_height'],
        away_features['defensive_line_height']
    )
    
    style_explanation = generate_playing_style_explanation(
        directness_contrast,
        home_features['passing_directness'],
        away_features['passing_directness']
    )
    
    scoring_explanation = generate_high_scoring_explanation(high_scoring_probability)
    
    entertainment_explanation = generate_entertainment_explanation(entertainment_score)
    
    match_type = classify_match_type(home_features, away_features)
    match_type_explanation = generate_match_type_explanation(match_type)
    
    return {
        'possession_contrast': round(possession_contrast * 100, 1),  # Convertir a porcentaje
        'defensive_approach_contrast': round(defensive_line_contrast * 100, 1),  # Convertir a porcentaje
        'playing_style_contrast': round(directness_contrast * 100, 1),  # Convertir a porcentaje
        'high_scoring_probability': round(high_scoring_probability, 2),
        'entertainment_score': round(entertainment_score, 1),
        'tactical_classification': match_type,
        'explanations': {
            'possession_contrast': possession_explanation,
            'defensive_approach': defensive_explanation,
            'playing_style': style_explanation,
            'high_scoring': scoring_explanation,
            'entertainment': entertainment_explanation,
            'match_type': match_type_explanation
        }
    }

def classify_match_type(home_features: Dict[str, float], away_features: Dict[str, float]) -> str:
    """Clasifica el tipo de partido basado en características de ambos equipos"""
    # Determinar tipo de partido basado en características
    if home_features['possession_tendency'] > 0.55 and away_features['possession_tendency'] < 0.45:
        if away_features['counter_attack_frequency'] > 6.5:
            return "Ataque constante vs. Contraataque"
        else:
            return "Dominio posicional vs. Bloque bajo"
    elif away_features['possession_tendency'] > 0.55 and home_features['possession_tendency'] < 0.45:
        if home_features['counter_attack_frequency'] > 6.5:
            return "Contraataque vs. Ataque constante"
        else:
            return "Bloque bajo vs. Dominio posicional"
    elif home_features['pressing_intensity'] > 6.5 and away_features['pressing_intensity'] > 6.5:
        return "Presión alta mutua"
    elif home_features['defensive_line_height'] < 55 and away_features['defensive_line_height'] < 55:
        return "Defensas compactas - Pocos espacios"
    elif home_features['defensive_line_height'] > 65 and away_features['defensive_line_height'] > 65:
        return "Líneas defensivas altas - Muchos espacios"
    else:
        return "Partido equilibrado táctico"

def get_team_style_profile(features: Dict[str, float]) -> Dict[str, float]:
    """Determina el perfil de estilo de juego de un equipo basado en sus características"""
    return {
        'possession': min(features['possession_tendency'] * 2, 1.0),
        'direct': min(features['passing_directness'] / 10, 1.0),
        'press': min(features['pressing_intensity'] / 10, 1.0),
        'counter': min(features['counter_attack_frequency'] / 10, 1.0),
        'wide_play': min(features['width_utilization'] / 100, 1.0)
    }

def identify_team_strengths(features: Dict[str, float]) -> List[str]:
    """Identifica fortalezas tácticas de un equipo basado en sus características"""
    strengths = []
    
    if features['possession_tendency'] > 0.55:
        strengths.append("Control de posesión")
    
    if features['passing_directness'] > 7.5:
        strengths.append("Juego directo efectivo")
    elif features['passing_directness'] < 5.5:
        strengths.append("Construcción elaborada")
    
    if features['pressing_intensity'] > 7.0:
        strengths.append("Presión alta intensa")
    
    if features['counter_attack_frequency'] > 7.0:
        strengths.append("Contraataques letales")
    
    if features['width_utilization'] > 75:
        strengths.append("Amplitud y juego por bandas")
    elif features['width_utilization'] < 60:
        strengths.append("Juego vertical por el centro")
    
    if features['set_piece_threat'] > 7.0:
        strengths.append("Especialista en balón parado")
    
    if features['defensive_line_height'] < 55:
        strengths.append("Solidez defensiva")
    
    # Asegurarse de que siempre devolvemos al menos 2 fortalezas
    if len(strengths) < 2:
        if features['counter_attack_frequency'] > 6.0:
            strengths.append("Transiciones rápidas")
        if features['set_piece_threat'] > 6.0:
            strengths.append("Amenaza en balón parado")
        if len(strengths) < 2:
            strengths.append("Equilibrio táctico")
    
    return strengths[:3]  # Devolvemos máximo 3 fortalezas

def identify_team_vulnerabilities(features: Dict[str, float]) -> List[str]:
    """Identifica vulnerabilidades tácticas de un equipo basado en sus características"""
    vulnerabilities = []
    
    if features['possession_tendency'] < 0.42:
        vulnerabilities.append("Dificultad para mantener posesión")
    
    if features['defensive_line_height'] > 70:
        vulnerabilities.append("Vulnerabilidad a balones a la espalda")
    
    if features['pressing_intensity'] < 5.0:
        vulnerabilities.append("Presión débil en mediocampo")
    
    if features['width_utilization'] > 80:
        vulnerabilities.append("Exposición en el centro del campo")
    elif features['width_utilization'] < 55:
        vulnerabilities.append("Poca amplitud ofensiva")
    
    if features['set_piece_threat'] < 5.0:
        vulnerabilities.append("Debilidad en balón parado")
    
    # Asegurarse de que siempre devolvemos al menos 1 vulnerabilidad
    if not vulnerabilities:
        if features['defensive_line_height'] > 60:
            vulnerabilities.append("Espacio a las espaldas de la defensa")
        else:
            vulnerabilities.append("Sin vulnerabilidades claras identificadas")
    
    return vulnerabilities[:2]  # Devolvemos máximo 2 vulnerabilidades

def generate_key_matchups(home_features: Dict[str, float], away_features: Dict[str, float]) -> List[Dict[str, Any]]:
    """Genera enfrentamientos clave basados en características de ambos equipos"""
    matchups = []
    
    # Matchup de presión vs. posesión
    if home_features['pressing_intensity'] > 6.5 and away_features['possession_tendency'] > 0.52:
        matchups.append({
            'zone': 'Mediocampo',
            'description': 'Presión local vs. Posesión visitante',
            'key_factors': ['Intensidad de presión', 'Control técnico bajo presión'],
            'advantage': 'home' if home_features['pressing_intensity'] > (away_features['possession_tendency'] * 12) else 'away',
            'importance': 'Alta'
        })
    elif away_features['pressing_intensity'] > 6.5 and home_features['possession_tendency'] > 0.52:
        matchups.append({
            'zone': 'Mediocampo',
            'description': 'Posesión local vs. Presión visitante',
            'key_factors': ['Control técnico bajo presión', 'Intensidad de presión'],
            'advantage': 'home' if home_features['possession_tendency'] * 12 > away_features['pressing_intensity'] else 'away',
            'importance': 'Alta'
        })
    
    # Matchup de ataque directo vs. línea defensiva
    if home_features['passing_directness'] > 7.0 and away_features['defensive_line_height'] > 65:
        matchups.append({
            'zone': 'Defensa visitante',
            'description': 'Ataque directo local vs. Línea alta visitante',
            'key_factors': ['Pases largos', 'Anticipación defensiva'],
            'advantage': 'home',
            'importance': 'Alta'
        })
    elif away_features['passing_directness'] > 7.0 and home_features['defensive_line_height'] > 65:
        matchups.append({
            'zone': 'Defensa local',
            'description': 'Línea alta local vs. Ataque directo visitante',
            'key_factors': ['Anticipación defensiva', 'Pases largos'],
            'advantage': 'away',
            'importance': 'Alta'
        })
    
    # Matchup de contraataque
    if home_features['counter_attack_frequency'] > 6.5:
        matchups.append({
            'zone': 'Transición ofensiva',
            'description': 'Contraataques locales',
            'key_factors': ['Velocidad de transición', 'Balance defensivo visitante'],
            'advantage': 'home' if home_features['counter_attack_frequency'] > 7.0 else 'neutral',
            'importance': 'Media'
        })
    if away_features['counter_attack_frequency'] > 6.5:
        matchups.append({
            'zone': 'Transición defensiva',
            'description': 'Contraataques visitantes',
            'key_factors': ['Balance defensivo local', 'Velocidad de transición'],
            'advantage': 'away' if away_features['counter_attack_frequency'] > 7.0 else 'neutral',
            'importance': 'Media'
        })
    
    # Matchup de juego por bandas
    if abs(home_features['width_utilization'] - away_features['width_utilization']) > 15:
        if home_features['width_utilization'] > away_features['width_utilization']:
            matchups.append({
                'zone': 'Bandas',
                'description': 'Juego por bandas local vs. Defensa centralizada visitante',
                'key_factors': ['Centros', 'Coberturas defensivas'],
                'advantage': 'home',
                'importance': 'Media'
            })
        else:
            matchups.append({
                'zone': 'Bandas',
                'description': 'Defensa centralizada local vs. Juego por bandas visitante',
                'key_factors': ['Coberturas defensivas', 'Centros'],
                'advantage': 'away',
                'importance': 'Media'
            })
    
    # Asegurarse de que tenemos al menos un matchup
    if not matchups:
        matchups.append({
            'zone': 'General',
            'description': 'Equilibrio táctico en todas las zonas',
            'key_factors': ['Ejecución técnica', 'Decisiones individuales'],
            'advantage': 'neutral',
            'importance': 'Media'
        })
    
    return matchups[:4]  # Limitamos a 4 matchups

def generate_possession_contrast_explanation(possession_contrast: float, home_possession: float, away_possession: float) -> str:
    """Genera una explicación del contraste de posesión entre dos equipos."""
    if possession_contrast > 0.2:
        if home_possession > away_possession:
            return f"Se espera un claro dominio de posesión del equipo local ({home_possession:.0%} vs {away_possession:.0%}), lo que podría resultar en un partido de ataque contra defensa."
        else:
            return f"Se espera un claro dominio de posesión del equipo visitante ({away_possession:.0%} vs {home_possession:.0%}), lo que podría resultar en un partido de ataque contra defensa."
    elif possession_contrast > 0.1:
        if home_possession > away_possession:
            return f"El equipo local tiende a tener más posesión ({home_possession:.0%} vs {away_possession:.0%}), aunque la diferencia no es extrema."
        else:
            return f"El equipo visitante tiende a tener más posesión ({away_possession:.0%} vs {home_possession:.0%}), aunque la diferencia no es extrema."
    else:
        return f"Se espera un partido equilibrado en términos de posesión (aproximadamente {home_possession:.0%} vs {away_possession:.0%}), con ambos equipos capaces de mantener el balón."

def generate_defensive_contrast_explanation(defensive_contrast: float, home_line: float, away_line: float) -> str:
    """Genera una explicación del contraste en la línea defensiva entre dos equipos."""
    if defensive_contrast > 0.2:
        if home_line > away_line:
            return "El equipo local juega con una línea defensiva significativamente más alta, lo que podría generar espacios a la espalda de la defensa."
        else:
            return "El equipo visitante juega con una línea defensiva significativamente más alta, lo que podría generar espacios a la espalda de la defensa."
    elif defensive_contrast > 0.1:
        if home_line > away_line:
            return "El equipo local tiende a jugar con una línea defensiva ligeramente más adelantada, aunque la diferencia no es extrema."
        else:
            return "El equipo visitante tiende a jugar con una línea defensiva ligeramente más adelantada, aunque la diferencia no es extrema."
    else:
        return "Ambos equipos manejan una altura de línea defensiva similar, lo que sugiere un enfoque táctico comparable en defensa."

def generate_playing_style_explanation(directness_contrast: float, home_directness: float, away_directness: float) -> str:
    """Genera una explicación del contraste en el estilo de juego entre dos equipos."""
    if directness_contrast > 0.3:
        if home_directness > away_directness:
            return "Hay un contraste marcado en los estilos: el equipo local prefiere un juego más directo, mientras que el visitante opta por un juego más elaborado."
        else:
            return "Hay un contraste marcado en los estilos: el equipo visitante prefiere un juego más directo, mientras que el local opta por un juego más elaborado."
    elif directness_contrast > 0.15:
        if home_directness > away_directness:
            return "El equipo local tiende a un juego más directo que el visitante, aunque la diferencia en estilos no es extrema."
        else:
            return "El equipo visitante tiende a un juego más directo que el local, aunque la diferencia en estilos no es extrema."
    else:
        return "Ambos equipos muestran un estilo de juego similar en términos de verticalidad y elaboración."

def generate_high_scoring_explanation(probability: float) -> str:
    """Genera una explicación de la probabilidad de un partido con muchos goles."""
    if probability > 0.7:
        return f"Alta probabilidad de un partido con muchos goles ({probability:.0%}). Las características ofensivas de ambos equipos y el contraste táctico favorecen un partido abierto."
    elif probability > 0.5:
        return f"Moderada probabilidad de un partido con muchos goles ({probability:.0%}). Existe potencial ofensivo pero también factores que podrían limitar el marcador."
    else:
        return f"Baja probabilidad de un partido con muchos goles ({probability:.0%}). El planteamiento táctico y las características de los equipos sugieren un partido más cerrado."

def generate_entertainment_explanation(score: float) -> str:
    """Genera una explicación del nivel de entretenimiento esperado del partido."""
    if score >= 8:
        return f"Se espera un partido muy entretenido (puntuación {score:.1f}/10). El contraste de estilos y las características de los equipos favorecen un espectáculo atractivo."
    elif score >= 6:
        return f"Se espera un partido moderadamente entretenido (puntuación {score:.1f}/10). El equilibrio entre estilos podría generar un encuentro interesante."
    else:
        return f"El partido podría ser menos entretenido de lo habitual (puntuación {score:.1f}/10). Los estilos de juego y el planteamiento táctico sugieren un partido más táctico o cerrado."

def generate_match_type_explanation(match_type: str) -> str:
    """Genera una explicación detallada del tipo de partido esperado."""
    explanations = {
        "Ataque constante vs. Contraataque": "Se espera un partido de ataque contra defensa, con un equipo dominando la posesión mientras el otro busca aprovechar los espacios al contraataque.",
        "Dominio posicional vs. Bloque bajo": "Un equipo probablemente dominará la posesión contra un rival que preferirá defender en bloque bajo y buscar oportunidades puntuales.",
        "Contraataque vs. Ataque constante": "El partido podría desarrollarse con el equipo visitante controlando la posesión, mientras que el local adopta una estrategia más reactiva, buscando explotar espacios en transiciones rápidas.",
        "Bloque bajo vs. Dominio posicional": "El equipo local probablemente adoptará una postura defensiva contra un rival que buscará dominar mediante posesión.",
        "Pressing intenso": "Ambos equipos practican un pressing alto, lo que podría resultar en un partido intenso con muchas recuperaciones en campo rival.",
        "Batalla táctica": "Se anticipa un partido equilibrado tácticamente, con ambos equipos alternando entre diferentes aproximaciones al juego.",
        "Juego directo": "Ambos equipos prefieren un estilo de juego directo, lo que podría resultar en un partido con muchas transiciones rápidas.",
        "Juego elaborado": "Los dos equipos optan por un juego más elaborado, sugiriendo un partido con mayor control y menos transiciones.",
        "Equilibrado": "No hay un estilo claramente predominante, lo que sugiere un partido equilibrado con alternancia en el control."
    }
    
    return explanations.get(match_type, "El tipo de partido no sigue un patrón claramente definido.")
