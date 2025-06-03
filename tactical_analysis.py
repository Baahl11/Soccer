import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TacticalAnalyzer:
    """
    Analiza el estilo de juego y patrones tácticos de los equipos
    """
    def __init__(self):
        pass  # No need for feature_engineering dependency
    
    def analyze_team_style(self, team_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analiza el estilo de juego completo de un equipo
        """
        try:
            # Use internal methods instead of external feature engineering
            possession_features = self._extract_possession_features(team_matches)
            pressing_features = self._extract_pressing_features(team_matches)
            attacking_features = self._extract_attacking_features(team_matches)
            defensive_features = self._extract_defensive_features(team_matches)
            
            # Map internal features to expected pattern structure
            possession_patterns = {
                'avg_possession': possession_features.get('possession_control', 0.5),
                'final_third_possession': possession_features.get('territorial_dominance', 0.5),
                'progression_ratio': possession_features.get('progression_efficiency', 0.5),
                'possession_recovery_rate': possession_features.get('pressure_resistance', 0.5),
                'buildup_success_rate': possession_features.get('possession_control', 0.5),
                'pressure_resistance_rate': possession_features.get('pressure_resistance', 0.5)
            }
            
            pressing_stats = {
                'pressing_intensity': pressing_features.get('pressing_intensity', 0.5),
                'pressing_success_rate': pressing_features.get('pressing_success', 0.5),
                'press_resistance': pressing_features.get('pressure_resistance', 0.5),
                'defensive_aggression': pressing_features.get('defensive_aggression', 0.5),
                'pressing_coordination': pressing_features.get('defensive_aggression', 0.5),
                'press_efficiency': pressing_features.get('pressing_success', 0.5),
                'avg_press_distance': pressing_features.get('pressing_line_height', 0.5) * 50  # Convert to meters
            }
            
            attacking_patterns = {
                'direct_play_ratio': attacking_features.get('attack_directness', 0.5),
                'wing_play_ratio': attacking_features.get('wing_play', 0.5),
                'width_variance': attacking_features.get('creative_play', 0.5),
                'transition_speed': attacking_features.get('counter_attack_speed', 0.5)
            }
            
            defensive_patterns = {
                'defensive_line_height': defensive_features.get('line_height', 0.5),
                'defensive_action_rate': defensive_features.get('defensive_compactness', 0.5),
                'high_press_tendency': defensive_features.get('line_height', 0.5),
                'interception_rate': defensive_features.get('tackle_success_rate', 0.5),
                'clearance_tendency': defensive_features.get('aerial_dominance', 0.5)
            }
            
            # Combinar todos los análisis
            style_profile: Dict[str, Any] = {
                'possession_style': {
                    'control_tendency': possession_patterns['avg_possession'],
                    'territorial_dominance': possession_patterns['final_third_possession'],
                    'progression_efficiency': possession_patterns['progression_ratio'],
                    'ball_recovery': possession_patterns['possession_recovery_rate'],
                    'buildup_success': possession_patterns['buildup_success_rate'],
                    'pressure_resistance': possession_patterns['pressure_resistance_rate']
                },
                'pressing_style': {
                    'intensity': pressing_stats['pressing_intensity'],
                    'effectiveness': pressing_stats['pressing_success_rate'],
                    'press_resistance': pressing_stats['press_resistance'],
                    'defensive_aggression': pressing_stats['defensive_aggression'],
                    'coordination': pressing_stats['pressing_coordination'],
                    'efficiency': pressing_stats['press_efficiency'],
                    'avg_distance': pressing_stats['avg_press_distance']
                },
                'attacking_style': {
                    'directness': attacking_patterns['direct_play_ratio'],
                    'wing_play': attacking_patterns['wing_play_ratio'],
                    'positional_fluidity': attacking_patterns['width_variance'],
                    'transition_speed': attacking_patterns['transition_speed']
                },
                'defensive_style': {
                    'line_height': defensive_patterns['defensive_line_height'],
                    'defensive_activity': defensive_patterns['defensive_action_rate'],
                    'pressing_tendency': defensive_patterns['high_press_tendency'],
                    'interception_focus': defensive_patterns['interception_rate'],
                    'clearance_preference': defensive_patterns['clearance_tendency']
                }
            }
            
            # Identificar fortalezas y debilidades principales
            strengths, weaknesses = self._identify_tactical_traits(style_profile)
            style_profile['tactical_traits'] = {
                'strengths': strengths,
                'weaknesses': weaknesses
            }
            
            # Calcular índices tácticos compuestos
            style_profile['tactical_indices'] = self._calculate_tactical_indices(style_profile)
            
            return style_profile
            
        except Exception as e:
            logger.error(f"Error en el análisis táctico: {e}")
            return self._get_default_style_profile()

    def get_tactical_matchup(self, team1_matches: List[Dict[str, Any]], 
                           team2_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analiza el enfrentamiento táctico entre dos equipos
        """
        try:
            # Obtener perfiles tácticos
            team1_style = self.analyze_team_style(team1_matches)
            team2_style = self.analyze_team_style(team2_matches)
            
            # Analizar matchups específicos
            possession_battle = self._analyze_possession_matchup(
                team1_style['possession_style'],
                team2_style['possession_style']
            )
            
            pressing_dynamics = self._analyze_pressing_matchup(
                team1_style['pressing_style'],
                team2_style['pressing_style']
            )
            
            attacking_comp = self._analyze_attacking_matchup(
                team1_style['attacking_style'],
                team2_style['defensive_style']
            )
            
            # Identificar batallas tácticas clave
            key_battles = self._identify_key_tactical_battles(team1_style, team2_style)
            
            # Calcular ventajas tácticas
            tactical_advantages = self._calculate_tactical_advantages(
                team1_style, team2_style,
                {'possession_battle': possession_battle, 'pressing_dynamics': pressing_dynamics}
            )
            
            return {
                'possession_battle': possession_battle,
                'pressing_dynamics': pressing_dynamics,
                'attacking_comparison': attacking_comp,
                'key_battles': key_battles,
                'tactical_advantages': tactical_advantages,
                'team1_profile': team1_style,
                'team2_profile': team2_style
            }
            
        except Exception as e:
            logger.error(f"Error en análisis de matchup: {e}")
            return self._get_default_matchup_analysis()
            
    def _analyze_possession_matchup(self, team1_possession: Dict[str, float],
                                  team2_possession: Dict[str, float]) -> Dict[str, float]:
        """
        Analiza el enfrentamiento en términos de posesión
        """
        return {
            'expected_possession_ratio': team1_possession['control_tendency'] / 
                                      (team1_possession['control_tendency'] + team2_possession['control_tendency']),
            'territorial_advantage': team1_possession['territorial_dominance'] - team2_possession['territorial_dominance'],
            'progression_comparison': team1_possession['progression_efficiency'] / 
                                   max(team2_possession['progression_efficiency'], 0.1),
            'buildup_advantage': team1_possession['buildup_success'] - team2_possession['pressure_resistance'],
            'pressure_resistance_diff': team1_possession['pressure_resistance'] - team2_possession['pressure_resistance']
        }
        
    def _analyze_pressing_matchup(self, team1_pressing: Dict[str, float],
                                team2_pressing: Dict[str, float]) -> Dict[str, float]:
        """
        Analiza el enfrentamiento en términos de pressing
        """
        return {
            'pressing_battle': team1_pressing['intensity'] - team2_pressing['intensity'],
            'press_resistance_advantage': team1_pressing['press_resistance'] - team2_pressing['press_resistance'],
            'defensive_aggression_comparison': team1_pressing['defensive_aggression'] / 
                                            max(team2_pressing['defensive_aggression'], 0.1),
            'coordination_advantage': team1_pressing['coordination'] - team2_pressing['coordination'],
            'pressing_efficiency_ratio': team1_pressing['efficiency'] / max(team2_pressing['efficiency'], 0.1)
        }
        
    def _analyze_attacking_matchup(self, team1_attack: Dict[str, float],
                               team2_defense: Dict[str, float]) -> Dict[str, float]:
        """
        Analiza el enfrentamiento entre el ataque de un equipo y la defensa del otro
        
        Args:
            team1_attack: Datos de ataque del equipo 1
            team2_defense: Datos de defensa del equipo 2
            
        Returns:
            Análisis del matchup ofensivo-defensivo
        """        
        return {
            'attacking_advantage': team1_attack['directness'] - team2_defense['interception_focus'],
            'transition_effectiveness': team1_attack['transition_speed'] - team2_defense['defensive_activity'],
            'wing_play_effectiveness': team1_attack['wing_play'] - team2_defense['pressing_tendency'],
            'fluidity_vs_structure': team1_attack['positional_fluidity'] - (1 - team2_defense['line_height'])
        }
    
    def _identify_key_tactical_battles(self, team1_style: Dict[str, Any], 
                                     team2_style: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Identifica las batallas tácticas clave entre dos equipos
        
        Args:
            team1_style: Perfil de estilo táctico del equipo 1
            team2_style: Perfil de estilo táctico del equipo 2
            
        Returns:
            Lista de batallas tácticas clave
        """
        key_battles = []
        
        # Posesión vs pressing
        if team1_style['possession_style']['control_tendency'] > 0.6 and team2_style['pressing_style']['intensity'] > 0.6:
            key_battles.append({
                'description': "Posesión vs Pressing intenso",
                'team1_approach': "Mantener posesión bajo presión",
                'team2_approach': "Pressing agresivo para recuperar balón",
                'importance': 0.8
            })
            
        # Ataques rápidos vs defensa alta
        if team1_style['attacking_style']['transition_speed'] > 0.65 and team2_style['defensive_style']['line_height'] > 0.7:
            key_battles.append({
                'description': "Contraataques vs Línea alta",
                'team1_approach': "Transiciones rápidas al ataque",
                'team2_approach': "Defensa adelantada con trampa del fuera de juego",
                'importance': 0.85
            })
            
        # Juego por bandas vs defensa central
        if team1_style['attacking_style']['wing_play'] > 0.7 and team2_style['defensive_style']['clearance_preference'] > 0.7:
            key_battles.append({
                'description': "Juego por bandas vs Defensa centralizada",
                'team1_approach': "Amplitud y centros",
                'team2_approach': "Bloque central compacto",
                'importance': 0.75
            })
            
        # Añadir al menos una batalla por defecto si no se identificó ninguna
        if not key_battles:
            key_battles.append({
                'description': "Equilibrio táctico general",
                'team1_approach': "Estilo equilibrado",
                'team2_approach': "Estilo equilibrado",
                'importance': 0.5
            })
            
        return key_battles
        
    def _calculate_tactical_advantages(self, team1_style: Dict[str, Any], 
                                     team2_style: Dict[str, Any],
                                     matchup_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula ventajas tácticas entre equipos por zonas y fases de juego
        
        Args:
            team1_style: Perfil de estilo táctico del equipo 1
            team2_style: Perfil de estilo táctico del equipo 2
            matchup_analysis: Análisis de enfrentamientos específicos
            
        Returns:
            Ventajas tácticas por zonas y fases
        """
        advantages = {
            'zones': {
                'defensive_third': 0.0,
                'middle_third': 0.0,
                'final_third': 0.0,
                'wings': 0.0,
                'central_areas': 0.0
            },
            'phases': {
                'buildup': 0.0,
                'established_attack': 0.0,
                'transitions': 0.0,
                'defensive_organization': 0.0,
                'defensive_transitions': 0.0,
                'set_pieces': 0.0
            }
        }
        
        # Calcular ventajas zonales
        advantages['zones']['defensive_third'] = (
            team1_style['defensive_style']['defensive_activity'] - 
            team2_style['attacking_style']['directness']
        ) * 0.5
        
        advantages['zones']['middle_third'] = (
            team1_style['possession_style']['progression_efficiency'] - 
            team2_style['pressing_style']['effectiveness']
        ) * 0.5
        
        advantages['zones']['final_third'] = (
            team1_style['attacking_style']['directness'] - 
            team2_style['defensive_style']['defensive_activity']
        ) * 0.5
        
        advantages['zones']['wings'] = (
            team1_style['attacking_style']['wing_play'] - 
            team2_style['defensive_style']['pressing_tendency']
        ) * 0.5
        
        advantages['zones']['central_areas'] = (
            team1_style['possession_style']['buildup_success'] - 
            team2_style['pressing_style']['coordination']
        ) * 0.5
        
        # Calcular ventajas por fases
        advantages['phases']['buildup'] = (
            team1_style['possession_style']['buildup_success'] - 
            team2_style['pressing_style']['effectiveness']
        ) * 0.5
        
        advantages['phases']['established_attack'] = (
            team1_style['possession_style']['territorial_dominance'] - 
            team2_style['defensive_style']['defensive_activity']
        ) * 0.5
        
        advantages['phases']['transitions'] = (
            team1_style['attacking_style']['transition_speed'] - 
            team2_style['defensive_style']['defensive_activity']
        ) * 0.5
        
        advantages['phases']['defensive_organization'] = (
            team1_style['defensive_style']['defensive_activity'] - 
            team2_style['possession_style']['territorial_dominance']
        ) * 0.5
        
        advantages['phases']['defensive_transitions'] = (
            team1_style['pressing_style']['intensity'] - 
            team2_style['attacking_style']['transition_speed']
        ) * 0.5
        
        return advantages
        
    def _get_default_matchup_analysis(self) -> Dict[str, Any]:
        """
        Retorna un análisis de matchup por defecto
        
        Returns:
            Análisis de matchup táctico por defecto
        """
        return {
            'possession_battle': {
                'expected_possession_ratio': 0.5,
                'territorial_advantage': 0.0,
                'buildup_quality_diff': 0.0,
                'pressure_resistance_diff': 0.0
            },
            'pressing_dynamics': {
                'pressing_battle': 0.0,
                'press_resistance_advantage': 0.0,
                'defensive_aggression_comparison': 1.0
            },            'attacking_comparison': {
                'attacking_advantage': 0.0,
                'transition_effectiveness': 0.0,
                'wing_play_effectiveness': 0.0,
                'fluidity_vs_structure': 0.0
            },
            'key_battles': [
                {
                    'description': "Equilibrio táctico general",
                    'team1_approach': "Estilo equilibrado",
                    'team2_approach': "Estilo equilibrado",
                    'importance': 0.5
                }
            ],
            'tactical_advantages': {
                'zones': {
                    'defensive_third': 0.0,
                    'middle_third': 0.0,
                    'final_third': 0.0,
                    'wings': 0.0,
                    'central_areas': 0.0
                },
                'phases': {
                    'buildup': 0.0,
                    'established_attack': 0.0,
                    'transitions': 0.0,
                    'defensive_organization': 0.0,
                    'defensive_transitions': 0.0,
                    'set_pieces': 0.0
                }
            }
        }
        
    def _get_default_style_profile(self) -> Dict[str, Any]:
        """
        Retorna un perfil de estilo táctico por defecto
        
        Returns:
            Perfil de estilo táctico por defecto
        """
        return {
            'possession_style': {
                'control_tendency': 0.5,
                'territorial_dominance': 0.5,
                'progression_efficiency': 0.5,
                'ball_recovery': 0.5,
                'buildup_success': 0.5, 
                'pressure_resistance': 0.5
            },
            'pressing_style': {
                'intensity': 0.5,
                'effectiveness': 0.5,
                'press_resistance': 0.5,
                'defensive_aggression': 0.5,
                'coordination': 0.5,
                'efficiency': 0.5,
                'avg_distance': 0.5
            },
            'attacking_style': {
                'directness': 0.5,
                'wing_play': 0.5,
                'positional_fluidity': 0.5,
                'transition_speed': 0.5
            },
            'defensive_style': {
                'line_height': 0.5,
                'defensive_activity': 0.5,
                'pressing_tendency': 0.5,
                'interception_focus': 0.5,
                'clearance_preference': 0.5
            },
            'tactical_traits': {
                'strengths': ["Estilo equilibrado"],
                'weaknesses': ["Sin debilidades específicas identificadas"]
            },
            'tactical_indices': {
                'dominance_index': 0.5,
                'defensive_aggression_index': 0.5,
                'attack_threat_index': 0.5,
                'tactical_flexibility_index': 0.5
            }
        }
    
    def _identify_tactical_traits(self, style_profile: Dict[str, Any]) -> tuple:
        """
        Identifica fortalezas y debilidades tácticas a partir del perfil de estilo
        
        Args:
            style_profile: Perfil de estilo táctico del equipo
            
        Returns:
            Tupla de (fortalezas, debilidades)
        """
        strengths = []
        weaknesses = []
        
        # Analizar posesión
        if style_profile['possession_style']['control_tendency'] > 0.65:
            strengths.append("Dominio de posesión")
        elif style_profile['possession_style']['control_tendency'] < 0.4:
            weaknesses.append("Dificultad para mantener posesión")
            
        # Análisis de presión
        if style_profile['pressing_style']['intensity'] > 0.7:
            strengths.append("Pressing intenso")
        if style_profile['pressing_style']['effectiveness'] > 0.65:
            strengths.append("Pressing efectivo")
        elif style_profile['pressing_style']['effectiveness'] < 0.4:
            weaknesses.append("Pressing ineficaz")
            
        # Análisis ofensivo
        if style_profile['attacking_style']['directness'] > 0.7:
            strengths.append("Ataque directo efectivo")
        if style_profile['attacking_style']['transition_speed'] > 0.7:
            strengths.append("Contraataques rápidos")
              # Análisis defensivo
        if style_profile['defensive_style']['defensive_activity'] > 0.65:
            strengths.append("Defensa activa")
        if style_profile['defensive_style']['line_height'] < 0.3:
            strengths.append("Bloque defensivo sólido")
        elif style_profile['defensive_style']['line_height'] > 0.7:
            if style_profile['pressing_style']['effectiveness'] < 0.5:
                weaknesses.append("Línea defensiva vulnerable")
                
        return strengths, weaknesses
    
    def _calculate_tactical_indices(self, tactical_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calcula índices tácticos compuestos a partir de datos tácticos
        
        Args:
            tactical_data: Datos tácticos del equipo
            
        Returns:
            Diccionario con índices tácticos compuestos
        """
        indices = {}
        
        # Índice de dominancia
        if 'possession_style' in tactical_data:
            possession = tactical_data['possession_style']
            indices['dominance_index'] = (
                possession.get('control_tendency', 0.5) * 0.5 +
                possession.get('territorial_dominance', 0.5) * 0.3 +
                possession.get('progression_efficiency', 0.5) * 0.2
            )
        
        # Índice de agresividad defensiva
        if 'pressing_style' in tactical_data and 'defensive_style' in tactical_data:
            press = tactical_data['pressing_style']
            defense = tactical_data['defensive_style']
            
            indices['defensive_aggression_index'] = (
                press.get('intensity', 0.5) * 0.4 +
                press.get('effectiveness', 0.5) * 0.3 +
                defense.get('defensive_activity', 0.5) * 0.3
            )
        
        # Índice de peligrosidad ofensiva
        if 'attacking_style' in tactical_data:
            attack = tactical_data['attacking_style']
            
            indices['attack_threat_index'] = (
                attack.get('directness', 0.5) * 0.25 +
                attack.get('transition_speed', 0.5) * 0.35 +
                attack.get('positional_fluidity', 0.5) * 0.15 +
                attack.get('wing_play', 0.5) * 0.25
            )
        
        # Índice de adaptabilidad táctica
        if all(key in tactical_data for key in ['possession_style', 'pressing_style', 'attacking_style', 'defensive_style']):
            indices['tactical_flexibility_index'] = (                tactical_data['possession_style'].get('pressure_resistance', 0.5) * 0.25 +
                tactical_data['pressing_style'].get('coordination', 0.5) * 0.25 +
                tactical_data['attacking_style'].get('positional_fluidity', 0.5) * 0.25 +
                tactical_data['defensive_style'].get('defensive_activity', 0.5) * 0.25
            )
        
        return indices
    
    def extract_tactical_features(self, match_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extrae características tácticas avanzadas a partir de datos de partidos
        
        Args:
            match_data: Lista de datos de partidos para analizar
            
        Returns:
            Diccionario con características tácticas extraídas
        """
        try:
            if not match_data or len(match_data) < 2:
                return self._get_default_tactical_features()
                
            # Inicializar contenedor de características
            tactical_features = {}
            
            # Extraer características de posesión
            possession_features = self._extract_possession_features(match_data)
            tactical_features.update(possession_features)
            
            # Extraer características de pressing
            pressing_features = self._extract_pressing_features(match_data)
            tactical_features.update(pressing_features)
            
            # Extraer características de ataque
            attacking_features = self._extract_attacking_features(match_data)
            tactical_features.update(attacking_features)
            
            # Extraer características defensivas
            defensive_features = self._extract_defensive_features(match_data)
            tactical_features.update(defensive_features)
            
            # Extraer características de transición
            transition_features = self._extract_transition_features(match_data)
            tactical_features.update(transition_features)
            
            # Añadir índices tácticos compuestos
            tactical_features.update(self._calculate_tactical_indices(tactical_features))
            
            return tactical_features
            
        except Exception as e:
            logger.error(f"Error al extraer características tácticas: {e}")
            return self._get_default_tactical_features()
    
    def _get_default_tactical_features(self) -> Dict[str, float]:
        """Devuelve características tácticas por defecto"""
        return {
            'possession_control': 0.5,
            'territorial_dominance': 0.5,
            'progression_efficiency': 0.5,
            'pressure_resistance': 0.5,
            'pressing_intensity': 0.5,
            'pressing_success': 0.5, 
            'defensive_aggression': 0.5,
            'pressing_line_height': 0.5,
            'attack_directness': 0.5,
            'wing_play': 0.5,
            'creative_play': 0.5,            'counter_attack_threat': 0.5,
            'line_height': 0.5,
            'defensive_compactness': 0.5,
            'tackle_success_rate': 0.5,
            'aerial_dominance': 0.5,
            'counter_attack_speed': 0.5,
            'defensive_transition_organization': 0.5,
            'transition_goals_for': 0.5,
            'transition_goals_against': 0.5,
            'dominance_index': 0.5,
            'defensive_aggression_index': 0.5,
            'attack_threat_index': 0.5,
            'tactical_flexibility_index': 0.5
        }
    
    def _extract_possession_features(self, match_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extrae características relacionadas con la posesión
        
        Args:
            match_data: Datos de partidos para analizar
            
        Returns:
            Características de posesión extraídas
        """
        features = {}
        
        # Calcular estadísticas de posesión
        total_possession = 0.0
        territorial_dominance = 0.0
        progression_efficiency = 0.0
        pressure_resistance = 0.0
        
        for match in match_data:
            stats = match.get('statistics', {})
            
            # Posesión básica
            possession = stats.get('ball_possession', 0.0)
            if isinstance(possession, str) and possession.endswith('%'):
                possession = float(possession.strip('%')) / 100
            else:
                possession = float(possession) if possession else 0.5
            total_possession += possession
            
            # Dominancia territorial (basada en posesión in campo contrario)
            opp_half_possession = stats.get('possession_in_opponent_half', 0.0)
            if isinstance(opp_half_possession, str) and opp_half_possession.endswith('%'):
                opp_half_possession = float(opp_half_possession.strip('%')) / 100
            else:
                opp_half_possession = float(opp_half_possession) if opp_half_possession else 0.5
            territorial_dominance += opp_half_possession
            
            # Eficiencia de progresión (pases progresivos completados)
            progressive_passes = stats.get('progressive_passes', 0)
            progressive_pass_accuracy = stats.get('progressive_pass_accuracy', 0.0)
            if isinstance(progressive_pass_accuracy, str) and progressive_pass_accuracy.endswith('%'):
                progressive_pass_accuracy = float(progressive_pass_accuracy.strip('%')) / 100
            else:
                progressive_pass_accuracy = float(progressive_pass_accuracy) if progressive_pass_accuracy else 0.5
            progression_efficiency += progressive_pass_accuracy
            
            # Resistencia a la presión (% pases completados bajo presión)
            passes_under_pressure = stats.get('passes_under_pressure_completed', 0.0)
            if isinstance(passes_under_pressure, str) and passes_under_pressure.endswith('%'):
                passes_under_pressure = float(passes_under_pressure.strip('%')) / 100
            else:
                passes_under_pressure = float(passes_under_pressure) if passes_under_pressure else 0.5
            pressure_resistance += passes_under_pressure
        
        # Promediar valores
        num_matches = max(1, len(match_data))
        features['possession_control'] = total_possession / num_matches
        features['territorial_dominance'] = territorial_dominance / num_matches
        features['progression_efficiency'] = progression_efficiency / num_matches
        features['pressure_resistance'] = pressure_resistance / num_matches
        
        return features
        
    def _extract_pressing_features(self, match_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extrae características relacionadas con el pressing
        
        Args:
            match_data: Datos de partidos para analizar
            
        Returns:
            Características de pressing extraídas
        """
        features = {}
        
        # Inicializar valores
        pressing_intensity = 0.0
        pressing_success = 0.0
        defensive_aggression = 0.0
        pressing_line_height = 0.0
        
        for match in match_data:
            stats = match.get('statistics', {})
            
            # Intensidad de pressing (PPDA - pases permitidos por acción defensiva)
            ppda = stats.get('ppda', 0.0)
            # PPDA más bajo indica pressing más intenso, por lo que invertimos la relación
            if ppda > 0:
                pressing_intensity += 1 / min(20, ppda)  # Normalizar con máximo de 20
            else:
                pressing_intensity += 0.5
                
            # Éxito de pressing (balones recuperados en campo contrario)
            recoveries = stats.get('ball_recoveries_opponent_half', 0)
            norm_recoveries = min(1.0, recoveries / 30.0)  # Normalizar con máximo de 30
            pressing_success += norm_recoveries
            
            # Agresividad defensiva (faltas, duelos ganados, intercepciones)
            fouls = stats.get('fouls', 0)
            tackles = stats.get('tackles_successful', 0)
            interceptions = stats.get('interceptions', 0)
            
            defensive_actions = fouls + tackles + interceptions
            norm_defensive_actions = min(1.0, defensive_actions / 50.0)  # Máximo 50
            defensive_aggression += norm_defensive_actions
            
            # Altura de la línea de pressing
            defensive_line = stats.get('defensive_line_height', 0.0)
            if isinstance(defensive_line, str) and defensive_line.endswith('m'):
                defensive_line = float(defensive_line.strip('m'))
                # Normalizar - línea más alta (menor distancia a portería rival) indica pressing más alto
                normalized_line = min(1.0, max(0.0, (50 - defensive_line) / 20))
                pressing_line_height += normalized_line
            else:
                pressing_line_height += 0.5
        
        # Promediar valores
        num_matches = max(1, len(match_data))
        features['pressing_intensity'] = pressing_intensity / num_matches
        features['pressing_success'] = pressing_success / num_matches
        features['defensive_aggression'] = defensive_aggression / num_matches
        features['pressing_line_height'] = pressing_line_height / num_matches
        
        return features
        
    def _extract_attacking_features(self, match_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extrae características relacionadas con el ataque
        
        Args:
            match_data: Datos de partidos para analizar
            
        Returns:
            Características de ataque extraídas
        """
        features = {}
        
        # Inicializar valores
        directness = 0.0
        wing_play = 0.0
        creative_play = 0.0
        counter_attack_threat = 0.0
        
        for match in match_data:
            stats = match.get('statistics', {})
            
            # Directness (ratio de pases directos vs. pases cortos)
            long_passes = stats.get('long_passes', 0)
            total_passes = stats.get('total_passes', 1)  # Evitar división por cero
            
            directness_ratio = min(1.0, long_passes / max(1, total_passes) * 3)  # Factor 3 para normalizar
            directness += directness_ratio
            
            # Juego por bandas (centros, jugadas por banda)
            crosses = stats.get('crosses_completed', 0)
            attacks_from_wings = stats.get('attacks_from_wings', 0)
            
            wing_metric = min(1.0, (crosses + attacks_from_wings) / 30.0)  # Normalizar con máximo de 30
            wing_play += wing_metric
            
            # Juego creativo (pases clave, ocasiones creadas)
            key_passes = stats.get('key_passes', 0)
            created_chances = stats.get('big_chances_created', 0)
            
            creative_metric = min(1.0, (key_passes + created_chances * 2) / 15.0)  # Normalizar, doble peso a grandes ocasiones
            creative_play += creative_metric
            
            # Amenaza de contraataque (goles en contraataque, velocidad de transición)
            counter_attack_goals = stats.get('goals_from_counter', 0)
            counter_attack_shots = stats.get('shots_from_counter', 0)
            
            counter_metric = min(1.0, (counter_attack_goals * 3 + counter_attack_shots) / 10.0)  # Normalizar
            counter_attack_threat += counter_metric
        
        # Promediar valores
        num_matches = max(1, len(match_data))
        features['attack_directness'] = directness / num_matches
        features['wing_play'] = wing_play / num_matches
        features['creative_play'] = creative_play / num_matches
        features['counter_attack_threat'] = counter_attack_threat / num_matches
        
        return features
        
    def _extract_defensive_features(self, match_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extrae características relacionadas con la defensa
        
        Args:
            match_data: Datos de partidos para analizar
            
        Returns:
            Características defensivas extraídas
        """
        features = {}
        
        # Inicializar valores
        defensive_line_height = 0.0
        compactness = 0.0
        tackle_success = 0.0
        aerial_dominance = 0.0
        
        for match in match_data:
            stats = match.get('statistics', {})
            
            # Altura de la línea defensiva
            def_line = stats.get('defensive_line_height', 0.0)
            if isinstance(def_line, str) and def_line.endswith('m'):
                def_line = float(def_line.strip('m'))
                # Normalizar - menor distancia indica línea más alta
                normalized_height = min(1.0, max(0.0, (50 - def_line) / 20))
                defensive_line_height += normalized_height
            else:
                defensive_line_height += 0.5
                
            # Compacidad defensiva (distancia entre líneas)
            team_width = stats.get('team_width', 0.0)
            team_depth = stats.get('team_depth', 0.0)
            
            if team_width > 0 and team_depth > 0:
                # Menor ancho y profundidad indican mayor compacidad
                compactness_value = 1.0 - min(1.0, (team_width * team_depth) / 1200)
                compactness += compactness_value
            else:
                compactness += 0.5
                
            # Éxito en tackles
            tackles_attempted = max(1, stats.get('tackles_attempted', 1))  # Evitar división por cero
            tackles_successful = stats.get('tackles_successful', 0)
            
            tackle_rate = min(1.0, tackles_successful / tackles_attempted)
            tackle_success += tackle_rate
            
            # Dominio aéreo
            aerial_duels_won = stats.get('aerial_duels_won', 0)
            aerial_duels_total = max(1, stats.get('aerial_duels_total', 1))  # Evitar división por cero
            
            aerial_rate = min(1.0, aerial_duels_won / aerial_duels_total)
            aerial_dominance += aerial_rate
          # Promediar valores
        num_matches = max(1, len(match_data))
        features['line_height'] = defensive_line_height / num_matches
        features['defensive_compactness'] = compactness / num_matches
        features['tackle_success_rate'] = tackle_success / num_matches
        features['aerial_dominance'] = aerial_dominance / num_matches
        
        return features
        
    def _extract_transition_features(self, match_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Extrae características relacionadas con transiciones
        
        Args:
            match_data: Datos de partidos para analizar
            
        Returns:
            Características de transición extraídas
        """
        features = {}
        
        # Inicializar valores
        counter_attack_speed = 0.0
        defensive_transition_organization = 0.0
        transition_goals_for = 0.0
        transition_goals_against = 0.0
        
        for match in match_data:
            stats = match.get('statistics', {})
            
            # Velocidad de contraataque
            counter_attack_time = stats.get('avg_counter_attack_time', 0.0)
            if counter_attack_time > 0:
                # Menor tiempo indica contraataque más rápido
                normalized_speed = min(1.0, 15.0 / counter_attack_time)
                counter_attack_speed += normalized_speed
            else:
                counter_attack_speed += 0.5
                
            # Organización en transición defensiva (goles recibidos en contraataque)
            goals_conceded_counter = stats.get('goals_conceded_from_counter', 0)
            # Menos goles concedidos indica mejor organización
            defensive_organization = max(0.0, 1.0 - min(1.0, goals_conceded_counter / 3.0))
            defensive_transition_organization += defensive_organization
            
            # Goles a favor en transiciones
            goals_from_transitions = stats.get('goals_from_transitions', 0)
            normalized_goals_for = min(1.0, goals_from_transitions / 3.0)
            transition_goals_for += normalized_goals_for
            
            # Goles en contra en transiciones
            goals_against_transitions = stats.get('goals_conceded_from_transitions', 0)
            # Invertimos para que mayor valor sea mejor defensa
            normalized_goals_against = max(0.0, 1.0 - min(1.0, goals_against_transitions / 3.0))
            transition_goals_against += normalized_goals_against
        
        # Promediar valores
        num_matches = max(1, len(match_data))
        features['counter_attack_speed'] = counter_attack_speed / num_matches
        features['defensive_transition_organization'] = defensive_transition_organization / num_matches
        features['transition_goals_for'] = transition_goals_for / num_matches
        features['transition_goals_against'] = transition_goals_against / num_matches
        
        return features


class FormationAnalyzer:
    """
    Analiza las formaciones de los equipos y su impacto táctico.
    """
    
    # Catálogo de formaciones comunes
    COMMON_FORMATIONS = {
        '4-3-3': {
            'description': 'Formación ofensiva equilibrada con tres atacantes',
            'strengths': ['amplitud ofensiva', 'presión alta', 'transiciones rápidas'],
            'weaknesses': ['vulnerabilidad en contraataques', 'espacios entre líneas'],
            'counter_formations': ['5-3-2', '4-5-1', '3-5-2']
        },
        '4-4-2': {
            'description': 'Formación clásica equilibrada',
            'strengths': ['solidez defensiva', 'simplicidad táctica', 'presencia en área'],
            'weaknesses': ['desventaja numérica en medio campo', 'dificultad contra posesión'],
            'counter_formations': ['4-3-3', '4-2-3-1', '3-4-3']
        },
        '4-2-3-1': {
            'description': 'Formación moderna con equilibrio defensivo-ofensivo',
            'strengths': ['solidez defensiva', 'transición ofensiva', 'presión coordinada'],
            'weaknesses': ['dependencia del mediapunta', 'amplitud limitada'],
            'counter_formations': ['3-5-2', '4-3-3', '4-1-4-1']
        },
        '3-5-2': {
            'description': 'Formación con tres centrales y carrileros',
            'strengths': ['solidez defensiva central', 'superioridad numérica en medio', 'contraataque'],
            'weaknesses': ['vulnerabilidad en bandas', 'dependencia de carrileros'],
            'counter_formations': ['4-3-3', '3-4-3', '4-4-2']
        },
        '3-4-3': {
            'description': 'Formación ofensiva con tres centrales',
            'strengths': ['presión alta', 'superioridad ofensiva', 'amplitud'],
            'weaknesses': ['espacios entre líneas', 'transición defensiva'],
            'counter_formations': ['4-5-1', '5-3-2', '4-2-3-1']
        },
        '5-3-2': {
            'description': 'Formación defensiva con cinco defensas',
            'strengths': ['solidez defensiva extrema', 'contraataque', 'compacidad'],
            'weaknesses': ['dificultad para atacar', 'posesión limitada'],
            'counter_formations': ['3-4-3', '4-2-3-1', '4-3-3']        
        }
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def identify_formation(self, lineup_data: Optional[Dict[str, Any]]) -> str:
        """
        Identifica la formación basada en datos de alineación.
        
        Args:
            lineup_data: Datos de alineación del equipo
            
        Returns:
            String con la formación identificada (ej: "4-3-3")
        """
        try:
            if not lineup_data:
                return '4-4-2'  # Formación por defecto si no hay datos
                
            if 'formation' in lineup_data:
                return lineup_data.get('formation', '4-4-2')  # Usar formación proporcionada o default
            
            # Si no hay formación explícita, intentar deducirla de las posiciones
            if 'startXI' in lineup_data:
                positions = []
                for player in lineup_data['startXI']:
                    if isinstance(player, dict) and 'player' in player:
                        positions.append(player['player'].get('pos', ''))
                
                # Contar por posiciones
                defenders = len([p for p in positions if p == 'D'])
                midfielders = len([p for p in positions if p == 'M'])
                forwards = len([p for p in positions if p == 'F'])
                
                # Determinar formación basada en conteos
                formation = f"{defenders}-{midfielders}-{forwards}"
                
                # Verificar formación válida
                if formation in self.COMMON_FORMATIONS:
                    return formation
                
                # Si no es una formación estándar, aproximar a la más cercana
                return self._approximate_formation(defenders, midfielders, forwards)
            
            return '4-4-2'  # Formación por defecto
        
        except Exception as e:
            self.logger.error(f"Error identificando formación: {e}")
            return '4-4-2'  # Formación por defecto
    
    def _approximate_formation(self, defenders: int, midfielders: int, forwards: int) -> str:
        """
        Aproxima la formación más cercana a los conteos dados.
        """
        common_counts = {
            '4-3-3': (4, 3, 3),
            '4-4-2': (4, 4, 2),
            '4-2-3-1': (4, 5, 1),  # 2 + 3 mediocampistas
            '3-5-2': (3, 5, 2),
            '3-4-3': (3, 4, 3),
            '5-3-2': (5, 3, 2)
        }
        
        best_match = '4-4-2'
        min_diff = float('inf')
        
        for formation, counts in common_counts.items():
            diff = abs(defenders - counts[0]) + abs(midfielders - counts[1]) + abs(forwards - counts[2])
            if diff < min_diff:
                min_diff = diff
                best_match = formation
        
        return best_match
    
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
                'zones_analysis': zones_analysis,
                'key_battles': self._identify_key_tactical_battles(home_formation, away_formation)
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error analizando enfrentamiento de formaciones: {e}")
            return {
                'home_formation': home_formation,
                'away_formation': away_formation,
                'tactical_advantage': 'neutral',
                'advantage_score': 0.0
            }
    
    def _analyze_formation_zones(self, home_formation: str, away_formation: str) -> Dict[str, Any]:
        """
        Analiza las ventajas por zonas del campo según las formaciones.
        """
        # Definición de formaciones por zonas (central, banda izquierda, banda derecha)
        zone_strengths = {
            '4-3-3': {'central': 0.6, 'wings': 0.8},
            '4-4-2': {'central': 0.7, 'wings': 0.6},
            '4-2-3-1': {'central': 0.8, 'wings': 0.6},
            '3-5-2': {'central': 0.8, 'wings': 0.5},
            '3-4-3': {'central': 0.6, 'wings': 0.8},
            '5-3-2': {'central': 0.9, 'wings': 0.4}
        }
        
        # Obtener fortalezas de zona
        home_strengths = zone_strengths.get(home_formation, {'central': 0.7, 'wings': 0.6})
        away_strengths = zone_strengths.get(away_formation, {'central': 0.7, 'wings': 0.6})
        
        # Calcular ventajas relativas
        central_advantage = home_strengths['central'] - away_strengths['central']
        wings_advantage = home_strengths['wings'] - away_strengths['wings']
        
        return {
            'central_midfield': {
                'advantage': 'home' if central_advantage > 0.1 else 'away' if central_advantage < -0.1 else 'neutral',
                'advantage_score': round(central_advantage, 2)
            },
            'wings': {
                'advantage': 'home' if wings_advantage > 0.1 else 'away' if wings_advantage < -0.1 else 'neutral',
                'advantage_score': round(wings_advantage, 2)
            }
        }
    
    def _identify_key_tactical_battles(self, home_formation: str, away_formation: str) -> List[Dict[str, str]]:
        """
        Identifica batallas tácticas clave entre las formaciones.
        """
        battles = []
        
        # Definir batallas específicas según combinaciones de formaciones
        if home_formation == '4-3-3' and away_formation == '4-4-2':
            battles.append({
                'description': 'Mediocampo central del 4-3-3 contra la línea de 4 mediocampistas del 4-4-2',
                'key_factor': 'El 4-3-3 podría dominar el centro, pero el 4-4-2 tiene ventaja en transiciones'
            })
        elif home_formation == '3-5-2' and away_formation == '4-3-3':
            battles.append({
                'description': 'Carrileros del 3-5-2 contra extremos del 4-3-3',
                'key_factor': 'Los extremos del 4-3-3 podrían explotar los espacios detrás de los carrileros'
            })
        elif home_formation == '4-2-3-1' and away_formation == '4-4-2':
            battles.append({
                'description': 'Doble pivote del 4-2-3-1 contra los dos delanteros del 4-4-2',
                'key_factor': 'El doble pivote puede superar la primera línea de presión del 4-4-2'
            })
        else:
            # Batalla genérica si no hay una específica definida
            battles.append({
                'description': f'Enfrentamiento táctico entre {home_formation} y {away_formation}',
                'key_factor': 'La efectividad en transiciones y la presión podrían ser determinantes'
            })
        
        return battles

# Integrar el FormationAnalyzer con el TacticalAnalyzer
def analyze_tactical_formation_matchup(home_team_id: int, away_team_id: int, fixture_id: int) -> Dict[str, Any]:
    """
    Analiza el enfrentamiento táctico completo, incluyendo formaciones.
    
    Args:
        home_team_id: ID del equipo local
        away_team_id: ID del equipo visitante
        fixture_id: ID del partido
        
    Returns:
        Diccionario con análisis táctico completo
    """
    from data import get_lineup_data
    
    try:
        # Obtener datos de alineaciones
        lineup_data = get_lineup_data(fixture_id)
        
        if not lineup_data or 'response' not in lineup_data or not lineup_data['response']:
            return {
                'tactical_analysis': 'No hay datos suficientes para realizar un análisis táctico',
                'formation_matchup': 'Desconocido',
                'tactical_advantage': 'neutral',
                'advantage_score': 0.0
            }
        
        # Extraer alineaciones
        home_lineup = None
        away_lineup = None
        
        for team_lineup in lineup_data['response']:
            team_id = team_lineup.get('team', {}).get('id')
            if team_id == home_team_id:
                home_lineup = team_lineup
            elif team_id == away_team_id:
                away_lineup = team_lineup
        
        # Analizar formaciones
        formation_analyzer = FormationAnalyzer()
        home_formation = formation_analyzer.identify_formation(home_lineup)
        away_formation = formation_analyzer.identify_formation(away_lineup)
        
        # Analizar matchup de formaciones
        formation_matchup = formation_analyzer.analyze_formation_matchup(home_formation, away_formation)
        
        # Añadir otra información táctica relevante aquí
        
        return {
            'formation_matchup': formation_matchup,
            'tactical_insights': {
                'home_tactical_style': formation_analyzer.COMMON_FORMATIONS.get(home_formation, {}).get('description', 'Estilo táctico desconocido'),
                'away_tactical_style': formation_analyzer.COMMON_FORMATIONS.get(away_formation, {}).get('description', 'Estilo táctico desconocido'),
                'key_matchup_zones': formation_matchup['zones_analysis'],
                'key_battles': formation_matchup['key_battles']
            },
            'advantage': {
                'team': formation_matchup['tactical_advantage'],
                'score': formation_matchup['advantage_score'],
                'description': f"Ventaja {'local' if formation_matchup['tactical_advantage'] == 'home' else 'visitante' if formation_matchup['tactical_advantage'] == 'away' else 'neutral'} basada en análisis de formaciones tácticas."
            }
        }
        
    except Exception as e:
        logger.error(f"Error en análisis táctico de formaciones: {e}")
        return {
            'tactical_analysis': f'Error en análisis: {str(e)}',
            'formation_matchup': 'Error',
            'tactical_advantage': 'neutral',
            'advantage_score': 0.0
        }
