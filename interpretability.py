# interpretability.py
import shap
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import logging
from typing import Dict, List, Any, Union, Optional
import seaborn as sns
from tactical_analysis import TacticalAnalyzer

logger = logging.getLogger(__name__)

def shap_analysis(model_output: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """
    Perform SHAP analysis on model output
    """
    # Convert inputs to numpy arrays if needed
    if not isinstance(model_output, np.ndarray):
        model_output = np.array(model_output)
    
    # Calculate feature importance
    importance_dict = {}
    for idx, name in enumerate(feature_names):
        if idx < len(model_output):
            importance_dict[name] = float(abs(model_output[idx]))
    
    return importance_dict

def plot_feature_importance(importance_dict: Dict[str, float]) -> Figure:
    """
    Create feature importance plot
    """
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    features = list(importance_dict.keys())
    values = list(importance_dict.values())
    
    # Convert to numpy arrays for plotting
    y_pos = np.arange(len(features))
    values = np.array(values)
    
    ax.barh(y_pos, values)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance Analysis')
    
    return fig

def visualize_predictions(predictions: Dict[str, Any]) -> Figure:
    """
    Create visualization of prediction probabilities
    """
    fig = Figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    
    # Extract probabilities
    probs = [
        predictions.get('home_win', 0.0),
        predictions.get('draw', 0.0),
        predictions.get('away_win', 0.0)
    ]
    labels = ['Home Win', 'Draw', 'Away Win']
    
    # Create bar plot
    ax.bar(labels, probs)
    ax.set_ylim(0, 1)
    ax.set_ylabel('Probability')
    ax.set_title('Match Outcome Probabilities')
    
    return fig

class TacticalInterpreter:
    """
    Interpreta y visualiza análisis tácticos de partidos de fútbol
    """
    
    def __init__(self):
        self.tactical_analyzer = TacticalAnalyzer()
        
    def generate_tactical_report(self, team_matches: List[Dict[str, Any]], 
                               opponent_matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Genera un reporte táctico completo del enfrentamiento
        """        # Obtener análisis táctico
        tactical_matchup = self.tactical_analyzer.get_tactical_matchup(team_matches, opponent_matches)
        
        # Generar interpretaciones
        report = {
            'style_comparison': self._interpret_style_comparison(tactical_matchup),
            'key_battles': self._interpret_key_battles(tactical_matchup['key_battles']),
            'tactical_advantages': self._identify_tactical_advantages(tactical_matchup),
            'suggested_adaptations': self._suggest_tactical_adaptations(tactical_matchup),
            'visual_analysis': self._create_tactical_visualizations(tactical_matchup)
        }
        
        # Añadir información de impacto proyectado
        report['projected_impact'] = self._calculate_projected_impact(tactical_matchup)
        
        return report
    
    def _interpret_style_comparison(self, tactical_matchup: Dict[str, Any]) -> Dict[str, str]:
        """
        Interpreta la comparación de estilos de juego
        """
        interpretations = {}
        
        # Interpretar batalla de posesión
        poss_ratio = tactical_matchup['possession_battle']['expected_possession_ratio']
        if poss_ratio > 0.55:
            interpretations['possession'] = "Dominio esperado en posesión"
        elif poss_ratio < 0.45:
            interpretations['possession'] = "Posible inferioridad en posesión"
        else:
            interpretations['possession'] = "Equilibrio en posesión esperado"
        
        # Interpretar pressing
        press_battle = tactical_matchup['pressing_dynamics']['pressing_battle']
        if press_battle > 0.2:
            interpretations['pressing'] = "Ventaja en intensidad de pressing"
        elif press_battle < -0.2:
            interpretations['pressing'] = "Desventaja en pressing, considerar juego directo"
        else:
            interpretations['pressing'] = "Pressing equilibrado"
          # Interpretar amenaza de ataque
        attack_threat = tactical_matchup['attacking_comparison']['attacking_threat']
        if attack_threat > 0.6:
            interpretations['attack'] = "Clara ventaja ofensiva"
        elif attack_threat < 0.4:
            interpretations['attack'] = "Posibles dificultades ofensivas"
        else:
            interpretations['attack'] = "Potencial ofensivo equilibrado"
        
        # Generar resumen general para el análisis completo
        interpretations['summary'] = self._generate_style_summary(interpretations, tactical_matchup)
        
        return interpretations
    
    def _generate_style_summary(self, interpretations: Dict[str, str], tactical_matchup: Dict[str, Any]) -> str:
        """
        Genera un resumen global del análisis de estilos.
        """
        possession_text = interpretations.get('possession', 'Equilibrio en posesión')
        pressing_text = interpretations.get('pressing', 'Pressing equilibrado')
        attack_text = interpretations.get('attack', 'Potencial ofensivo equilibrado')
        
        # Agregar datos sobre índices tácticos si están disponibles
        team1_indices = tactical_matchup.get('team1_profile', {}).get('tactical_indices', {})
        team2_indices = tactical_matchup.get('team2_profile', {}).get('tactical_indices', {})
        
        tactical_insights = []
        if team1_indices and team2_indices:
            # Comparar calidad de construcción
            build_up_diff = team1_indices.get('build_up_quality', 0.5) - team2_indices.get('build_up_quality', 0.5)
            if abs(build_up_diff) > 0.15:
                team = "local" if build_up_diff > 0 else "visitante"
                tactical_insights.append(f"Ventaja del equipo {team} en la construcción de juego")
            
            # Comparar eficiencia atacante
            attacking_diff = team1_indices.get('attacking_efficiency', 0.5) - team2_indices.get('attacking_efficiency', 0.5)
            if abs(attacking_diff) > 0.15:
                team = "local" if attacking_diff > 0 else "visitante"
                tactical_insights.append(f"Mayor eficiencia ofensiva del equipo {team}")
            
            # Comparar organización defensiva
            def_org_diff = team1_indices.get('defensive_organization', 0.5) - team2_indices.get('defensive_organization', 0.5)
            if abs(def_org_diff) > 0.15:
                team = "local" if def_org_diff > 0 else "visitante"
                tactical_insights.append(f"Mejor organización defensiva del equipo {team}")
        
        # Construir el resumen final
        summary = f"{possession_text}. {pressing_text}. {attack_text}."
        
        if tactical_insights:
            summary += " " + " ".join(tactical_insights)
            
        return summary
    
    def _interpret_key_battles(self, key_battles: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Interpreta las batallas tácticas clave
        """
        interpretations = []
        
        for battle in key_battles:
            interpretation = {
                'area': battle['description'],
                'importance': battle['importance'],
                'recommendation': self._get_battle_recommendation(battle)
            }
            interpretations.append(interpretation)
        
        return interpretations
    
    def _get_battle_recommendation(self, battle: Dict[str, Any]) -> str:
        """
        Genera recomendaciones específicas para cada tipo de batalla táctica
        """
        if 'Posesión vs Pressing Alto' in battle['description']:
            return ("Priorizar salida en corto con extremos abiertos y mediocentros bajando. "
                   "Alternar con juego directo ocasional para romper el pressing.")
        
        elif 'Pressing vs Construcción' in battle['description']:
            return ("Presionar agresivamente en pérdida. Mantener líneas juntas. "
                   "Cerrar líneas de pase interior.")
        
        elif 'Ataque Directo vs Línea Alta' in battle['description']:
            return ("Buscar pases al espacio. Extremos preparados para duelos 1v1. "
                   "Delantero listo para carreras a la espalda.")
        
        return "Mantener el plan táctico base y adaptar según el desarrollo del partido."
    def _identify_tactical_advantages(self, tactical_matchup: Dict[str, Any]) -> Dict[str, Any]:
        """
        Identifica ventajas tácticas específicas
        """
        advantages = []
        
        # Analizar ventajas en posesión
        if tactical_matchup['possession_battle']['territorial_advantage'] > 0.15:
            advantages.append({
                'aspect': 'Territorial',
                'description': 'Capacidad superior para dominar el campo rival',
                'exploitation': 'Mantener presión alta y circulación rápida'
            })
        
        # Analizar ventajas en pressing
        if tactical_matchup['pressing_dynamics']['press_resistance_advantage'] > 0.15:
            advantages.append({
                'aspect': 'Resistencia al Pressing',
                'description': 'Mayor capacidad para superar presión rival',
                'exploitation': 'Invitar pressing para explotar espacios'
            })
        
        # Analizar ventajas en transiciones
        if tactical_matchup['attacking_comparison']['transition_advantage'] > 1.2:
            advantages.append({
                'aspect': 'Transiciones',
                'description': 'Superior velocidad en contraataques',
                'exploitation': 'Buscar recuperaciones en campo rival'
            })
            
        # Analizar ventajas en coordinación
        if tactical_matchup['pressing_dynamics']['coordination_advantage'] > 0.15:
            advantages.append({
                'aspect': 'Pressing Organizado',
                'description': 'Mayor coordinación en la presión colectiva',
                'exploitation': 'Presionar en bloques para forzar errores'
            })
            
        # Analizar ventajas en eficiencia de pressing
        if tactical_matchup['pressing_dynamics']['pressing_efficiency_ratio'] > 1.15:
            advantages.append({
                'aspect': 'Eficiencia de Pressing',
                'description': 'Mejor rendimiento en recuperaciones por pressing',
                'exploitation': 'Aumentar intensidad de pressing en zonas clave'
            })
            
        # Analizar ventajas en juego por bandas
        if tactical_matchup['attacking_comparison']['wing_threat'] > 0.6:
            advantages.append({
                'aspect': 'Juego por Bandas',
                'description': 'Superioridad en ataques por las bandas',
                'exploitation': 'Priorizar ataques por las bandas y centros'
            })
            
        # Determinar qué equipo tiene más ventajas
        tactical_advantages = tactical_matchup.get('tactical_advantages', {})
        advantage_metrics = {
            'possession_control': tactical_advantages.get('possession_control', 0),
            'pressing_effectiveness': tactical_advantages.get('pressing_effectiveness', 0),
            'attacking_threat': tactical_advantages.get('attacking_threat', 0),
            'defensive_solidity': tactical_advantages.get('defensive_solidity', 0),
            'tactical_flexibility': tactical_advantages.get('tactical_flexibility', 0)
        }
        
        # Calcular ventaja general
        positive_advantages = sum(1 for v in advantage_metrics.values() if v > 0.1)
        negative_advantages = sum(1 for v in advantage_metrics.values() if v < -0.1)
        neutral_count = 5 - positive_advantages - negative_advantages
        
        # Generar resumen global de ventajas
        if positive_advantages > negative_advantages + 1:
            advantage_summary = "Ventaja táctica general para el equipo local"
        elif negative_advantages > positive_advantages + 1:
            advantage_summary = "Ventaja táctica general para el equipo visitante"
        else:
            advantage_summary = "Equilibrio táctico entre ambos equipos"
          # Generar resumen detallado
        detailed_summary = []
        for aspect, value in advantage_metrics.items():
            if abs(value) > 0.15:
                team = "local" if value > 0 else "visitante"
                aspect_name = aspect.replace('_', ' ').title()
                detailed_summary.append(f"Ventaja en {aspect_name} para equipo {team}")
                
        if not detailed_summary:
            detailed_summary.append("No hay ventajas tácticas claras entre los equipos")
            
        # Crear el objeto de retorno con toda la información
        return {
            'key_points': advantages,
            'metrics': advantage_metrics,
            'summary': advantage_summary,
            'detailed_advantages': detailed_summary
        }
          # El método _calculate_projected_impact se ha movido más abajo en la clase
    # para evitar la duplicación
        
    def _suggest_tactical_adaptations(self, tactical_matchup: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sugiere adaptaciones tácticas basadas en el análisis
        """
        suggestions = []
        
        # Analizar necesidad de adaptaciones en pressing
        press_battle = tactical_matchup['pressing_dynamics']['pressing_battle']
        if press_battle < -0.2:
            suggestions.append({
                'aspect': 'Pressing',
                'suggestion': 'Reducir intensidad de pressing, preferir bloque medio',
                'reason': 'Rival superior en resistencia al pressing'
            })
        
        # Analizar adaptaciones en posesión
        poss_ratio = tactical_matchup['possession_battle']['expected_possession_ratio']
        if poss_ratio < 0.45:
            suggestions.append({
                'aspect': 'Posesión',
                'suggestion': 'Priorizar posesión efectiva sobre cantidad',
                'reason': 'Probable inferioridad en posesión'
            })
        
        # Analizar adaptaciones defensivas
        if tactical_matchup['attacking_comparison']['wing_threat'] > 0.6:
            suggestions.append({
                'aspect': 'Defensa Lateral',
                'suggestion': 'Reforzar cobertura en bandas',
                'reason': 'Alta amenaza por juego de bandas rival'
            })
            
        # Analizar adaptaciones por resistencia al pressing
        if tactical_matchup['pressing_dynamics']['press_resistance_advantage'] < -0.15:
            suggestions.append({
                'aspect': 'Construcción',
                'suggestion': 'Evitar construcción elaborada bajo presión',
                'reason': 'Debilidad en resistencia al pressing'
            })
            
        # Analizar adaptaciones por falta de coordinación en pressing
        if tactical_matchup['pressing_dynamics']['coordination_advantage'] < -0.15:
            suggestions.append({
                'aspect': 'Organización Defensiva',
                'suggestion': 'Enfocar en bloque compacto en vez de pressing alto',
                'reason': 'Desventaja en coordinación de pressing'
            })
            
        # Generar resumen de adaptaciones sugeridas
        home_team_suggestions = []
        away_team_suggestions = []
        
        # Determinar qué equipo necesita cada adaptación
        for suggestion in suggestions:
            # Simplificación: si el aspecto está en las ventajas del equipo visitante,
            # entonces es una adaptación para el equipo local, y viceversa
            if self._is_away_team_advantage(tactical_matchup, suggestion['aspect']):
                home_team_suggestions.append(suggestion)
            else:
                away_team_suggestions.append(suggestion)
                
        # Generar resumen global
        if home_team_suggestions and away_team_suggestions:
            summary = "Ambos equipos necesitan adaptaciones tácticas"
        elif home_team_suggestions:
            summary = "El equipo local necesita adaptaciones tácticas"
        elif away_team_suggestions:
            summary = "El equipo visitante necesita adaptaciones tácticas"
        else:
            summary = "No se requieren adaptaciones tácticas significativas"
            
        # Retornar objeto completo
        return {
            'suggestions': suggestions,
            'home_team': home_team_suggestions,
            'away_team': away_team_suggestions,
            'summary': summary
        }
        
    def _is_away_team_advantage(self, tactical_matchup: Dict[str, Any], aspect: str) -> bool:
        """
        Determina si un aspecto táctico es ventaja para el equipo visitante
        """
        # Mapeo simplificado de aspectos a métricas relevantes
        aspect_mapping = {
            'Pressing': 'pressing_battle',
            'Posesión': 'expected_possession_ratio',
            'Defensa Lateral': 'wing_threat',
            'Construcción': 'press_resistance_advantage',
            'Organización Defensiva': 'coordination_advantage'
        }
        
        # Determinar qué métrica corresponde a este aspecto
        metric_key = aspect_mapping.get(aspect)
        if not metric_key:
            return False  # Por defecto, asumimos que no es ventaja visitante
            
        # Verificar la métrica correspondiente
        if metric_key in tactical_matchup.get('pressing_dynamics', {}):
            return tactical_matchup['pressing_dynamics'][metric_key] < 0
        elif metric_key in tactical_matchup.get('possession_battle', {}):
            return tactical_matchup['possession_battle'][metric_key] < 0.5
        elif metric_key in tactical_matchup.get('attacking_comparison', {}):
            return tactical_matchup['attacking_comparison'][metric_key] < 0.5
            
        return False
    
    def _create_tactical_visualizations(self, tactical_matchup: Dict[str, Any]) -> Dict[str, Any]:
        """
        Crea visualizaciones del análisis táctico
        """
        visualizations = {}
        
        # Visualización de comparación de estilos
        style_comparison = self._create_style_radar_chart(tactical_matchup)
        visualizations['style_radar'] = style_comparison
        
        # Visualización de zonas de pressing
        press_map = self._create_pressing_heatmap(tactical_matchup)
        visualizations['press_map'] = press_map
        
        # Visualización de patrones de ataque
        attack_patterns = self._create_attack_pattern_visualization(tactical_matchup)
        visualizations['attack_patterns'] = attack_patterns
        
        return visualizations
    
    def _create_style_radar_chart(self, tactical_matchup: Dict[str, Any]) -> Figure:
        """
        Crea un gráfico radar comparando estilos de juego
        """
        fig = plt.figure(figsize=(10, 10))
        # Implementar visualización con matplotlib
        categories = ['Posesión', 'Pressing', 'Ataque Directo', 
                     'Juego de Bandas', 'Defensa Alta', 'Transiciones']
        
        team1_stats = [
            tactical_matchup['possession_battle']['expected_possession_ratio'],
            tactical_matchup['pressing_dynamics']['pressing_battle'] + 0.5,
            tactical_matchup['attacking_comparison']['attacking_threat'],
            tactical_matchup['attacking_comparison']['wing_threat'],
            tactical_matchup['defensive_style']['line_height'],
            tactical_matchup['attacking_comparison']['transition_advantage']
        ]
        
        ax = fig.add_subplot(111, projection='polar')
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        
        # Completar el círculo
        values = np.concatenate((team1_stats, [team1_stats[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        ax.plot(angles, values)
        ax.fill(angles, values, alpha=0.25)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        
        plt.title('Perfil Táctico del Equipo')
        
        return fig
    
    def _create_pressing_heatmap(self, tactical_matchup: Dict[str, Any]) -> Figure:
        """
        Crea un mapa de calor mostrando zonas de pressing
        """
        fig = plt.figure(figsize=(12, 8))
        
        # Simular datos de pressing para visualización
        press_data = np.random.rand(10, 6)  # Simplificado para ejemplo
        
        sns.heatmap(press_data, cmap='YlOrRd')
        plt.title('Intensidad de Pressing por Zonas')
        
        return fig
    
    def _create_attack_pattern_visualization(self, tactical_matchup: Dict[str, Any]) -> Figure:
        """
        Crea una visualización de patrones de ataque
        """
        fig = plt.figure(figsize=(12, 8))
        
        # Convertir datos a listas para plt.bar
        attack_zones = ['Central', 'Banda Derecha', 'Banda Izquierda']
        attack_values = [
            float(tactical_matchup['attacking_comparison']['attacking_threat'] * 100),
            float(tactical_matchup['attacking_comparison']['wing_threat'] * 100),
            float(tactical_matchup['attacking_comparison']['wing_threat'] * 90)
        ]
        
        plt.bar(attack_zones, attack_values)
        plt.title('Distribución de Ataques por Zona')
        
        return fig
    
    def _calculate_projected_impact(self, tactical_matchup: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calcula el impacto proyectado de las tácticas en el resultado
        
        Args:
            tactical_matchup: Análisis táctico del enfrentamiento
            
        Returns:
            Dict con el impacto proyectado en goles y probabilidad de victoria
        """
        try:
            # Extraer las métricas clave del análisis táctico
            tactical_advantages = tactical_matchup.get('tactical_advantages', {})
            poss_battle = tactical_matchup.get('possession_battle', {})
            pressing = tactical_matchup.get('pressing_dynamics', {})
            attacking = tactical_matchup.get('attacking_comparison', {})
            
            # Calcular factores de impacto base
            possession_impact = poss_battle.get('territorial_advantage', 0) * 0.3
            pressing_impact = pressing.get('pressing_battle', 0) * 0.25
            attacking_impact = attacking.get('attacking_threat', 0) * 0.35
            
            # Calcular impacto adicional de ventajas tácticas clave
            advantage_impact = 0
            for advantage, value in tactical_advantages.items():
                if isinstance(value, (int, float)):
                    advantage_impact += value * 0.15
            
            # Calcular impacto total en el resultado
            total_impact = possession_impact + pressing_impact + attacking_impact + advantage_impact
            
            # Traducir impacto a goles esperados (valores típicos entre -0.5 y 0.5)
            goal_impact = min(max(total_impact * 0.8, -0.8), 0.8)
            
            # Calcular impacto en probabilidad de victoria (valores típicos entre -10% y 10%)
            win_prob_impact = min(max(total_impact * 15, -15), 15)
            
            # Generar interpretación textual del impacto
            if abs(goal_impact) < 0.15:
                impact_description = "Tácticas equilibradas con impacto mínimo en el resultado"
            elif goal_impact > 0:
                impact_description = f"Ventaja táctica para el equipo local que podría aumentar su producción ofensiva en {abs(goal_impact):.2f} goles"
            else:
                impact_description = f"Ventaja táctica para el equipo visitante que podría aumentar su producción ofensiva en {abs(goal_impact):.2f} goles"
                
            # Generar recomendaciones de apuesta basadas en el análisis táctico
            betting_insight = self._generate_tactical_betting_insight(goal_impact, win_prob_impact)
            
            # Crear objeto de retorno con todos los datos calculados
            return {
                'goal_impact': round(goal_impact, 2),
                'win_probability_impact': round(win_prob_impact, 1),
                'description': impact_description,
                'betting_insight': betting_insight,
                'confidence': 'Media' if abs(goal_impact) < 0.3 else 'Alta',
                'key_factors': [
                    {'factor': 'Posesión', 'impact': round(possession_impact * 100, 1)},
                    {'factor': 'Pressing', 'impact': round(pressing_impact * 100, 1)},
                    {'factor': 'Ataque', 'impact': round(attacking_impact * 100, 1)},
                    {'factor': 'Ventajas Tácticas', 'impact': round(advantage_impact * 100, 1)}
                ]
            }
            
        except Exception as e:
            logger.error(f"Error al calcular impacto táctico proyectado: {e}")
            return {
                'goal_impact': 0,
                'win_probability_impact': 0,
                'description': "No se pudo calcular el impacto táctico",
                'betting_insight': "Sin datos suficientes para recomendación",
                'confidence': 'Baja',
                'error': str(e)
            }
            
    def _generate_tactical_betting_insight(self, goal_impact: float, win_prob_impact: float) -> str:
        """
        Genera una recomendación de apuesta basada en el análisis táctico
        
        Args:
            goal_impact: Impacto estimado en goles
            win_prob_impact: Impacto en probabilidad de victoria (porcentaje)
            
        Returns:
            Recomendación para apostadores
        """
        if abs(goal_impact) < 0.2 and abs(win_prob_impact) < 5:
            return "Sin ventaja clara para apuestas basadas en tácticas"
            
        insights = []
        
        # Analizar impacto en goles
        if goal_impact > 0.4:
            insights.append("Considerar Over 2.5 goles por ventaja táctica ofensiva local")
        elif goal_impact < -0.4:
            insights.append("Considerar Over 2.5 goles por ventaja táctica ofensiva visitante")
        elif abs(goal_impact) < 0.2:
            insights.append("Tácticas sugieren un partido equilibrado sin clara ventaja para goles")
            
        # Analizar impacto en resultado
        if win_prob_impact > 8:
            insights.append("Ventaja táctica significativa para victoria local")
        elif win_prob_impact < -8:
            insights.append("Ventaja táctica significativa para victoria visitante")
        elif abs(win_prob_impact) < 3:
            insights.append("Equilibrio táctico sugiere valor potencial en empate")
            
        if not insights:
            return "Sin ventaja clara para apuestas basadas en tácticas"
            
        return " | ".join(insights)
