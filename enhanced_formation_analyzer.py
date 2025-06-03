from typing import Dict, List, Any, Optional
from tactical_analysis import TacticalAnalyzer

class EnhancedFormationAnalyzer:
    """
    Analizador mejorado de formaciones y tácticas que combina análisis de formación
    con análisis táctico detallado.
    """
    
    def __init__(self):
        self.tactical_analyzer = TacticalAnalyzer()
        
    def identify_formation(self, lineup_data: Optional[Dict[str, Any]]) -> str:
        """Identifica la formación base del equipo."""
        if not lineup_data or 'formation' not in lineup_data:
            return 'Unknown'
        return lineup_data.get('formation', 'Unknown')

    def analyze_formation_matchup(self, home_formation: str, away_formation: str) -> Dict[str, Any]:
        """
        Analiza el enfrentamiento táctico entre dos formaciones.
        """
        # Mapeo de formaciones a estilos tácticos generales
        formation_styles = {
            '4-3-3': {'style': 'attacking', 'possession': 'high', 'press': 'high'},
            '4-2-3-1': {'style': 'balanced', 'possession': 'medium', 'press': 'selective'},
            '3-5-2': {'style': 'wing-play', 'possession': 'medium', 'press': 'medium'},
            '5-3-2': {'style': 'defensive', 'possession': 'low', 'press': 'low'},
            '4-4-2': {'style': 'direct', 'possession': 'medium', 'press': 'medium'},
            '3-4-3': {'style': 'attacking', 'possession': 'high', 'press': 'high'},
            '5-4-1': {'style': 'defensive', 'possession': 'low', 'press': 'low'},
            '4-5-1': {'style': 'defensive', 'possession': 'medium', 'press': 'medium'},
            '4-1-4-1': {'style': 'possession', 'possession': 'high', 'press': 'medium'},
            '4-4-1-1': {'style': 'counter', 'possession': 'low', 'press': 'selective'},
        }

        home_style = formation_styles.get(home_formation, {'style': 'unknown', 'possession': 'unknown', 'press': 'unknown'})
        away_style = formation_styles.get(away_formation, {'style': 'unknown', 'possession': 'unknown', 'press': 'unknown'})

        # Análisis del matchup
        analysis = self._analyze_style_matchup(home_style, away_style)
        
        # Predicción de posesión basada en estilos
        possession_prediction = self._predict_possession(home_style, away_style)
        
        # Ventajas tácticas
        advantages = self._identify_tactical_advantages(home_formation, away_formation)

        return {
            'analysis': analysis,
            'possession_prediction': possession_prediction,
            'advantages': advantages
        }

    def _analyze_style_matchup(self, home_style: Dict[str, str], away_style: Dict[str, str]) -> str:
        """Analiza el enfrentamiento de estilos tácticos."""
        if home_style['style'] == 'unknown' or away_style['style'] == 'unknown':
            return 'No hay suficiente información para analizar el matchup táctico'
            
        analysis = []
        
        # Analizar dinámica de posesión
        if home_style['possession'] == 'high' and away_style['possession'] == 'high':
            analysis.append('Se espera una batalla por la posesión')
        elif home_style['possession'] == 'high' and away_style['possession'] == 'low':
            analysis.append('Local probablemente domine la posesión')
        elif home_style['possession'] == 'low' and away_style['possession'] == 'high':
            analysis.append('Visitante probablemente domine la posesión')
            
        # Analizar presión
        if home_style['press'] == 'high':
            analysis.append('Local buscará presionar alto')
        if away_style['press'] == 'high':
            analysis.append('Visitante buscará presionar alto')
            
        # Análisis de estilos
        if home_style['style'] == 'attacking' and away_style['style'] == 'defensive':
            analysis.append('Partido de ataque vs defensa')
        elif home_style['style'] == 'defensive' and away_style['style'] == 'attacking':
            analysis.append('Local buscará contragolpear')
            
        return '. '.join(analysis) + '.'

    def _predict_possession(self, home_style: Dict[str, str], away_style: Dict[str, str]) -> Dict[str, float]:
        """Predice la distribución probable de la posesión."""
        possession_values = {'high': 0.6, 'medium': 0.5, 'low': 0.4, 'unknown': 0.5}
        
        home_base = possession_values[home_style['possession']]
        away_base = possession_values[away_style['possession']]
        
        # Normalizar para que sumen 1
        total = home_base + away_base
        return {
            'home': round(home_base / total, 2),
            'away': round(away_base / total, 2)
        }

    def _identify_tactical_advantages(self, home_formation: str, away_formation: str) -> List[str]:
        """Identifica ventajas tácticas basadas en las formaciones."""
        advantages = []
        
        # Ventajas numéricas por zona
        home_numbers = self._get_formation_numbers(home_formation)
        away_numbers = self._get_formation_numbers(away_formation)
        
        if not home_numbers or not away_numbers:
            return []
            
        # Ventaja en mediocampo
        if home_numbers['mid'] > away_numbers['mid']:
            advantages.append('Local tiene superioridad numérica en mediocampo')
        elif away_numbers['mid'] > home_numbers['mid']:
            advantages.append('Visitante tiene superioridad numérica en mediocampo')
            
        # Ventaja en ataque
        if home_numbers['fwd'] > away_numbers['def']:
            advantages.append('Local tiene superioridad ofensiva')
        if away_numbers['fwd'] > home_numbers['def']:
            advantages.append('Visitante tiene superioridad ofensiva')
            
        return advantages

    def _get_formation_numbers(self, formation: str) -> Optional[Dict[str, int]]:
        """Extrae números de jugadores por línea de la formación."""
        try:
            if formation == 'Unknown':
                return None
                
            parts = formation.split('-')
            if len(parts) < 3:
                return None
                
            return {
                'def': int(parts[0]),
                'mid': sum(int(x) for x in parts[1:-1]),
                'fwd': int(parts[-1])
            }
        except:
            return None
