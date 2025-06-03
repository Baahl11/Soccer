"""
Módulo de normalización de datos de odds

Este módulo implementa funciones para normalizar datos de odds entre diferentes
formatos, incluyendo:
- Conversión de formato API a formato interno
- Generación de datos simulados cuando es necesario
- Cálculo de probabilidades implícitas
- Análisis de valor y mercado

Autor: Equipo de Desarrollo
Fecha: Mayo 24, 2025
"""

import logging
import random
from typing import Dict, Any, Optional, List
from datetime import datetime

# Configuración de logging
logger = logging.getLogger('odds_normalizer')

class OddsNormalizer:
    """Clase para normalizar datos de odds."""
    
    def __init__(self, bookmaker_priorities: List[int] = None):
        """
        Inicializa el normalizador de odds.
        
        Args:
            bookmaker_priorities: Lista de IDs de bookmakers en orden de prioridad
        """
        self.bookmaker_priorities = bookmaker_priorities or [1, 6, 8, 2]  # Valores por defecto
        
    def normalize_api_data(self, api_odds_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normaliza datos directamente desde el formato de la API al formato interno.
        
        Args:
            api_odds_data: Datos de la API en formato original
            
        Returns:
            Datos normalizados según el formato interno del sistema
        """
        try:
            # Extraer información básica
            fixture_id = api_odds_data.get("fixture", {}).get("id", 0)
            bookmakers = api_odds_data.get("bookmakers", [])
            league = api_odds_data.get("league", {})
            
            # Preparar estructura normalizada
            normalized = {
                "simulated": len(bookmakers) == 0,
                "source": "API Football" if len(bookmakers) > 0 else "Datos simulados",
                "timestamp": datetime.now().isoformat(),
                "fixture_id": fixture_id,
                "league_id": league.get("id", 0),
                "league_name": league.get("name", "Unknown League"),
                "bookmakers_count": len(bookmakers)
            }
            
            # Si no hay datos de bookmakers, generar datos simulados
            if not bookmakers:
                return self.generate_simulated_odds(fixture_id)
            
            # Encontrar el bookmaker preferido
            preferred_bookmaker = None
            for priority_id in self.bookmaker_priorities:
                preferred_bookmaker = next((b for b in bookmakers if b.get("id") == priority_id), None)
                if preferred_bookmaker:
                    break
            
            # Si no encontramos un bookmaker preferido, usar el primero
            if not preferred_bookmaker and bookmakers:
                preferred_bookmaker = bookmakers[0]
            
            # Si no hay bookmaker, usar datos simulados
            if not preferred_bookmaker:
                return self.generate_simulated_odds(fixture_id)
            
            # Extraer y normalizar datos de mercados principales
            normalized.update(self._extract_main_markets(preferred_bookmaker, fixture_id))
            
            # Agregar todos los bookmakers disponibles
            normalized["available_bookmakers"] = [
                {"id": b.get("id"), "name": b.get("name")} for b in bookmakers
            ]
            
            # Análisis de mercado y valor
            if normalized.get("market_odds") and all(k in normalized["market_odds"] for k in ["home", "draw", "away"]):
                normalized.update(self._calculate_market_sentiment(normalized["market_odds"]))
                
            return normalized
                
        except Exception as e:
            logger.error(f"Error normalizando datos de odds: {str(e)}")
            return self.generate_simulated_odds(fixture_id if 'fixture_id' in locals() else 0)

    def _extract_main_markets(self, bookmaker: Dict[str, Any], fixture_id: int) -> Dict[str, Any]:
        """
        Extrae los mercados principales de un bookmaker.
        
        Args:
            bookmaker: Datos del bookmaker
            fixture_id: ID del partido
            
        Returns:
            Diccionario con datos de mercados normalizados
        """
        result = {}
        
        try:
            # Extraer mercados principales
            market_odds = {}
            
            # Match Winner (1X2)
            match_winner_bets = next((bet for bet in bookmaker.get("bets", []) if bet.get("id") == 1), None)
            if match_winner_bets and "values" in match_winner_bets:
                values = match_winner_bets["values"]
                home_odd = next((float(v.get("odd", "0")) for v in values if v.get("value") == "Home"), 0)
                draw_odd = next((float(v.get("odd", "0")) for v in values if v.get("value") == "Draw"), 0)
                away_odd = next((float(v.get("odd", "0")) for v in values if v.get("value") == "Away"), 0)
                
                if home_odd > 0 and draw_odd > 0 and away_odd > 0:
                    market_odds["home"] = home_odd
                    market_odds["draw"] = draw_odd
                    market_odds["away"] = away_odd
            
            # Over/Under 2.5
            over_under_bets = next((bet for bet in bookmaker.get("bets", []) if bet.get("id") == 2), None)
            if over_under_bets and "values" in over_under_bets:
                values = over_under_bets["values"]
                over_odd = next((float(v.get("odd", "0")) for v in values if v.get("value") == "Over 2.5"), 0)
                under_odd = next((float(v.get("odd", "0")) for v in values if v.get("value") == "Under 2.5"), 0)
                
                if over_odd > 0 and under_odd > 0:
                    market_odds["over_2.5"] = over_odd
                    market_odds["under_2.5"] = under_odd
            
            # Ambos equipos marcan (BTTS)
            btts_bets = next((bet for bet in bookmaker.get("bets", []) if bet.get("id") == 3), None)
            if btts_bets and "values" in btts_bets:
                values = btts_bets["values"]
                yes_odd = next((float(v.get("odd", "0")) for v in values if v.get("value") == "Yes"), 0)
                no_odd = next((float(v.get("odd", "0")) for v in values if v.get("value") == "No"), 0)
                
                if yes_odd > 0 and no_odd > 0:
                    market_odds["btts_yes"] = yes_odd
                    market_odds["btts_no"] = no_odd
            
            result["market_odds"] = market_odds
            result["bookmaker"] = {
                "id": bookmaker.get("id"),
                "name": bookmaker.get("name")
            }
            
            return result
            
        except Exception as e:
            logger.warning(f"Error extrayendo mercados para partido {fixture_id}: {str(e)}")
            return result

    def _calculate_market_sentiment(self, market_odds: Dict[str, float]) -> Dict[str, Any]:
        """
        Calcula el sentimiento del mercado basado en las cuotas.
        
        Args:
            market_odds: Diccionario con las cuotas del mercado
            
        Returns:
            Diccionario con análisis del mercado y oportunidades de valor
        """
        result = {}
        
        try:
            # Calcular probabilidades implícitas
            home_odd = market_odds.get("home", 0)
            draw_odd = market_odds.get("draw", 0)
            away_odd = market_odds.get("away", 0)
            
            if home_odd <= 1 or draw_odd <= 1 or away_odd <= 1:
                return result
            
            # Calcular probabilidades implícitas
            home_prob = 1 / home_odd
            draw_prob = 1 / draw_odd
            away_prob = 1 / away_odd
            
            # Normalizar probabilidades (eliminar margen)
            prob_sum = home_prob + draw_prob + away_prob
            if prob_sum > 0:
                home_prob = home_prob / prob_sum
                draw_prob = draw_prob / prob_sum
                away_prob = away_prob / prob_sum
            
            # Analizar sentimiento del mercado
            result["market_sentiment"] = {
                "description": "Basado en datos de mercado",
                "implied_probabilities": {
                    "home_win": round(home_prob, 2),
                    "draw": round(draw_prob, 2),
                    "away_win": round(away_prob, 2)
                }
            }
            
            # Calcular margen del bookmaker
            margin = ((1/home_odd) + (1/draw_odd) + (1/away_odd)) - 1
            result["market_sentiment"]["bookmaker_margin"] = round(margin, 2)
            
            # Identificar oportunidades de valor
            value_opportunities = []
            
            # Umbral mínimo para recomendar apuestas (odds >= 1.5)
            min_odd = 1.5
            
            if home_odd >= min_odd:
                value_opportunities.append({
                    "market": "Match Winner",
                    "selection": "Home",
                    "market_odds": home_odd,
                    "fair_odds": round(1/home_prob, 2),
                    "recommendation": self._get_recommendation(home_odd, 1/home_prob),
                    "confidence": "Media",
                    "value": round((home_odd * home_prob) - 1, 2)
                })
            
            if draw_odd >= min_odd:
                value_opportunities.append({
                    "market": "Match Winner",
                    "selection": "Draw",
                    "market_odds": draw_odd,
                    "fair_odds": round(1/draw_prob, 2),
                    "recommendation": self._get_recommendation(draw_odd, 1/draw_prob),
                    "confidence": "Baja",
                    "value": round((draw_odd * draw_prob) - 1, 2)
                })
            
            if away_odd >= min_odd:
                value_opportunities.append({
                    "market": "Match Winner",
                    "selection": "Away",
                    "market_odds": away_odd,
                    "fair_odds": round(1/away_prob, 2),
                    "recommendation": self._get_recommendation(away_odd, 1/away_prob),
                    "confidence": "Media",
                    "value": round((away_odd * away_prob) - 1, 2)
                })
                
            result["value_opportunities"] = value_opportunities
            
            return result
            
        except Exception as e:
            logger.warning(f"Error calculando sentimiento de mercado: {str(e)}")
            return result
    
    def _get_recommendation(self, market_odds: float, fair_odds: float) -> str:
        """
        Genera una recomendación basada en la comparación entre odds de mercado y odds justas.
        
        Args:
            market_odds: Odds del mercado
            fair_odds: Odds justas calculadas
            
        Returns:
            Recomendación como string
        """
        # Calcular el valor
        value = (market_odds / fair_odds) - 1
        
        # Determinar recomendación basada en el valor
        if value >= 0.1:  # 10% o más de valor
            return "Fuerte Valor"
        elif value >= 0.05:  # 5-10% de valor
            return "Valor"
        elif value >= 0:  # 0-5% de valor
            return "Valor Marginal"
        else:  # Valor negativo
            return "Neutral"
    
    def generate_simulated_odds(self, fixture_id: int) -> Dict[str, Any]:
        """
        Genera datos de odds simulados cuando no hay datos reales disponibles.
        
        Args:
            fixture_id: ID del partido
            
        Returns:
            Datos de odds simulados en el formato interno
        """
        # Establecer semilla para reproducibilidad pero con variación entre partidos
        random.seed(fixture_id + datetime.now().day)
        
        # Generar odds simuladas realistas para 1X2
        home_strength = random.uniform(0.35, 0.55)  # Fuerza relativa del equipo local
        away_strength = random.uniform(0.25, 0.45)  # Fuerza relativa del equipo visitante
        draw_strength = 1 - home_strength - away_strength  # El resto para el empate
        
        # Añadir un poco de margen del bookmaker (5-10%)
        margin = random.uniform(0.05, 0.10)
        total_prob = (home_strength + away_strength + draw_strength) * (1 + margin)
        
        # Calcular odds con margen
        home_odd = round(1 / (home_strength * (1 + margin) / total_prob), 2)
        draw_odd = round(1 / (draw_strength * (1 + margin) / total_prob), 2)
        away_odd = round(1 / (away_strength * (1 + margin) / total_prob), 2)
        
        # Generar odds para over/under 2.5
        over_prob = random.uniform(0.48, 0.55)
        under_prob = 1 - over_prob
        over_odd = round(1 / (over_prob * (1 + margin/2)), 2)
        under_odd = round(1 / (under_prob * (1 + margin/2)), 2)
        
        # Generar odds para BTTS (ambos equipos marcan)
        btts_yes_prob = random.uniform(0.5, 0.65)
        btts_no_prob = 1 - btts_yes_prob
        btts_yes_odd = round(1 / (btts_yes_prob * (1 + margin/2)), 2)
        btts_no_odd = round(1 / (btts_no_prob * (1 + margin/2)), 2)
        
        # Estructura de datos simulada
        simulated_data = {
            "simulated": True,
            "source": "Datos simulados",
            "fixture_id": fixture_id,
            "timestamp": datetime.now().isoformat(),
            "bookmakers_count": 0,
            "market_odds": {
                "home": home_odd,
                "draw": draw_odd,
                "away": away_odd,
                "over_2.5": over_odd,
                "under_2.5": under_odd,
                "btts_yes": btts_yes_odd,
                "btts_no": btts_no_odd
            },
            "market_sentiment": {
                "description": "Basado en simulación",
                "implied_probabilities": {
                    "home_win": round(home_strength, 2),
                    "draw": round(draw_strength, 2),
                    "away_win": round(away_strength, 2)
                },
                "bookmaker_margin": round(margin, 2)
            },
            "value_opportunities": []  # No hay oportunidades de valor en datos simulados
        }
        
        return simulated_data

# Función de conveniencia para facilitar el uso
def normalize_odds(api_data: Dict[str, Any], bookmaker_priorities: List[int] = None) -> Dict[str, Any]:
    """
    Normaliza datos de odds directamente desde la API al formato interno.
    
    Args:
        api_data: Datos de la API en formato original
        bookmaker_priorities: Lista opcional de IDs de bookmakers en orden de prioridad
        
    Returns:
        Datos normalizados según el formato interno del sistema
    """
    normalizer = OddsNormalizer(bookmaker_priorities)
    return normalizer.normalize_api_data(api_data)

def generate_simulated_odds(fixture_id: int) -> Dict[str, Any]:
    """
    Función auxiliar para generar datos simulados.
    
    Args:
        fixture_id: ID del partido
        
    Returns:
        Datos de odds simulados
    """
    normalizer = OddsNormalizer()
    return normalizer.generate_simulated_odds(fixture_id)
