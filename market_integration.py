"""
Módulo para integrar datos de mercado en el sistema de predicción.
Se encarga de añadir odds pre-partido como características,
implementar calibración basada en movimiento de odds y
crear un monitor para movimientos significativos.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
from odds_analyzer import OddsAnalyzer
from config import ODDS_ENDPOINTS, ODDS_BOOKMAKERS_PRIORITY, ODDS_DEFAULT_MARKETS

logger = logging.getLogger(__name__)

class MarketDataIntegrator:
    """
    Integra datos de mercado (odds) en el sistema de predicción.
    """
    def __init__(self):
        """Inicializar el integrador de datos de mercado."""
        self.odds_analyzer = OddsAnalyzer()
        self.market_trends = {}  # Almacena tendencias de mercado por partido
        
    def extract_odds_features(self, fixture_id: int) -> Dict[str, float]:
        """
        Extrae características de odds pre-partido para usar como features en los modelos.
        
        Args:
            fixture_id: ID del partido
            
        Returns:
            Diccionario con características extraídas de las odds
        """
        features = {}
        
        try:
            # Obtener datos de odds del analizador
            odds_data = self.odds_analyzer.get_fixture_odds(fixture_id)
            
            if not odds_data:
                logger.warning(f"No se pudieron obtener odds para el partido {fixture_id}")
                return self._get_default_odds_features()
                
            # Extraer odds para mercados principales
            best_match_odds = self.odds_analyzer.get_best_odds(odds_data, "Match Winner")
            best_ou_odds = self.odds_analyzer.get_best_odds(odds_data, "Over/Under")
            best_btts_odds = self.odds_analyzer.get_best_odds(odds_data, "Both Teams Score")
            
            # Calcular eficiencia del mercado
            market_metrics = self.odds_analyzer.calculate_market_efficiency(odds_data)
            
            # Extraer características de match winner
            if best_match_odds:
                if "home" in best_match_odds:
                    features["home_odds"] = best_match_odds["home"]["odds"]
                if "draw" in best_match_odds:
                    features["draw_odds"] = best_match_odds["draw"]["odds"]
                if "away" in best_match_odds:
                    features["away_odds"] = best_match_odds["away"]["odds"]
                    
                # Calcular implied probabilities (ajustadas por margen)
                if all(k in best_match_odds for k in ["home", "draw", "away"]):
                    implied_prob_sum = (1/best_match_odds["home"]["odds"] + 
                                       1/best_match_odds["draw"]["odds"] + 
                                       1/best_match_odds["away"]["odds"])
                    
                    features["implied_prob_home"] = (1/best_match_odds["home"]["odds"]) / implied_prob_sum
                    features["implied_prob_draw"] = (1/best_match_odds["draw"]["odds"]) / implied_prob_sum
                    features["implied_prob_away"] = (1/best_match_odds["away"]["odds"]) / implied_prob_sum
            
            # Extraer características de over/under
            if best_ou_odds:
                for key in best_ou_odds:
                    if "over 2.5" in key.lower():
                        features["over25_odds"] = best_ou_odds[key]["odds"]
                    if "under 2.5" in key.lower():
                        features["under25_odds"] = best_ou_odds[key]["odds"]
            
            # Extraer características de BTTS
            if best_btts_odds:
                if "yes" in best_btts_odds:
                    features["btts_yes_odds"] = best_btts_odds["yes"]["odds"]
                if "no" in best_btts_odds:
                    features["btts_no_odds"] = best_btts_odds["no"]["odds"]
            
            # Añadir métricas de mercado
            features["market_efficiency"] = market_metrics.get("efficiency", 0.95)
            features["market_margin"] = market_metrics.get("margin", 0.05)
            
            # Calcular intensidad de favorito
            if "home_odds" in features and "away_odds" in features:
                features["favorite_intensity"] = min(features["home_odds"], features["away_odds"]) / max(features["home_odds"], features["away_odds"])
                features["is_home_favorite"] = 1.0 if features["home_odds"] < features["away_odds"] else 0.0
            
            # Calcular volatilidad del mercado (si hay datos históricos)
            market_confidence = self.odds_analyzer.movement_tracker.get_market_confidence(fixture_id)
            features["market_confidence"] = market_confidence
            
            # Detectar movimientos significativos
            movement_data = self.analyze_market_movements(fixture_id)
            if movement_data["significant_movements"]:
                features["has_significant_movement"] = 1.0
                # Capturar dirección del movimiento para equipo local/visitante
                if "home_movement_direction" in movement_data:
                    features["home_movement"] = 1.0 if movement_data["home_movement_direction"] == "decreasing" else -1.0
                if "away_movement_direction" in movement_data:
                    features["away_movement"] = 1.0 if movement_data["away_movement_direction"] == "decreasing" else -1.0
            else:
                features["has_significant_movement"] = 0.0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extrayendo características de odds: {e}")
            return self._get_default_odds_features()
    
    def _get_default_odds_features(self) -> Dict[str, float]:
        """Retorna valores por defecto para características de odds."""
        return {
            "home_odds": 2.5,
            "draw_odds": 3.2,
            "away_odds": 2.9,
            "implied_prob_home": 0.36,
            "implied_prob_draw": 0.28,
            "implied_prob_away": 0.36,
            "over25_odds": 1.95,
            "under25_odds": 1.95,
            "btts_yes_odds": 1.9,
            "btts_no_odds": 1.9,
            "market_efficiency": 0.95,
            "market_margin": 0.05,
            "favorite_intensity": 0.8,
            "is_home_favorite": 0.5,
            "market_confidence": 0.5,
            "has_significant_movement": 0.0
        }
    
    def analyze_market_movements(self, fixture_id: int) -> Dict[str, Any]:
        """
        Analiza movimientos en el mercado para un partido determinado.
        
        Args:
            fixture_id: ID del partido
            
        Returns:
            Diccionario con análisis de movimientos
        """
        try:
            # Obtener análisis de movimientos del analizador
            basic_movement_data = self.odds_analyzer.detect_significant_market_movements(fixture_id)
            
            # Enriquecer con análisis adicional
            result = basic_movement_data.copy()
            
            if basic_movement_data["significant_movements"]:
                movements = basic_movement_data.get("movements", [])
                
                # Analizar movimientos de equipo local
                home_movements = [m for m in movements 
                                 if m["market"] == "Match Winner" and 
                                 (m["selection"].lower() == "home" or m["selection"] == "1")]
                
                # Analizar movimientos de equipo visitante
                away_movements = [m for m in movements 
                                 if m["market"] == "Match Winner" and 
                                 (m["selection"].lower() == "away" or m["selection"] == "2")]
                
                # Capturar dirección de movimiento para local y visitante
                if home_movements:
                    result["home_movement_direction"] = home_movements[0]["trend"]
                    result["home_movement_strength"] = abs(home_movements[0]["change"])
                    
                if away_movements:
                    result["away_movement_direction"] = away_movements[0]["trend"]
                    result["away_movement_strength"] = abs(away_movements[0]["change"])
                
                # Determinar si hay movimientos importantes de goles
                ou_movements = [m for m in movements if m["market"] == "Over/Under"]
                if ou_movements:
                    # Tendencia en expectativa de goles
                    over_movements = [m for m in ou_movements if "over" in m["selection"].lower()]
                    if over_movements:
                        over_trend = "increasing" if over_movements[0]["trend"] == "decreasing" else "decreasing"
                        result["goals_expectation_trend"] = over_trend
            
            # Almacenar para referencia futura
            self.market_trends[fixture_id] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error analizando movimientos de mercado: {e}")
            return {"fixture_id": fixture_id, "significant_movements": False, "error": str(e)}
    
    def calibrate_prediction_with_movements(self, prediction: Dict[str, Any], fixture_id: int) -> Dict[str, Any]:
        """
        Calibra la predicción base con información de movimientos del mercado.
        
        Args:
            prediction: Predicción original
            fixture_id: ID del partido
            
        Returns:
            Predicción calibrada con movimientos del mercado
        """
        try:
            # Obtener la predicción calibrada con odds del analizador
            calibrated = self.odds_analyzer.calibrate_prediction_with_market(prediction, fixture_id)
            
            # Analizar movimientos recientes
            movement_analysis = self.analyze_market_movements(fixture_id)
            
            # Si no hay movimientos significativos, retornar la calibración estándar
            if not movement_analysis["significant_movements"]:
                return calibrated
            
            # Obtener confianza de mercado
            market_confidence = self.odds_analyzer.movement_tracker.get_market_confidence(fixture_id)
            
            # Ajustar predicción según tendencias recientes
            adjusted = calibrated.copy()
            
            # Ajustar predicciones de resultado
            if "home_movement_direction" in movement_analysis:
                # Si las odds del local bajan (más favorito), aumentar probabilidad
                if movement_analysis["home_movement_direction"] == "decreasing":
                    strength = movement_analysis.get("home_movement_strength", 0.05)
                    # Limitar el ajuste basado en confianza de mercado
                    adjustment = min(strength * 0.5, 0.1) * market_confidence
                    
                    # Ajustar probabilidades
                    adjusted["prob_home_win"] = min(1.0, adjusted["prob_home_win"] + adjustment)
                    # Redistribuir el resto
                    remaining = 1.0 - adjusted["prob_home_win"]
                    total_other = calibrated["prob_draw"] + calibrated["prob_away_win"]
                    if total_other > 0:
                        adjusted["prob_draw"] = remaining * (calibrated["prob_draw"] / total_other)
                        adjusted["prob_away_win"] = remaining * (calibrated["prob_away_win"] / total_other)
                
                # Si las odds del local suben (menos favorito), disminuir probabilidad
                elif movement_analysis["home_movement_direction"] == "increasing":
                    strength = movement_analysis.get("home_movement_strength", 0.05)
                    # Limitar el ajuste basado en confianza de mercado
                    adjustment = min(strength * 0.5, 0.1) * market_confidence
                    
                    # Ajustar probabilidades
                    adjusted["prob_home_win"] = max(0.05, adjusted["prob_home_win"] - adjustment)
                    # Redistribuir el resto
                    remaining = 1.0 - adjusted["prob_home_win"]
                    total_other = calibrated["prob_draw"] + calibrated["prob_away_win"]
                    if total_other > 0:
                        adjusted["prob_draw"] = remaining * (calibrated["prob_draw"] / total_other)
                        adjusted["prob_away_win"] = remaining * (calibrated["prob_away_win"] / total_other)
            
            # Similar ajuste para el equipo visitante
            if "away_movement_direction" in movement_analysis:
                # Si las odds del visitante bajan (más favorito), aumentar probabilidad
                if movement_analysis["away_movement_direction"] == "decreasing":
                    strength = movement_analysis.get("away_movement_strength", 0.05)
                    # Limitar el ajuste basado en confianza de mercado
                    adjustment = min(strength * 0.5, 0.1) * market_confidence
                    
                    # Ajustar probabilidades
                    adjusted["prob_away_win"] = min(1.0, adjusted["prob_away_win"] + adjustment)
                    # Redistribuir el resto
                    remaining = 1.0 - adjusted["prob_away_win"]
                    total_other = calibrated["prob_draw"] + calibrated["prob_home_win"]
                    if total_other > 0:
                        adjusted["prob_draw"] = remaining * (calibrated["prob_draw"] / total_other)
                        adjusted["prob_home_win"] = remaining * (calibrated["prob_home_win"] / total_other)
                
                # Si las odds del visitante suben (menos favorito), disminuir probabilidad
                elif movement_analysis["away_movement_direction"] == "increasing":
                    strength = movement_analysis.get("away_movement_strength", 0.05)
                    # Limitar el ajuste basado en confianza de mercado
                    adjustment = min(strength * 0.5, 0.1) * market_confidence
                    
                    # Ajustar probabilidades
                    adjusted["prob_away_win"] = max(0.05, adjusted["prob_away_win"] - adjustment)
                    # Redistribuir el resto
                    remaining = 1.0 - adjusted["prob_away_win"]
                    total_other = calibrated["prob_draw"] + calibrated["prob_home_win"]
                    if total_other > 0:
                        adjusted["prob_draw"] = remaining * (calibrated["prob_draw"] / total_other)
                        adjusted["prob_home_win"] = remaining * (calibrated["prob_home_win"] / total_other)
            
            # Ajustar expectativa de goles si hay tendencia clara
            if "goals_expectation_trend" in movement_analysis:
                if movement_analysis["goals_expectation_trend"] == "increasing":
                    # Aumentar expectativa de goles
                    adjusted["expected_goals"] = calibrated["expected_goals"] * 1.1
                    adjusted["prob_over_2_5"] = min(0.95, calibrated["prob_over_2_5"] * 1.15)
                    adjusted["prob_under_2_5"] = 1.0 - adjusted["prob_over_2_5"]
                elif movement_analysis["goals_expectation_trend"] == "decreasing":
                    # Disminuir expectativa de goles
                    adjusted["expected_goals"] = calibrated["expected_goals"] * 0.9
                    adjusted["prob_over_2_5"] = max(0.05, calibrated["prob_over_2_5"] * 0.85)
                    adjusted["prob_under_2_5"] = 1.0 - adjusted["prob_over_2_5"]
            
            # Añadir metadatos sobre el ajuste
            if "market_calibration" not in adjusted:
                adjusted["market_calibration"] = {}
                
            adjusted["market_calibration"]["movement_adjustment"] = {
                "applied": True,
                "significant_movements": movement_analysis["significant_movements"],
                "market_confidence": market_confidence,
                "original_calibrated": {
                    "prob_home_win": calibrated["prob_home_win"],
                    "prob_draw": calibrated["prob_draw"],
                    "prob_away_win": calibrated["prob_away_win"],
                    "expected_goals": calibrated.get("expected_goals", 0)
                }
            }
            
            return adjusted
            
        except Exception as e:
            logger.error(f"Error calibrando predicción con movimientos: {e}")
            return prediction
    
    def enrich_prediction_with_market_data(self, prediction: Dict[str, Any], fixture_id: int) -> Dict[str, Any]:
        """
        Enriquece la predicción con datos de mercado completos.
        
        Args:
            prediction: Predicción original
            fixture_id: ID del partido
            
        Returns:
            Predicción enriquecida con datos de mercado
        """
        try:
            # Paso 1: Obtener datos de odds
            odds_data = self.odds_analyzer.get_fixture_odds(fixture_id)
            if not odds_data:
                logger.warning(f"No se pudieron obtener odds para fixture {fixture_id}")
                return prediction
            
            # Paso 2: Realizar calibración con movimientos
            calibrated = self.calibrate_prediction_with_movements(prediction, fixture_id)
            
            # Paso 3: Detectar oportunidades de valor
            value_opportunities = self.odds_analyzer.get_value_opportunities(fixture_id, calibrated)
            
            # Paso 4: Obtener análisis de movimientos
            movement_summary = self.odds_analyzer.get_odds_movement_analysis(fixture_id)
              # Paso 5: Enriquecer predicción con toda la información
            enriched = calibrated.copy()
              # Añadir datos de mercado con explicaciones detalladas
            
            # Obtener métricas de mercado
            efficiency = value_opportunities.get("market_analysis", {}).get("efficiency", 0) if value_opportunities else 0
            margin = value_opportunities.get("market_analysis", {}).get("margin", 0) if value_opportunities else 0
            confidence = movement_summary.get("market_confidence", 0) if movement_summary else 0
            
            # Generar explicaciones basadas en los valores
            efficiency_explanation = self._generate_efficiency_explanation(efficiency)
            margin_explanation = self._generate_margin_explanation(margin)
            confidence_explanation = self._generate_confidence_explanation(confidence)
            liquidity_explanation = self._generate_liquidity_explanation(odds_data)
            
            # Analizar movimientos de odds
            movement_detected = movement_summary.get("movement_detected", False) if movement_summary else False
            significant_markets = [m.get("market") for m in movement_summary.get("markets", {}).values()] if movement_summary else []
            movements_explanation = self._generate_movements_explanation(movement_detected, significant_markets, movement_summary)
            
            enriched["market_data"] = {
                "analysis": {
                    "efficiency": efficiency,
                    "margin": margin,
                    "confidence": confidence,
                    "explanations": {
                        "efficiency": efficiency_explanation,
                        "margin": margin_explanation,
                        "confidence": confidence_explanation,
                        "liquidity": liquidity_explanation
                    }
                },
                "movements": {
                    "detected": movement_detected,
                    "significant_markets": significant_markets,
                    "explanation": movements_explanation
                }
            }
            
            # Añadir oportunidades de valor (si existen)
            if value_opportunities and len(value_opportunities) > 1:  # Más que solo market_analysis
                enriched["value_opportunities"] = {}
                
                # Añadir oportunidades por mercado
                for market, opps in value_opportunities.items():
                    if market != "market_analysis":
                        enriched["value_opportunities"][market] = opps
            
            # Añadir expectativa de retorno
            if "implied_prob_home" in enriched and "home_odds" in enriched:
                implied_prob = enriched["implied_prob_home"]
                model_prob = enriched["prob_home_win"]
                if model_prob > implied_prob:
                    enriched["home_expected_value"] = (model_prob * enriched["home_odds"]) - 1
                else:
                    enriched["home_expected_value"] = 0
                    
            if "implied_prob_away" in enriched and "away_odds" in enriched:
                implied_prob = enriched["implied_prob_away"]
                model_prob = enriched["prob_away_win"]
                if model_prob > implied_prob:
                    enriched["away_expected_value"] = (model_prob * enriched["away_odds"]) - 1
                else:
                    enriched["away_expected_value"] = 0
            
            return enriched
            
        except Exception as e:
            logger.error(f"Error enriqueciendo predicción con datos de mercado: {e}")
            return prediction

    def analyze_market_data(self, odds_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze market data to provide detailed insights and explanations

        Args:
            odds_data: Dictionary containing odds and market data

        Returns:
            Dictionary containing market analysis results
        """
        try:
            if not odds_data:
                return self._get_default_analysis()

            # Get market metrics using existing functionality
            market_metrics = self.odds_analyzer.calculate_market_efficiency(odds_data)
              # Calculate efficiency and confidence
            efficiency_score = market_metrics.get("efficiency", 0.8)
            fixture_id = odds_data.get("fixture_id")
            market_confidence = self.odds_analyzer.movement_tracker.get_market_confidence(fixture_id) if fixture_id is not None else 0.5
            
            # Analyze market liquidity
            liquidity_score = self._analyze_market_liquidity(odds_data)
            
            # Get significant movements
            movement_analysis = self._analyze_odds_movements(odds_data)
            
            # Find value opportunities by comparing model probabilities with market odds
            value_opps = self._find_value_opportunities(odds_data)

            return {
                "efficiency_score": efficiency_score,
                "liquidity_score": liquidity_score,
                "market_confidence": market_confidence,
                "significant_moves": movement_analysis.get("significant_moves", []),
                "movement_implications": movement_analysis.get("implications", []),
                "value_opportunities": value_opps
            }
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            return self._get_default_analysis()

    def _analyze_market_liquidity(self, odds_data: Dict[str, Any]) -> float:
        """Analyze market liquidity based on number of bookmakers and bet volumes"""
        try:
            # Get number of bookmakers offering odds
            odds_by_market = self.odds_analyzer.get_all_fixture_odds(odds_data.get("fixture_id", 0))
            num_bookmakers = len(odds_by_market.get("bookmakers", []))
            max_bookmakers = 50  # Typical maximum number of bookmakers
            
            # Scale 0-1 based on number of bookmakers and market depth
            liquidity = min(1.0, num_bookmakers / max_bookmakers)
            
            # Adjust based on betting volumes if available
            volumes = odds_data.get("betting_volumes", {})
            if volumes:
                volume_score = min(1.0, sum(volumes.values()) / ODDS_CONFIG.get("high_volume_threshold", 1000000))
                liquidity = (liquidity + volume_score) / 2
            
            return round(liquidity, 3)
        except Exception:
            return 0.7

    def _analyze_odds_movements(self, odds_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze odds movements and their implications"""
        try:
            if not odds_data.get("fixture_id"):
                return {
                    "significant_moves": [],
                    "implications": ["No fixture ID provided for odds movement analysis"]
                }

            # Get odds movement history from the tracker
            movements = self.odds_analyzer.movement_tracker.get_odds_movements(odds_data["fixture_id"])
            if not movements:
                return {
                    "significant_moves": [],
                    "implications": ["No significant odds movements detected"]
                }

            significant_moves = []
            implications = []

            # Process significant moves
            for move in movements:
                if self._is_significant_move(move):
                    significant_moves.append(move)
                    implications.append(self._interpret_movement(move))

            return {
                "significant_moves": significant_moves,
                "implications": implications or ["No significant odds movements detected"]
            }
        except Exception:
            return {
                "significant_moves": [],
                "implications": ["Error analyzing odds movements"]            }
    
    def _find_value_opportunities(self, odds_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify potential value betting opportunities"""
        try:            # Get best available odds for main markets
            best_match_odds = self.odds_analyzer.get_best_odds(odds_data, "Match Winner") or {}
            best_ou_odds = self.odds_analyzer.get_best_odds(odds_data, "Over/Under") or {}
            best_btts_odds = self.odds_analyzer.get_best_odds(odds_data, "Both Teams Score") or {}
            
            # Compare with model probabilities
            model_probs = odds_data.get("model_probabilities", {}) if odds_data.get("model_probabilities") is not None else {}
            
            value_opps = {}
            
            # Check match winner market
            for outcome in ["home", "draw", "away"]:
                if best_match_odds and outcome in best_match_odds:
                    market_prob = 1 / best_match_odds[outcome]["odds"]
                    model_prob = model_probs.get(outcome, 0)
                    
                    if model_prob > market_prob:
                        value_opps[f"{outcome}_win"] = {
                            "edge": round((model_prob - market_prob) * 100, 1),
                            "confidence": self._calculate_edge_confidence(model_prob, market_prob)
                        }
              # Check goals markets
            if best_ou_odds and "over 2.5" in best_ou_odds and model_probs and "model_over25_prob" in model_probs:
                market_prob = 1 / best_ou_odds["over 2.5"]["odds"]
                model_prob = model_probs.get("model_over25_prob", 0)
                
                if model_prob > market_prob:
                    value_opps["over25"] = {
                        "edge": round((model_prob - market_prob) * 100, 1),
                        "confidence": self._calculate_edge_confidence(model_prob, market_prob)
                    }
            
            return value_opps
        except Exception:
            return {}

    def _is_significant_move(self, move: Dict[str, Any]) -> bool:
        """Determine if an odds movement is significant"""
        try:
            threshold = ODDS_CONFIG.get("significant_move_threshold", 0.1)
            return abs(move.get("probability_change", 0)) > threshold
        except Exception:
            return False

    def _interpret_movement(self, move: Dict[str, Any]) -> str:
        """Interpret the meaning of an odds movement"""
        try:
            direction = "increase" if move.get("probability_change", 0) > 0 else "decrease"
            magnitude = abs(move.get("probability_change", 0))
            outcome = move.get("outcome", "Unknown")
            
            # Add context based on timing and magnitude
            timing = "Early" if move.get("is_early_move", False) else "Late"
            strength = "Strong" if magnitude > 0.15 else "Moderate"
            
            return f"{timing} {strength} {direction} in probability for {outcome} outcome"
        except Exception:
            return "Unspecified odds movement"

    def _calculate_edge_confidence(self, model_prob: float, market_prob: float) -> float:
        """Calculate confidence in the identified edge"""
        try:
            edge = abs(model_prob - market_prob)
            base_confidence = min(0.9, edge * 3)
            
            # Adjust confidence based on market efficiency
            market_efficiency = self.odds_analyzer.movement_tracker.get_market_confidence(0)  # 0 as default fixture_id
            
            # Lower confidence when market is highly efficient
            adjusted_confidence = base_confidence * (1 - market_efficiency * 0.3)
            
            return round(adjusted_confidence, 2)
        except Exception:
            return 0.5

    def _get_default_analysis(self) -> Dict[str, Any]:
        """Return default market analysis when data is unavailable"""
        return {
            "efficiency_score": 0.8,
            "liquidity_score": 0.7,
            "market_confidence": 0.75,
            "significant_moves": [],
            "movement_implications": ["Insufficient market data available"],
            "value_opportunities": {}
        }

    def _generate_efficiency_explanation(self, efficiency: float) -> str:
        """Generate explanation about market efficiency"""
        if efficiency > 0.95:
            return "The market is highly efficient with minimal margins, suggesting highly accurate pricing"
        elif efficiency > 0.90:
            return "The market shows good efficiency with reasonable margins"
        elif efficiency > 0.85:
            return "The market has moderate efficiency with higher margins than optimal"
        else:
            return "The market shows low efficiency with high margins, suggesting potential pricing inaccuracies"
    
    def _generate_margin_explanation(self, margin: float) -> str:
        """Generate explanation about bookmaker margins"""
        if margin < 0.03:
            return "Extremely competitive margins indicating a mature, liquid market"
        elif margin < 0.05:
            return "Low margins suggesting strong competition among bookmakers"
        elif margin < 0.08:
            return "Average margins for a typical football match"
        else:
            return "High margins indicating lower market confidence or limited competition"
    
    def _generate_confidence_explanation(self, confidence: float) -> str:
        """Generate explanation about market confidence"""
        if confidence > 0.8:
            return "High market confidence with stable odds and minimal fluctuation"
        elif confidence > 0.6:
            return "Good market confidence with typical odds movement patterns"
        elif confidence > 0.4:
            return "Moderate market confidence with some notable odds movements"
        else:
            return "Low market confidence with significant odds volatility"
    
    def _generate_liquidity_explanation(self, odds_data: Dict[str, Any]) -> str:
        """Generate explanation about market liquidity"""
        # Count number of bookmakers
        bookmakers = odds_data.get('bookmakers', [])
        count = len(bookmakers)
        
        if count > 30:
            return "Extremely liquid market with extensive bookmaker coverage"
        elif count > 20:
            return "Highly liquid market with many bookmakers offering odds"
        elif count > 10:
            return "Moderately liquid market with sufficient bookmaker coverage"
        else:
            return "Limited liquidity with few bookmakers offering odds"
    
    def _generate_movements_explanation(self, movement_detected: bool, 
                                     significant_markets: List, 
                                     movement_summary: Dict[str, Any]) -> str:
        """Generate explanation about odds movements"""
        if not movement_detected:
            return "No significant odds movements detected, suggesting a stable market"
        
        if not significant_markets:
            return "Minor odds movements detected, but nothing significant"
        
        # Generate explanation based on markets with movements
        explanations = []
        
        if "Match Winner" in significant_markets:
            explanations.append("Significant movement in match winner odds")
        
        if "Over/Under" in significant_markets:
            explanations.append("Notable shifts in goals over/under lines")
        
        if "Both Teams Score" in significant_markets:
            explanations.append("Changes in both teams to score market")
        
        # Add timing context if available
        confidence = movement_summary.get("market_confidence", 0) if movement_summary else 0
        if confidence > 0.7:
            explanations.append("Market appears to be settling with high confidence")
        elif confidence < 0.4:
            explanations.append("Continued volatility suggests uncertainty")
        
        if not explanations:
            return "Some odds movements detected across markets"
        
        return " and ".join(explanations)
        
def integrate_market_features(features: Dict[str, Any], fixture_id: int) -> Dict[str, Any]:
    """
    Función de utilidad para integrar características de mercado en un set de features existente.
    
    Args:
        features: Diccionario de características existente
        fixture_id: ID del partido
        
    Returns:
        Características enriquecidas con datos de mercado
    """
    try:
        integrator = MarketDataIntegrator()
        market_features = integrator.extract_odds_features(fixture_id)
        
        # Combinar características
        enriched_features = features.copy()
        enriched_features.update(market_features)
        
        return enriched_features
        
    except Exception as e:
        logger.error(f"Error integrando características de mercado: {e}")
        return features

def create_market_monitor(fixture_ids: List[int]) -> Dict[int, Dict[str, Any]]:
    """
    Crea un monitor para movimientos significativos en múltiples partidos.
    
    Args:
        fixture_ids: Lista de IDs de partidos a monitorear
        
    Returns:
        Diccionario con información de movimientos por partido
    """
    monitor_results = {}
    integrator = MarketDataIntegrator()
    
    for fixture_id in fixture_ids:
        try:
            # Analizar movimientos
            movement_data = integrator.analyze_market_movements(fixture_id)
            
            # Si hay movimientos significativos, añadir al monitor
            if movement_data["significant_movements"]:
                monitor_results[fixture_id] = {
                    "significant_movements": True,
                    "movements": movement_data.get("movements", []),
                    "implications": movement_data.get("implications", []),
                    "last_updated": datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error monitoreando fixture {fixture_id}: {e}")
    
    return monitor_results
