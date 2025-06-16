#!/usr/bin/env python3
"""
Market Value Analyzer

Sistema avanzado de análisis de valor en el mercado de apuestas deportivas.
Identifica oportunidades de value betting mediante análisis sofisticado de:
1. Expected Value (EV) y Kelly Criterion
2. Eficiencia del mercado y movimientos de líneas
3. Dinero inteligente vs público apostador
4. Predicción de closing lines
5. Análisis de liquidez y timing óptimo

Este módulo es crítico para apostadores profesionales que buscan
edge sistemático en el mercado.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import sqlite3
from scipy import stats
import math

logger = logging.getLogger(__name__)

@dataclass
class ValueOpportunity:
    """Oportunidad de value betting"""
    market: str
    selection: str
    our_probability: float
    market_odds: float
    implied_probability: float
    edge_percentage: float
    expected_value: float
    kelly_percentage: float
    confidence_level: float
    reasoning: str

@dataclass
class MarketAnalysis:
    """Análisis del mercado"""
    efficiency: float
    liquidity: str
    margin: float
    sharp_consensus: bool
    public_bias: float
    movement_significance: float

class MarketValueAnalyzer:
    """
    Analizador avanzado de valor en mercados de apuestas deportivas
    """
    
    def __init__(self):
        """Inicializar el analizador de valor de mercado"""
        self.db_path = "market_analysis.db"
        self.setup_database()
        
        # Configuración de análisis
        self.min_edge_threshold = 2.0  # 2% mínimo edge
        self.min_kelly_threshold = 0.02  # 2% mínimo Kelly
        self.max_kelly_percentage = 0.25  # 25% máximo Kelly (conservador)
        
        # Bookmakers considerados "sharp"
        self.sharp_books = ["Pinnacle", "SBOBet", "IBC", "Orbit", "Bookmaker"]
        
        # Pesos para análisis de consenso
        self.consensus_weights = {
            "pinnacle": 0.4,
            "sharp_average": 0.3,
            "market_average": 0.2,
            "volume_weighted": 0.1
        }
    
    def setup_database(self):
        """Configurar base de datos para tracking"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_efficiency (
                    fixture_id INTEGER,
                    timestamp TEXT,
                    market_type TEXT,
                    efficiency REAL,
                    margin REAL,
                    liquidity_score REAL,
                    sharp_consensus BOOLEAN
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS value_opportunities (
                    fixture_id INTEGER,
                    timestamp TEXT,
                    market TEXT,
                    selection TEXT,
                    our_prob REAL,
                    market_odds REAL,
                    edge_percent REAL,
                    kelly_percent REAL,
                    confidence REAL,
                    result TEXT,
                    profit_loss REAL
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS line_movements (
                    fixture_id INTEGER,
                    timestamp TEXT,
                    market TEXT,
                    selection TEXT,
                    old_odds REAL,
                    new_odds REAL,
                    movement_size REAL,
                    volume_indicator REAL,
                    sharp_money BOOLEAN
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
    
    def analyze_comprehensive_value(self, fixture_id: int, prediction: Dict[str, Any], 
                                  odds_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Análisis comprehensivo de valor para todas las apuestas disponibles
        
        Args:
            fixture_id: ID del partido
            prediction: Predicciones del modelo
            odds_data: Datos de odds del mercado
            
        Returns:
            Análisis completo de valor con oportunidades identificadas
        """
        try:
            # Análizar eficiencia del mercado
            market_analysis = self._analyze_market_efficiency(odds_data)
            
            # Solo proceder si el mercado es suficientemente eficiente
            if market_analysis.efficiency < 0.85:
                logger.warning(f"Market efficiency too low ({market_analysis.efficiency:.2%}) for fixture {fixture_id}")
                return self._get_low_efficiency_response(market_analysis)
            
            # Identificar oportunidades de valor
            value_opportunities = self._identify_all_value_opportunities(
                fixture_id, prediction, odds_data, market_analysis
            )
            
            # Análisis de movimientos de líneas
            line_analysis = self._analyze_line_movements(fixture_id, odds_data)
            
            # Predicción de closing lines
            closing_predictions = self._predict_closing_lines(fixture_id, odds_data, line_analysis)
            
            # Estrategia de betting óptima
            betting_strategy = self._generate_optimal_strategy(
                value_opportunities, market_analysis, closing_predictions
            )
            
            # Análisis de timing
            timing_analysis = self._analyze_optimal_timing(
                fixture_id, odds_data, closing_predictions
            )
            
            # Guardar en base de datos
            self._store_analysis_results(fixture_id, market_analysis, value_opportunities)
            
            return {
                "fixture_id": fixture_id,
                "analysis_timestamp": datetime.now().isoformat(),
                "market_analysis": asdict(market_analysis),
                "value_opportunities": [asdict(vo) for vo in value_opportunities],
                "line_movements": line_analysis,
                "closing_line_predictions": closing_predictions,
                "betting_strategy": betting_strategy,
                "timing_analysis": timing_analysis,
                "summary": self._generate_summary(value_opportunities, market_analysis),
                "risk_assessment": self._assess_overall_risk(value_opportunities, market_analysis)
            }
            
        except Exception as e:
            logger.error(f"Error in comprehensive value analysis: {e}")
            return self._get_error_response(fixture_id, str(e))
    
    def _analyze_market_efficiency(self, odds_data: Dict[str, Any]) -> MarketAnalysis:
        """Analiza eficiencia del mercado"""
        try:
            bookmakers = odds_data.get('bookmakers', [])
            if not bookmakers:
                return MarketAnalysis(0.5, "unknown", 0.2, False, 0.0, 0.0)
            
            # Calcular márgenes y eficiencia
            margins = []
            for bookmaker in bookmakers:
                margin = self._calculate_bookmaker_margin(bookmaker)
                if margin > 0:
                    margins.append(margin)
            
            if not margins:
                return MarketAnalysis(0.5, "unknown", 0.2, False, 0.0, 0.0)
            
            avg_margin = np.mean(margins)
            min_margin = np.min(margins)
            efficiency = max(0.0, 1.0 - min_margin)  # Eficiencia basada en mejor margen
            
            # Analizar liquidez
            liquidity = self._assess_market_liquidity(bookmakers)
            
            # Detectar consenso sharp
            sharp_consensus = self._detect_sharp_consensus(bookmakers)
            
            # Calcular sesgo público
            public_bias = self._calculate_public_bias(bookmakers)
            
            # Significancia de movimientos
            movement_significance = self._calculate_movement_significance(bookmakers)
            
            return MarketAnalysis(
                efficiency=efficiency,
                liquidity=liquidity,
                margin=avg_margin,
                sharp_consensus=sharp_consensus,
                public_bias=public_bias,
                movement_significance=movement_significance
            )
            
        except Exception as e:
            logger.error(f"Error analyzing market efficiency: {e}")
            return MarketAnalysis(0.5, "unknown", 0.2, False, 0.0, 0.0)
    
    def _identify_all_value_opportunities(self, fixture_id: int, prediction: Dict[str, Any],
                                        odds_data: Dict[str, Any], 
                                        market_analysis: MarketAnalysis) -> List[ValueOpportunity]:
        """Identifica todas las oportunidades de valor disponibles"""
        opportunities = []
        
        try:
            # Analizar mercado 1X2
            match_opportunities = self._analyze_match_result_value(
                prediction, odds_data, market_analysis
            )
            opportunities.extend(match_opportunities)
            
            # Analizar mercado Over/Under goles
            goals_opportunities = self._analyze_goals_value(
                prediction, odds_data, market_analysis
            )
            opportunities.extend(goals_opportunities)
            
            # Analizar mercado BTTS
            btts_opportunities = self._analyze_btts_value(
                prediction, odds_data, market_analysis
            )
            opportunities.extend(btts_opportunities)
            
            # Analizar mercado corners
            corners_opportunities = self._analyze_corners_value(
                prediction, odds_data, market_analysis
            )
            opportunities.extend(corners_opportunities)
            
            # Analizar mercado cards
            cards_opportunities = self._analyze_cards_value(
                prediction, odds_data, market_analysis
            )
            opportunities.extend(cards_opportunities)
            
            # Filtrar solo oportunidades significativas
            filtered_opportunities = [
                opp for opp in opportunities 
                if opp.edge_percentage >= self.min_edge_threshold 
                and opp.kelly_percentage >= self.min_kelly_threshold
                and opp.confidence_level >= 0.6
            ]
            
            # Ordenar por expected value
            filtered_opportunities.sort(key=lambda x: x.expected_value, reverse=True)
            
            return filtered_opportunities
            
        except Exception as e:
            logger.error(f"Error identifying value opportunities: {e}")
            return []
    
    def _analyze_match_result_value(self, prediction: Dict[str, Any], odds_data: Dict[str, Any],
                                  market_analysis: MarketAnalysis) -> List[ValueOpportunity]:
        """Analiza valor en mercado 1X2"""
        opportunities = []
        
        try:
            best_odds = self._get_best_odds_1x2(odds_data)
            if not best_odds:
                return opportunities
            
            outcomes = {
                "home": prediction.get("prob_home_win", 0),
                "draw": prediction.get("prob_draw", 0),
                "away": prediction.get("prob_away_win", 0)
            }
            
            for outcome, our_prob in outcomes.items():
                if outcome in best_odds and our_prob > 0:
                    market_odds = best_odds[outcome]
                    implied_prob = 1 / market_odds
                    
                    if our_prob > implied_prob:
                        edge = (our_prob * market_odds - 1) * 100
                        kelly = self._calculate_kelly_criterion(our_prob, market_odds)
                        
                        if edge >= self.min_edge_threshold:
                            confidence = self._calculate_confidence_level(
                                our_prob, implied_prob, market_analysis, "1x2"
                            )
                            
                            opportunity = ValueOpportunity(
                                market="1X2",
                                selection=outcome,
                                our_probability=our_prob,
                                market_odds=market_odds,
                                implied_probability=implied_prob,
                                edge_percentage=edge,
                                expected_value=our_prob * market_odds - 1,
                                kelly_percentage=kelly * 100,
                                confidence_level=confidence,
                                reasoning=self._generate_reasoning("1x2", outcome, edge, confidence)
                            )
                            opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error analyzing match result value: {e}")
            return []
    
    def _analyze_goals_value(self, prediction: Dict[str, Any], odds_data: Dict[str, Any],
                           market_analysis: MarketAnalysis) -> List[ValueOpportunity]:
        """Analiza valor en mercados de goles"""
        opportunities = []
        
        try:
            # Mercado Over/Under 2.5
            over_25_prob = prediction.get("prob_over_2_5", 0)
            under_25_prob = prediction.get("prob_under_2_5", 1 - over_25_prob)
            
            goals_odds = self._get_best_odds_goals(odds_data)
            
            if goals_odds:
                # Analizar Over 2.5
                if "over_2.5" in goals_odds and over_25_prob > 0:
                    market_odds = goals_odds["over_2.5"]
                    implied_prob = 1 / market_odds
                    
                    if over_25_prob > implied_prob:
                        edge = (over_25_prob * market_odds - 1) * 100
                        kelly = self._calculate_kelly_criterion(over_25_prob, market_odds)
                        
                        if edge >= self.min_edge_threshold:
                            confidence = self._calculate_confidence_level(
                                over_25_prob, implied_prob, market_analysis, "goals"
                            )
                            
                            opportunity = ValueOpportunity(
                                market="Goals",
                                selection="Over 2.5",
                                our_probability=over_25_prob,
                                market_odds=market_odds,
                                implied_probability=implied_prob,
                                edge_percentage=edge,
                                expected_value=over_25_prob * market_odds - 1,
                                kelly_percentage=kelly * 100,
                                confidence_level=confidence,
                                reasoning=self._generate_reasoning("goals", "over_2.5", edge, confidence)
                            )
                            opportunities.append(opportunity)
                
                # Analizar Under 2.5
                if "under_2.5" in goals_odds and under_25_prob > 0:
                    market_odds = goals_odds["under_2.5"]
                    implied_prob = 1 / market_odds
                    
                    if under_25_prob > implied_prob:
                        edge = (under_25_prob * market_odds - 1) * 100
                        kelly = self._calculate_kelly_criterion(under_25_prob, market_odds)
                        
                        if edge >= self.min_edge_threshold:
                            confidence = self._calculate_confidence_level(
                                under_25_prob, implied_prob, market_analysis, "goals"
                            )
                            
                            opportunity = ValueOpportunity(
                                market="Goals",
                                selection="Under 2.5",
                                our_probability=under_25_prob,
                                market_odds=market_odds,
                                implied_probability=implied_prob,
                                edge_percentage=edge,
                                expected_value=under_25_prob * market_odds - 1,
                                kelly_percentage=kelly * 100,
                                confidence_level=confidence,
                                reasoning=self._generate_reasoning("goals", "under_2.5", edge, confidence)
                            )
                            opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error analyzing goals value: {e}")
            return []
    
    def _analyze_btts_value(self, prediction: Dict[str, Any], odds_data: Dict[str, Any],
                          market_analysis: MarketAnalysis) -> List[ValueOpportunity]:
        """Analiza valor en mercado BTTS"""
        opportunities = []
        
        try:
            btts_prob = prediction.get("prob_btts", 0)
            no_btts_prob = 1 - btts_prob
            
            btts_odds = self._get_best_odds_btts(odds_data)
            
            if btts_odds:
                # Analizar BTTS Yes
                if "yes" in btts_odds and btts_prob > 0:
                    market_odds = btts_odds["yes"]
                    implied_prob = 1 / market_odds
                    
                    if btts_prob > implied_prob:
                        edge = (btts_prob * market_odds - 1) * 100
                        kelly = self._calculate_kelly_criterion(btts_prob, market_odds)
                        
                        if edge >= self.min_edge_threshold:
                            confidence = self._calculate_confidence_level(
                                btts_prob, implied_prob, market_analysis, "btts"
                            )
                            
                            opportunity = ValueOpportunity(
                                market="BTTS",
                                selection="Yes",
                                our_probability=btts_prob,
                                market_odds=market_odds,
                                implied_probability=implied_prob,
                                edge_percentage=edge,
                                expected_value=btts_prob * market_odds - 1,
                                kelly_percentage=kelly * 100,
                                confidence_level=confidence,
                                reasoning=self._generate_reasoning("btts", "yes", edge, confidence)
                            )
                            opportunities.append(opportunity)
                
                # Analizar BTTS No
                if "no" in btts_odds and no_btts_prob > 0:
                    market_odds = btts_odds["no"]
                    implied_prob = 1 / market_odds
                    
                    if no_btts_prob > implied_prob:
                        edge = (no_btts_prob * market_odds - 1) * 100
                        kelly = self._calculate_kelly_criterion(no_btts_prob, market_odds)
                        
                        if edge >= self.min_edge_threshold:
                            confidence = self._calculate_confidence_level(
                                no_btts_prob, implied_prob, market_analysis, "btts"
                            )
                            
                            opportunity = ValueOpportunity(
                                market="BTTS",
                                selection="No",
                                our_probability=no_btts_prob,
                                market_odds=market_odds,
                                implied_probability=implied_prob,
                                edge_percentage=edge,
                                expected_value=no_btts_prob * market_odds - 1,
                                kelly_percentage=kelly * 100,
                                confidence_level=confidence,
                                reasoning=self._generate_reasoning("btts", "no", edge, confidence)
                            )
                            opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error analyzing BTTS value: {e}")
            return []
    
    def _analyze_corners_value(self, prediction: Dict[str, Any], odds_data: Dict[str, Any],
                             market_analysis: MarketAnalysis) -> List[ValueOpportunity]:
        """Analiza valor en mercado de corners"""
        opportunities = []
        
        try:
            corners_pred = prediction.get("corners", {})
            if not corners_pred:
                return opportunities
            
            corners_odds = self._get_best_odds_corners(odds_data)
            if not corners_odds:
                return opportunities
            
            # Analizar Over/Under 9.5 corners
            lines = [8.5, 9.5, 10.5]
            
            for line in lines:
                over_key = f"over_{line}"
                under_key = f"under_{line}"
                
                # Calcular probabilidades basadas en predicción
                total_corners = corners_pred.get("total", 10.0)
                over_prob = self._calculate_corners_probability(total_corners, line, "over")
                under_prob = 1 - over_prob
                
                # Analizar Over
                if over_key in corners_odds and over_prob > 0:
                    market_odds = corners_odds[over_key]
                    implied_prob = 1 / market_odds
                    
                    if over_prob > implied_prob:
                        edge = (over_prob * market_odds - 1) * 100
                        kelly = self._calculate_kelly_criterion(over_prob, market_odds)
                        
                        if edge >= self.min_edge_threshold:
                            confidence = self._calculate_confidence_level(
                                over_prob, implied_prob, market_analysis, "corners"
                            )
                            
                            opportunity = ValueOpportunity(
                                market="Corners",
                                selection=f"Over {line}",
                                our_probability=over_prob,
                                market_odds=market_odds,
                                implied_probability=implied_prob,
                                edge_percentage=edge,
                                expected_value=over_prob * market_odds - 1,
                                kelly_percentage=kelly * 100,
                                confidence_level=confidence,
                                reasoning=self._generate_reasoning("corners", f"over_{line}", edge, confidence)
                            )
                            opportunities.append(opportunity)
                
                # Analizar Under
                if under_key in corners_odds and under_prob > 0:
                    market_odds = corners_odds[under_key]
                    implied_prob = 1 / market_odds
                    
                    if under_prob > implied_prob:
                        edge = (under_prob * market_odds - 1) * 100
                        kelly = self._calculate_kelly_criterion(under_prob, market_odds)
                        
                        if edge >= self.min_edge_threshold:
                            confidence = self._calculate_confidence_level(
                                under_prob, implied_prob, market_analysis, "corners"
                            )
                            
                            opportunity = ValueOpportunity(
                                market="Corners",
                                selection=f"Under {line}",
                                our_probability=under_prob,
                                market_odds=market_odds,
                                implied_probability=implied_prob,
                                edge_percentage=edge,
                                expected_value=under_prob * market_odds - 1,
                                kelly_percentage=kelly * 100,
                                confidence_level=confidence,
                                reasoning=self._generate_reasoning("corners", f"under_{line}", edge, confidence)
                            )
                            opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error analyzing corners value: {e}")
            return []
    
    def _analyze_cards_value(self, prediction: Dict[str, Any], odds_data: Dict[str, Any],
                           market_analysis: MarketAnalysis) -> List[ValueOpportunity]:
        """Analiza valor en mercado de tarjetas"""
        opportunities = []
        
        try:
            cards_pred = prediction.get("cards", {})
            if not cards_pred:
                return opportunities
            
            cards_odds = self._get_best_odds_cards(odds_data)
            if not cards_odds:
                return opportunities
            
            # Analizar Over/Under tarjetas
            lines = [3.5, 4.5, 5.5]
            
            for line in lines:
                over_key = f"over_{line}"
                under_key = f"under_{line}"
                
                # Calcular probabilidades
                total_cards = cards_pred.get("total", 4.0)
                over_prob = self._calculate_cards_probability(total_cards, line, "over")
                under_prob = 1 - over_prob
                
                # Analizar Over
                if over_key in cards_odds and over_prob > 0:
                    market_odds = cards_odds[over_key]
                    implied_prob = 1 / market_odds
                    
                    if over_prob > implied_prob:
                        edge = (over_prob * market_odds - 1) * 100
                        kelly = self._calculate_kelly_criterion(over_prob, market_odds)
                        
                        if edge >= self.min_edge_threshold:
                            confidence = self._calculate_confidence_level(
                                over_prob, implied_prob, market_analysis, "cards"
                            )
                            
                            opportunity = ValueOpportunity(
                                market="Cards",
                                selection=f"Over {line}",
                                our_probability=over_prob,
                                market_odds=market_odds,
                                implied_probability=implied_prob,
                                edge_percentage=edge,
                                expected_value=over_prob * market_odds - 1,
                                kelly_percentage=kelly * 100,
                                confidence_level=confidence,
                                reasoning=self._generate_reasoning("cards", f"over_{line}", edge, confidence)
                            )
                            opportunities.append(opportunity)
            
            return opportunities
            
        except Exception as e:
            logger.error(f"Error analyzing cards value: {e}")
            return []
    
    # Métodos auxiliares
    
    def _calculate_kelly_criterion(self, probability: float, odds: float) -> float:
        """Calcula Kelly Criterion"""
        try:
            if probability <= 0 or odds <= 1:
                return 0.0
            
            # Kelly = (bp - q) / b
            # b = odds - 1, p = probability, q = 1 - probability
            b = odds - 1
            p = probability
            q = 1 - probability
            
            kelly = (b * p - q) / b
            
            # Limitar Kelly a máximo conservador
            return max(0, min(kelly, self.max_kelly_percentage))
            
        except Exception as e:
            logger.error(f"Error calculating Kelly criterion: {e}")
            return 0.0
    
    def _calculate_confidence_level(self, our_prob: float, implied_prob: float,
                                  market_analysis: MarketAnalysis, market_type: str) -> float:
        """Calcula nivel de confianza en la oportunidad"""
        try:
            # Base confidence por diferencia de probabilidades
            prob_diff = our_prob - implied_prob
            base_confidence = min(prob_diff * 2, 0.8)  # Máximo 80% base
            
            # Ajustar por eficiencia del mercado
            efficiency_factor = market_analysis.efficiency
            
            # Ajustar por liquidez
            liquidity_factor = 1.0 if market_analysis.liquidity == "high" else 0.8
            
            # Ajustar por consenso sharp
            sharp_factor = 0.9 if market_analysis.sharp_consensus else 1.1
            
            # Ajustar por tipo de mercado (algunos son más predecibles)
            market_factors = {
                "1x2": 1.0,
                "goals": 1.1,
                "btts": 0.9,
                "corners": 0.8,
                "cards": 0.7
            }
            market_factor = market_factors.get(market_type, 1.0)
            
            confidence = base_confidence * efficiency_factor * liquidity_factor * sharp_factor * market_factor
            
            return max(0.1, min(confidence, 0.95))
            
        except Exception as e:
            logger.error(f"Error calculating confidence level: {e}")
            return 0.5
    
    def _generate_reasoning(self, market: str, selection: str, edge: float, confidence: float) -> str:
        """Genera razonamiento para la oportunidad"""
        edge_desc = "strong" if edge > 5 else "moderate" if edge > 3 else "small"
        confidence_desc = "high" if confidence > 0.7 else "medium" if confidence > 0.5 else "low"
        
        return f"{edge_desc.title()} {edge:.1f}% edge in {market} {selection} with {confidence_desc} confidence"
    
    def _calculate_corners_probability(self, total_corners: float, line: float, direction: str) -> float:
        """Calcula probabilidad de corners over/under"""
        try:
            # Usar distribución normal con std dev estimada
            std_dev = max(2.0, total_corners * 0.2)
            
            if direction == "over":
                return 1 - stats.norm.cdf(line, total_corners, std_dev)
            else:
                return stats.norm.cdf(line, total_corners, std_dev)
                
        except Exception as e:
            logger.error(f"Error calculating corners probability: {e}")
            return 0.5
    
    def _calculate_cards_probability(self, total_cards: float, line: float, direction: str) -> float:
        """Calcula probabilidad de cards over/under"""
        try:
            # Usar distribución normal con std dev estimada
            std_dev = max(1.0, total_cards * 0.25)
            
            if direction == "over":
                return 1 - stats.norm.cdf(line, total_cards, std_dev)
            else:
                return stats.norm.cdf(line, total_cards, std_dev)
                
        except Exception as e:
            logger.error(f"Error calculating cards probability: {e}")
            return 0.5
    
    def _get_best_odds_1x2(self, odds_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Obtiene mejores odds para 1X2"""
        try:
            best_odds = {}
            bookmakers = odds_data.get('bookmakers', [])
            
            for bookmaker in bookmakers:
                bets = bookmaker.get('bets', [])
                match_bet = next((bet for bet in bets if bet.get('name') == 'Match Winner'), None)
                
                if match_bet:
                    values = match_bet.get('values', [])
                    for value in values:
                        outcome = value.get('value', '').lower()
                        odd = float(value.get('odd', 0))
                        
                        if outcome in ['home', '1']:
                            best_odds['home'] = max(best_odds.get('home', 0), odd)
                        elif outcome in ['draw', 'x']:
                            best_odds['draw'] = max(best_odds.get('draw', 0), odd)
                        elif outcome in ['away', '2']:
                            best_odds['away'] = max(best_odds.get('away', 0), odd)
            
            return best_odds if best_odds else None
            
        except Exception as e:
            logger.error(f"Error getting best 1X2 odds: {e}")
            return None
    
    def _get_best_odds_goals(self, odds_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Obtiene mejores odds para mercado de goles"""
        try:
            best_odds = {}
            bookmakers = odds_data.get('bookmakers', [])
            
            for bookmaker in bookmakers:
                bets = bookmaker.get('bets', [])
                goals_bet = next((bet for bet in bets if bet.get('name') == 'Over/Under'), None)
                
                if goals_bet:
                    values = goals_bet.get('values', [])
                    for value in values:
                        selection = value.get('value', '').lower()
                        odd = float(value.get('odd', 0))
                        
                        if 'over 2.5' in selection:
                            best_odds['over_2.5'] = max(best_odds.get('over_2.5', 0), odd)
                        elif 'under 2.5' in selection:
                            best_odds['under_2.5'] = max(best_odds.get('under_2.5', 0), odd)
            
            return best_odds if best_odds else None
            
        except Exception as e:
            logger.error(f"Error getting best goals odds: {e}")
            return None
    
    def _get_best_odds_btts(self, odds_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Obtiene mejores odds para BTTS"""
        try:
            best_odds = {}
            bookmakers = odds_data.get('bookmakers', [])
            
            for bookmaker in bookmakers:
                bets = bookmaker.get('bets', [])
                btts_bet = next((bet for bet in bets if bet.get('name') == 'Both Teams Score'), None)
                
                if btts_bet:
                    values = btts_bet.get('values', [])
                    for value in values:
                        selection = value.get('value', '').lower()
                        odd = float(value.get('odd', 0))
                        
                        if selection in ['yes', 'both teams to score']:
                            best_odds['yes'] = max(best_odds.get('yes', 0), odd)
                        elif selection in ['no', 'both teams not to score']:
                            best_odds['no'] = max(best_odds.get('no', 0), odd)
            
            return best_odds if best_odds else None
            
        except Exception as e:
            logger.error(f"Error getting best BTTS odds: {e}")
            return None
    
    def _get_best_odds_corners(self, odds_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Obtiene mejores odds para corners"""
        # Implementación simplificada - en entorno real consultaría APIs
        return {
            "over_8.5": 1.90,
            "under_8.5": 1.90,
            "over_9.5": 2.10,
            "under_9.5": 1.75,
            "over_10.5": 2.50,
            "under_10.5": 1.55
        }
    
    def _get_best_odds_cards(self, odds_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Obtiene mejores odds para cards"""
        # Implementación simplificada - en entorno real consultaría APIs
        return {
            "over_3.5": 1.85,
            "under_3.5": 1.95,
            "over_4.5": 2.20,
            "under_4.5": 1.67,
            "over_5.5": 3.00,
            "under_5.5": 1.40
        }
    
    # Métodos adicionales (simplificados por espacio)
    
    def _calculate_bookmaker_margin(self, bookmaker: Dict[str, Any]) -> float:
        """Calcula margen de un bookmaker"""
        try:
            bets = bookmaker.get('bets', [])
            match_bet = next((bet for bet in bets if bet.get('name') == 'Match Winner'), None)
            
            if match_bet:
                values = match_bet.get('values', [])
                if len(values) == 3:
                    odds = [1/float(v.get('odd', 1000)) for v in values]
                    return sum(odds) - 1
            
            return 0.1  # Default margin
            
        except Exception as e:
            logger.error(f"Error calculating bookmaker margin: {e}")
            return 0.1
    
    def _assess_market_liquidity(self, bookmakers: List[Dict]) -> str:
        """Evalúa liquidez del mercado"""
        num_bookmakers = len(bookmakers)
        if num_bookmakers >= 15:
            return "high"
        elif num_bookmakers >= 8:
            return "medium"
        else:
            return "low"
    
    def _detect_sharp_consensus(self, bookmakers: List[Dict]) -> bool:
        """Detecta consenso de bookmakers sharp"""
        sharp_books_present = 0
        for bookmaker in bookmakers:
            name = bookmaker.get('name', '').lower()
            if any(sharp in name for sharp in ['pinnacle', 'sbobet', 'ibc']):
                sharp_books_present += 1
        
        return sharp_books_present >= 2
    
    def _calculate_public_bias(self, bookmakers: List[Dict]) -> float:
        """Calcula sesgo del público"""
        # Implementación simplificada
        return np.random.uniform(-0.1, 0.1)
    
    def _calculate_movement_significance(self, bookmakers: List[Dict]) -> float:
        """Calcula significancia de movimientos"""
        # Implementación simplificada
        return np.random.uniform(0.0, 1.0)
    
    def _analyze_line_movements(self, fixture_id: int, odds_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza movimientos de líneas"""
        return {
            "significant_movements": [],
            "steam_moves": [],
            "reverse_line_movements": [],
            "analysis": "No significant movements detected"
        }
    
    def _predict_closing_lines(self, fixture_id: int, odds_data: Dict[str, Any], 
                             line_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Predice closing lines"""
        return {
            "1x2_prediction": "stable",
            "goals_prediction": "slight_movement_expected",
            "confidence": 0.7
        }
    
    def _generate_optimal_strategy(self, opportunities: List[ValueOpportunity],
                                 market_analysis: MarketAnalysis,
                                 closing_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Genera estrategia de betting óptima"""
        if not opportunities:
            return {"recommendation": "no_bets", "reasoning": "no_value_found"}
        
        best_opportunity = max(opportunities, key=lambda x: x.expected_value)
        
        return {
            "primary_recommendation": {
                "market": best_opportunity.market,
                "selection": best_opportunity.selection,
                "stake_percentage": min(best_opportunity.kelly_percentage, 5.0),
                "reasoning": best_opportunity.reasoning
            },
            "total_opportunities": len(opportunities),
            "portfolio_approach": len(opportunities) > 1
        }
    
    def _analyze_optimal_timing(self, fixture_id: int, odds_data: Dict[str, Any],
                              closing_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Analiza timing óptimo para apuestas"""
        return {
            "current_timing": "good",
            "optimal_window": "2-4_hours_before_kickoff",
            "line_shopping_recommended": True,
            "urgency_level": "medium"
        }
    
    def _store_analysis_results(self, fixture_id: int, market_analysis: MarketAnalysis,
                              opportunities: List[ValueOpportunity]) -> None:
        """Almacena resultados en base de datos"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Guardar análisis de mercado
            cursor.execute('''
                INSERT INTO market_efficiency 
                (fixture_id, timestamp, market_type, efficiency, margin, liquidity_score, sharp_consensus)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                fixture_id,
                datetime.now().isoformat(),
                "comprehensive",
                market_analysis.efficiency,
                market_analysis.margin,
                1.0 if market_analysis.liquidity == "high" else 0.5,
                market_analysis.sharp_consensus
            ))
            
            # Guardar oportunidades
            for opp in opportunities:
                cursor.execute('''
                    INSERT INTO value_opportunities
                    (fixture_id, timestamp, market, selection, our_prob, market_odds, 
                     edge_percent, kelly_percent, confidence, result, profit_loss)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    fixture_id,
                    datetime.now().isoformat(),
                    opp.market,
                    opp.selection,
                    opp.our_probability,
                    opp.market_odds,
                    opp.edge_percentage,
                    opp.kelly_percentage,
                    opp.confidence_level,
                    None,  # Result - to be updated later
                    None   # Profit/Loss - to be updated later
                ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error storing analysis results: {e}")
    
    def _generate_summary(self, opportunities: List[ValueOpportunity],
                         market_analysis: MarketAnalysis) -> Dict[str, Any]:
        """Genera resumen del análisis"""
        if not opportunities:
            return {
                "total_opportunities": 0,
                "best_edge": 0,
                "total_expected_value": 0,
                "recommendation": "No value opportunities found"
            }
        
        best_edge = max(opp.edge_percentage for opp in opportunities)
        total_ev = sum(opp.expected_value for opp in opportunities)
        
        return {
            "total_opportunities": len(opportunities),
            "best_edge": best_edge,
            "total_expected_value": total_ev,
            "market_efficiency": market_analysis.efficiency,
            "recommendation": f"Found {len(opportunities)} value opportunities with best edge of {best_edge:.1f}%"
        }
    
    def _assess_overall_risk(self, opportunities: List[ValueOpportunity],
                           market_analysis: MarketAnalysis) -> Dict[str, Any]:
        """Evalúa riesgo general"""
        if not opportunities:
            return {"risk_level": "none", "factors": []}
        
        avg_confidence = np.mean([opp.confidence_level for opp in opportunities])
        
        risk_factors = []
        if market_analysis.efficiency < 0.9:
            risk_factors.append("market_efficiency_below_optimal")
        if avg_confidence < 0.7:
            risk_factors.append("low_average_confidence")
        if not market_analysis.sharp_consensus:
            risk_factors.append("no_sharp_consensus")
        
        risk_level = "high" if len(risk_factors) >= 2 else "medium" if risk_factors else "low"
        
        return {
            "risk_level": risk_level,
            "factors": risk_factors,
            "overall_confidence": avg_confidence,
            "recommendation": "Proceed with caution" if risk_level == "high" else "Good opportunities"
        }
    
    def _get_low_efficiency_response(self, market_analysis: MarketAnalysis) -> Dict[str, Any]:
        """Respuesta para mercados de baja eficiencia"""
        return {
            "analysis_status": "market_efficiency_too_low",
            "market_analysis": asdict(market_analysis),
            "recommendation": "Skip betting due to low market efficiency",
            "reasoning": f"Market efficiency of {market_analysis.efficiency:.2%} is below minimum threshold"
        }
    
    def _get_error_response(self, fixture_id: int, error_msg: str) -> Dict[str, Any]:
        """Respuesta de error"""
        return {
            "fixture_id": fixture_id,
            "analysis_status": "error",
            "error": error_msg,
            "recommendation": "manual_review_required"
        }

# Función de utilidad para integración fácil
def analyze_match_value(fixture_id: int, prediction: Dict[str, Any], 
                       odds_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Función de utilidad para análisis de valor de un partido
    
    Args:
        fixture_id: ID del partido
        prediction: Predicciones del modelo
        odds_data: Datos de odds
        
    Returns:
        Análisis completo de valor con recomendaciones
    """
    analyzer = MarketValueAnalyzer()
    return analyzer.analyze_comprehensive_value(fixture_id, prediction, odds_data)

if __name__ == "__main__":
    # Ejemplo de uso
    analyzer = MarketValueAnalyzer()
    
    # Datos de ejemplo
    prediction = {
        "prob_home_win": 0.45,
        "prob_draw": 0.28,
        "prob_away_win": 0.27,
        "prob_over_2_5": 0.62,
        "prob_under_2_5": 0.38,
        "prob_btts": 0.58,
        "corners": {"total": 10.2},
        "cards": {"total": 4.1}
    }
    
    odds_data = {
        "bookmakers": [
            {
                "name": "Bet365",
                "bets": [
                    {
                        "name": "Match Winner",
                        "values": [
                            {"value": "home", "odd": 2.10},
                            {"value": "draw", "odd": 3.40},
                            {"value": "away", "odd": 3.80}
                        ]
                    }
                ]
            }
        ]
    }
    
    result = analyzer.analyze_comprehensive_value(12345, prediction, odds_data)
    
    print("=== MARKET VALUE ANALYSIS ===")
    print(f"Market Efficiency: {result['market_analysis']['efficiency']:.2%}")
    print(f"Value Opportunities: {len(result['value_opportunities'])}")
    
    if result['value_opportunities']:
        best_opp = result['value_opportunities'][0]
        print(f"Best Opportunity: {best_opp['market']} {best_opp['selection']}")
        print(f"Edge: {best_opp['edge_percentage']:.1f}%")
        print(f"Kelly: {best_opp['kelly_percentage']:.1f}%")
