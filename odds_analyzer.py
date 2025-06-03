import logging
from typing import Dict, Any, Optional, List
import requests
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
from config import ODDS_CONFIG, VALUE_CONFIG, API_FOOTBALL_KEY, API_BASE_URL
from odds_movement_tracker import OddsMovementTracker

logger = logging.getLogger(__name__)

class OddsAnalyzer:
    def __init__(self):
        """Initialize OddsAnalyzer with configuration"""
        self.api_key = API_FOOTBALL_KEY
        self.base_url = API_BASE_URL
        self.odds_config = ODDS_CONFIG
        self.value_config = VALUE_CONFIG
        self.movement_tracker = OddsMovementTracker()

    def get_fixture_odds(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        """Get odds data for a specific fixture from the API"""
        try:
            # Usar optimize_odds_integration para obtener datos con soporte de caché y normalización
            from optimize_odds_integration import get_fixture_odds as get_fixture_odds_optimized
            
            # Obtener odds del sistema optimizado
            normalized_odds = get_fixture_odds_optimized(
                fixture_id=fixture_id,
                use_cache=True,
                force_refresh=False
            )
            
            # Registrar movimiento de odds si tenemos datos válidos
            if normalized_odds and not normalized_odds.get("simulated", True):
                self.movement_tracker.record_odds(fixture_id, normalized_odds)
            
            return normalized_odds
            
        except ImportError:
            # Fallback al método original si no podemos importar el optimizado
            try:
                url = f"{self.base_url}/odds"
                headers = {
                    'x-rapidapi-host': 'v3.football.api-sports.io',
                    'x-rapidapi-key': self.api_key
                }
                params = {
                    'fixture': fixture_id
                }
                
                response = requests.get(url, headers=headers, params=params)
                if response.status_code != 200:
                    logger.error(f"API error: {response.status_code}")
                    return None
                    
                data = response.json()
                if not data.get('response'):
                    return None
                
                try:
                    from odds_normalizer import normalize_odds
                    return normalize_odds(data['response'][0])
                except ImportError:
                    return data['response'][0]
                
            except Exception as e:
                logger.error(f"Error getting odds data: {e}")
                return None
            
    def get_real_time_odds(self, fixture_id: int) -> Optional[Dict[str, Any]]:
        """Get real-time odds data for a specific fixture"""
        try:
            odds_data = self.get_fixture_odds(fixture_id)
            if odds_data:
                # Extract and return only the real-time odds portion
                real_time_odds = odds_data.get("real_time_odds", {})
                return real_time_odds
            else:
                return {}
        except Exception as e:
            logger.error(f"Error getting real-time odds for fixture {fixture_id}: {e}")
            return {}
            
    def calculate_market_efficiency(self, odds_data: Dict[str, Any]) -> Dict[str, float]:
        """Calculate market efficiency metrics"""
        try:
            bookmakers = odds_data.get('bookmakers', [])
            if not bookmakers:
                return {"efficiency": 0.0, "margin": 1.0}
            
            # Get match winner odds from all bookmakers
            all_margins: List[float] = []
            all_efficiencies: List[float] = []
            
            for bookie in bookmakers:
                bets = bookie.get('bets', [])
                match_odds = next((bet for bet in bets if bet.get('name') == 'Match Winner'), None)
                
                if match_odds:
                    values = match_odds.get('values', [])
                    if len(values) == 3:  # Home, Draw, Away
                        odds = [1/float(v.get('odd', 1000)) for v in values]
                        margin = float(sum(odds) - 1)
                        efficiency = float(1 / (1 + margin))
                        
                        all_margins.append(margin)
                        all_efficiencies.append(efficiency)
                        
            if not all_efficiencies:
                return {"efficiency": 0.0, "margin": 1.0}
                
            # Use average of top 3 most efficient markets (or fewer if less than 3)
            all_efficiencies.sort(reverse=True)
            top_n = min(3, len(all_efficiencies))
            avg_efficiency = float(np.mean(all_efficiencies[:top_n])) if top_n > 0 else 0.0
            
            # Use minimum margin from all markets
            min_margin = float(min(all_margins)) if all_margins else 1.0
            
            return {
                "efficiency": avg_efficiency,
                "margin": min_margin
            }
            
        except Exception as e:
            logger.error(f"Error calculating market efficiency: {e}")
            return {"efficiency": 0.0, "margin": 1.0}
            
    def get_best_odds(self, odds_data: Dict[str, Any], market_type: str) -> Optional[Dict[str, Any]]:
        """Get best available odds for a market type"""
        try:
            bookmakers = odds_data.get('bookmakers', [])
            if not bookmakers:
                return None
                
            best_odds = {}
            
            for bookie in bookmakers:
                bets = bookie.get('bets', [])
                market = next((bet for bet in bets if bet.get('name') == market_type), None)
                
                if market:
                    for value in market.get('values', []):
                        outcome = value.get('value', '').lower()
                        odd = float(value.get('odd', 0))
                        
                        # Check if odds are within configured limits
                        if (self.odds_config['min_odds'] <= odd <= self.odds_config['max_odds'] and
                            (outcome not in best_odds or odd > best_odds[outcome]['odds'])):
                            best_odds[outcome] = {
                                'odds': odd,
                                'bookmaker': bookie.get('name')
                            }
                            
            return best_odds if best_odds else None
            
        except Exception as e:
            logger.error(f"Error getting best odds: {e}")
            return None
            
    def calculate_edge(self, our_prob: float, market_odds: float) -> float:
        """Calculate edge percentage for a bet"""
        try:
            implied_prob = 1 / market_odds
            edge = (our_prob * market_odds - 1) * 100  # Convert to percentage
            return round(edge, 2)
        except Exception as e:
            logger.error(f"Error calculating edge: {e}")
            return 0.0
            
    def get_value_opportunities(self, fixture_id: int, prediction: Dict[str, Any],
                              min_edge: float = 2.0,
                              min_efficiency: float = 0.90) -> Optional[Dict[str, Any]]:
        """Find value betting opportunities for a fixture"""
        try:
            # Get odds data
            odds_data = self.get_fixture_odds(fixture_id)
            if not odds_data:
                return None
                
            # Calculate market efficiency
            market_analysis = self.calculate_market_efficiency(odds_data)
            if market_analysis["efficiency"] < min_efficiency:
                logger.warning(f"Market efficiency below threshold: {market_analysis['efficiency']:.2%}")
                return {"market_analysis": market_analysis}
                
            value_opps = {"market_analysis": market_analysis}
            
            # Analyze match result market
            match_odds = self.get_best_odds(odds_data, "Match Winner")
            if match_odds:
                match_value = {}
                
                # Check home win
                if "home" in match_odds:
                    edge = self.calculate_edge(
                        prediction.get("prob_home_win", 0),
                        match_odds["home"]["odds"]
                    )
                    if abs(edge) >= min_edge:
                        match_value["home"] = {
                            "edge": edge,
                            "our_prob": prediction.get("prob_home_win", 0),
                            "market_odds": match_odds["home"]["odds"]
                        }
                
                # Check draw
                if "draw" in match_odds:
                    edge = self.calculate_edge(
                        prediction.get("prob_draw", 0),
                        match_odds["draw"]["odds"]
                    )
                    if abs(edge) >= min_edge:
                        match_value["draw"] = {
                            "edge": edge,
                            "our_prob": prediction.get("prob_draw", 0),
                            "market_odds": match_odds["draw"]["odds"]
                        }
                
                # Check away win
                if "away" in match_odds:
                    edge = self.calculate_edge(
                        prediction.get("prob_away_win", 0),
                        match_odds["away"]["odds"]
                    )
                    if abs(edge) >= min_edge:
                        match_value["away"] = {
                            "edge": edge,
                            "our_prob": prediction.get("prob_away_win", 0),
                            "market_odds": match_odds["away"]["odds"]
                        }
                        
                if match_value:
                    value_opps["match_result"] = match_value
                    
            # Analyze over/under markets
            ou_odds = self.get_best_odds(odds_data, "Over/Under")
            if ou_odds:
                ou_value = {}
                
                # Check over 2.5
                if "over 2.5" in ou_odds:
                    edge = self.calculate_edge(
                        prediction.get("prob_over_2_5", 0),
                        ou_odds["over 2.5"]["odds"]
                    )
                    if abs(edge) >= min_edge:
                        ou_value["over_2.5"] = {
                            "edge": edge,
                            "our_prob": prediction.get("prob_over_2_5", 0),
                            "market_odds": ou_odds["over 2.5"]["odds"]
                        }
                
                # Check under 2.5
                if "under 2.5" in ou_odds:
                    edge = self.calculate_edge(
                        prediction.get("prob_under_2_5", 0),
                        ou_odds["under 2.5"]["odds"]
                    )
                    if abs(edge) >= min_edge:
                        ou_value["under_2.5"] = {
                            "edge": edge,
                            "our_prob": prediction.get("prob_under_2_5", 0),
                            "market_odds": ou_odds["under 2.5"]["odds"]
                        }
                        
                if ou_value:
                    value_opps["goals"] = ou_value
                    
            # Add BTTS market analysis
            btts_odds = self.get_best_odds(odds_data, "Both Teams Score")
            if btts_odds:
                btts_value = {}
                
                if "yes" in btts_odds:
                    edge = self.calculate_edge(
                        prediction.get("prob_btts", 0),
                        btts_odds["yes"]["odds"]
                    )
                    if abs(edge) >= min_edge:
                        btts_value["yes"] = {
                            "edge": edge,
                            "our_prob": prediction.get("prob_btts", 0),
                            "market_odds": btts_odds["yes"]["odds"]
                        }
                        
                if "no" in btts_odds:
                    edge = self.calculate_edge(
                        1 - prediction.get("prob_btts", 0),
                        btts_odds["no"]["odds"]
                    )
                    if abs(edge) >= min_edge:
                        btts_value["no"] = {
                            "edge": edge,
                            "our_prob": 1 - prediction.get("prob_btts", 0),
                            "market_odds": btts_odds["no"]["odds"]
                        }
                        
                if btts_value:
                    value_opps["btts"] = btts_value
                    
            # Add corners market analysis if available
            corners_odds = self.get_best_odds(odds_data, "Corners Over/Under")
            if corners_odds and "corners" in prediction:
                corners_value = {}
                corners_pred = prediction["corners"]
                
                for line in [8.5, 9.5, 10.5]:
                    over_key = f"over {line}"
                    under_key = f"under {line}"
                    
                    if over_key in corners_odds:
                        prob_over = self._calculate_corners_probability(corners_pred, line, "over")
                        edge = self.calculate_edge(prob_over, corners_odds[over_key]["odds"])
                        if abs(edge) >= min_edge:
                            corners_value[f"over_{line}"] = {
                                "edge": edge,
                                "our_prob": prob_over,
                                "market_odds": corners_odds[over_key]["odds"]
                            }
                            
                    if under_key in corners_odds:
                        prob_under = self._calculate_corners_probability(corners_pred, line, "under")
                        edge = self.calculate_edge(prob_under, corners_odds[under_key]["odds"])
                        if abs(edge) >= min_edge:
                            corners_value[f"under_{line}"] = {
                                "edge": edge,
                                "our_prob": prob_under,
                                "market_odds": corners_odds[under_key]["odds"]
                            }
                            
                if corners_value:
                    value_opps["corners"] = corners_value
                    
            # Add cards market analysis if available
            cards_odds = self.get_best_odds(odds_data, "Cards Over/Under")
            if cards_odds and "cards" in prediction:
                cards_value = {}
                cards_pred = prediction["cards"]
                
                for line in [3.5, 4.5]:
                    over_key = f"over {line}"
                    under_key = f"under {line}"
                    
                    if over_key in cards_odds:
                        prob_over = self._calculate_cards_probability(cards_pred, line, "over")
                        edge = self.calculate_edge(prob_over, cards_odds[over_key]["odds"])
                        if abs(edge) >= min_edge:
                            cards_value[f"over_{line}"] = {
                                "edge": edge,
                                "our_prob": prob_over,
                                "market_odds": cards_odds[over_key]["odds"]
                            }
                            
                    if under_key in cards_odds:
                        prob_under = self._calculate_cards_probability(cards_pred, line, "under")
                        edge = self.calculate_edge(prob_under, cards_odds[under_key]["odds"])
                        if abs(edge) >= min_edge:
                            cards_value[f"under_{line}"] = {
                                "edge": edge,
                                "our_prob": prob_under,
                                "market_odds": cards_odds[under_key]["odds"]
                            }
                            
                if cards_value:
                    value_opps["cards"] = cards_value
                    
            return value_opps
            
        except Exception as e:
            logger.error(f"Error finding value opportunities: {e}")
            return None
            
    def _calculate_corners_probability(self, prediction: Dict[str, Any],
                                     line: float, direction: str) -> float:
        """Calculate probability for corners over/under"""
        try:
            mean = prediction.get("expected_corners", 10)
            std = prediction.get("corners_std", 2)
            
            if direction == "over":
                return float(1 - norm.cdf(line, mean, std))
            else:
                return float(norm.cdf(line, mean, std))
        except Exception as e:
            logger.error(f"Error calculating corners probability: {e}")
            return 0.5
            
    def _calculate_cards_probability(self, prediction: Dict[str, Any],
                                   line: float, direction: str) -> float:
        """Calculate probability for cards over/under"""
        try:
            mean = prediction.get("expected_cards", 4)
            std = prediction.get("cards_std", 1)
            
            if direction == "over":
                return float(1 - norm.cdf(line, mean, std))
            else:
                return float(norm.cdf(line, mean, std))
        except Exception as e:
            logger.error(f"Error calculating cards probability: {e}")
            return 0.5
        
    def detect_significant_market_movements(self, fixture_id: int) -> Dict[str, Any]:
        """
        Detect significant odds movements for a fixture.
        
        Args:
            fixture_id: ID of the fixture
            
        Returns:
            Dictionary with information about significant movements
        """
        try:
            # Get odds movements from tracker
            movements = self.movement_tracker.get_odds_movements(fixture_id)
            
            # Filter to significant movements
            significant_movements = [m for m in movements 
                                   if m["significance"] in ["strong", "very strong"]]
            
            result = {
                "fixture_id": fixture_id,
                "significant_movements": len(significant_movements) > 0,
                "movements": significant_movements,
                "movement_count": len(significant_movements)
            }
            
            # Add implications
            implications = []
            if significant_movements:
                for movement in significant_movements:
                    market = movement["market"]
                    selection = movement["selection"]
                    trend = movement["trend"]
                    desc = f"{selection} {trend} in {market} market"
                    implications.append(desc)
                    
                result["implications"] = implications
                
            return result
        except Exception as e:
            logger.error(f"Error detecting significant market movements: {e}")
            return {
                "fixture_id": fixture_id,
                "significant_movements": False,
                "movements": [],
                "movement_count": 0
            }
            
    def calibrate_prediction_with_market(self, prediction: Dict[str, Any], fixture_id: int) -> Dict[str, Any]:
        """
        Calibrate prediction values based on market odds.
        
        Args:
            prediction: Original prediction dictionary
            fixture_id: ID of the fixture
            
        Returns:
            Calibrated prediction
        """
        # Get odds data
        odds_data = self.get_fixture_odds(fixture_id)
        if not odds_data:
            return prediction
        
        try:
            bookmakers = odds_data.get('bookmakers', [])
            if not bookmakers:
                return prediction
                
            # Calibrate based on match result market
            match_odds = self.get_best_odds(odds_data, "Match Winner")
            if match_odds:
                for outcome, data in match_odds.items():
                    if outcome == "home":
                        prediction["prob_home_win"] = data["odds"]
                    elif outcome == "draw":
                        prediction["prob_draw"] = data["odds"]
                    elif outcome == "away":
                        prediction["prob_away_win"] = data["odds"]
            
            # Calibrate based on over/under 2.5 goals market
            ou_odds = self.get_best_odds(odds_data, "Over/Under")
            if ou_odds:
                if "over 2.5" in ou_odds:
                    prediction["prob_over_2_5"] = ou_odds["over 2.5"]["odds"]
                if "under 2.5" in ou_odds:
                    prediction["prob_under_2_5"] = ou_odds["under 2.5"]["odds"]
            
            # Calibrate based on BTTS market
            btts_odds = self.get_best_odds(odds_data, "Both Teams Score")
            if btts_odds:
                if "yes" in btts_odds:
                    prediction["prob_btts"] = btts_odds["yes"]["odds"]
                if "no" in btts_odds:
                    prediction["prob_btts_no"] = btts_odds["no"]["odds"]
            
            # Calibrate corners and cards probabilities if available
            if "corners" in prediction:
                corners_pred = prediction["corners"]
                prediction["prob_corners_over"] = self._calculate_corners_probability(corners_pred, 9.5, "over")
                prediction["prob_corners_under"] = self._calculate_corners_probability(corners_pred, 9.5, "under")
                
            if "cards" in prediction:
                cards_pred = prediction["cards"]
                prediction["prob_cards_over"] = self._calculate_cards_probability(cards_pred, 4.5, "over")
                prediction["prob_cards_under"] = self._calculate_cards_probability(cards_pred, 4.5, "under")
                
            return prediction
        except Exception as e:
            logger.error(f"Error calibrating prediction with market: {e}")
            return prediction
