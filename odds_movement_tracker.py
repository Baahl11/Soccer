"""
Module for tracking and analyzing odds movements over time.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class OddsMovementTracker:
    """Tracks and analyzes odds movements over time"""
    
    def __init__(self):
        """Initialize the odds movement tracker"""
        self.odds_history = {}  # Store historical odds by fixture ID
        self.movement_thresholds = {
            "significant": 0.10,  # 10% change in implied probability
            "strong": 0.15,      # 15% change in implied probability
            "very_strong": 0.20  # 20% change in implied probability
        }
    
    def record_odds(self, fixture_id: int, odds_data: Dict[str, Any]) -> None:
        """
        Record odds data for a fixture at the current time.
        
        Args:
            fixture_id: ID of the fixture
            odds_data: Dictionary containing odds information
        """
        try:
            timestamp = datetime.now().isoformat()
            
            if fixture_id not in self.odds_history:
                self.odds_history[fixture_id] = []
                
            # Extract key odds values to track (match odds, over/under, etc.)
            tracked_odds = self._extract_key_odds(odds_data)
            tracked_odds["timestamp"] = timestamp
            
            self.odds_history[fixture_id].append(tracked_odds)
            logger.debug(f"Recorded odds for fixture {fixture_id} at {timestamp}")
            
        except Exception as e:
            logger.error(f"Error recording odds: {e}")
    
    def _extract_key_odds(self, odds_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key odds values from raw odds data"""
        try:
            result = {
                "match_odds": {},
                "over_under": {},
                "btts": {}
            }
            
            bookmakers = odds_data.get("bookmakers", [])
            if not bookmakers:
                return result
                
            # Extract match winner odds
            for bookie in bookmakers:
                bets = bookie.get("bets", [])
                
                # Get match winner odds
                match_odds = next((bet for bet in bets if bet.get("name") == "Match Winner"), None)
                if match_odds:
                    values = match_odds.get("values", [])
                    for val in values:
                        outcome = val.get("value", "").lower()
                        odd = float(val.get("odd", 0))
                        if outcome in ["home", "1"]:
                            result["match_odds"]["home"] = odd
                        elif outcome in ["draw", "x"]:
                            result["match_odds"]["draw"] = odd
                        elif outcome in ["away", "2"]:
                            result["match_odds"]["away"] = odd
                
                # Get over/under odds
                ou_odds = next((bet for bet in bets if bet.get("name") == "Over/Under"), None)
                if ou_odds:
                    values = ou_odds.get("values", [])
                    for val in values:
                        outcome = val.get("value", "").lower()
                        odd = float(val.get("odd", 0))
                        result["over_under"][outcome] = odd
                
                # Get BTTS odds
                btts_odds = next((bet for bet in bets if bet.get("name") == "Both Teams Score"), None)
                if btts_odds:
                    values = btts_odds.get("values", [])
                    for val in values:
                        outcome = val.get("value", "").lower()
                        odd = float(val.get("odd", 0))
                        result["btts"][outcome] = odd
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting key odds: {e}")
            return {
                "match_odds": {},
                "over_under": {},
                "btts": {}
            }
    
    def get_odds_movements(self, fixture_id: int) -> List[Dict[str, Any]]:
        """
        Get odds movements for a fixture.
        
        Args:
            fixture_id: ID of the fixture
            
        Returns:
            List of movement objects with market, selection and change details
        """
        try:
            if fixture_id not in self.odds_history or len(self.odds_history[fixture_id]) < 2:
                return []
                
            history = self.odds_history[fixture_id]
            latest = history[-1]
            earliest = history[0]
            
            movements = []
            
            # Check match odds movements
            for outcome in ["home", "draw", "away"]:
                latest_odd = latest.get("match_odds", {}).get(outcome)
                earliest_odd = earliest.get("match_odds", {}).get(outcome)
                
                if latest_odd and earliest_odd:
                    latest_prob = 1 / latest_odd
                    earliest_prob = 1 / earliest_odd
                    prob_change = latest_prob - earliest_prob
                    trend = "decreasing" if latest_odd > earliest_odd else "increasing"
                    
                    # Check if movement is significant
                    if abs(prob_change) >= self.movement_thresholds["significant"]:
                        movements.append({
                            "market": "Match Winner",
                            "selection": outcome,
                            "initial_odds": earliest_odd,
                            "current_odds": latest_odd,
                            "probability_change": prob_change,
                            "trend": trend,
                            "significance": self._get_significance_level(abs(prob_change))
                        })
            
            # Check over/under movements
            for line in ["over 2.5", "under 2.5"]:
                latest_odd = latest.get("over_under", {}).get(line)
                earliest_odd = earliest.get("over_under", {}).get(line)
                
                if latest_odd and earliest_odd:
                    latest_prob = 1 / latest_odd
                    earliest_prob = 1 / earliest_odd
                    prob_change = latest_prob - earliest_prob
                    trend = "decreasing" if latest_odd > earliest_odd else "increasing"
                    
                    if abs(prob_change) >= self.movement_thresholds["significant"]:
                        movements.append({
                            "market": "Over/Under",
                            "selection": line,
                            "initial_odds": earliest_odd,
                            "current_odds": latest_odd,
                            "probability_change": prob_change,
                            "trend": trend,
                            "significance": self._get_significance_level(abs(prob_change))
                        })
            
            # Sort by significance of movement
            movements.sort(key=lambda x: abs(x["probability_change"]), reverse=True)
            return movements
            
        except Exception as e:
            logger.error(f"Error getting odds movements: {e}")
            return []
    
    def _get_significance_level(self, change: float) -> str:
        """Determine significance level of a probability change"""
        if change >= self.movement_thresholds["very_strong"]:
            return "very strong"
        elif change >= self.movement_thresholds["strong"]:
            return "strong"
        else:
            return "moderate"
    
    def get_market_confidence(self, fixture_id: int) -> float:
        """
        Calculate market confidence based on odds history and movements.
        Higher values indicate more stable markets with confident pricing.
        
        Args:
            fixture_id: ID of the fixture
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            if fixture_id not in self.odds_history or len(self.odds_history[fixture_id]) < 2:
                return 0.5  # Default medium confidence
                
            history = self.odds_history[fixture_id]
            
            # More data points indicate better confidence
            data_points_factor = min(1.0, len(history) / 10)
            
            # Calculate volatility based on odds movements
            movements = self.get_odds_movements(fixture_id)
            volatility = sum(abs(m["probability_change"]) for m in movements) if movements else 0
            volatility_factor = max(0, 1 - min(1, volatility / 0.5))
            
            # Calculate time span of data
            try:
                earliest = datetime.fromisoformat(history[0]["timestamp"])
                latest = datetime.fromisoformat(history[-1]["timestamp"])
                hours_diff = (latest - earliest).total_seconds() / 3600
                time_factor = min(1.0, hours_diff / 24)  # 24 hours gives full confidence
            except (ValueError, KeyError):
                time_factor = 0.5
            
            # Combine factors with appropriate weights
            confidence = (
                data_points_factor * 0.2 +
                volatility_factor * 0.6 +
                time_factor * 0.2
            )
            
            return max(0.1, min(0.95, confidence))
            
        except Exception as e:
            logger.error(f"Error calculating market confidence: {e}")
            return 0.5
