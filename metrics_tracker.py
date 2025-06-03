import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import sqlite3
import json
import pandas as pd
import numpy as np
from odds_analyzer import OddsAnalyzer

logger = logging.getLogger(__name__)

class MetricsTracker:
    def __init__(self, db_path: str = "soccer_cache.sqlite"):
        self.db_path = db_path
        self.odds_analyzer = OddsAnalyzer()
        self._init_db()
        
    def _init_db(self):
        """Initialize database tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Prediction accuracy tracking
            c.execute("""
                CREATE TABLE IF NOT EXISTS prediction_accuracy (
                    fixture_id INTEGER PRIMARY KEY,
                    timestamp TEXT,
                    predicted_home_goals REAL,
                    predicted_away_goals REAL,
                    actual_home_goals INTEGER,
                    actual_away_goals INTEGER,
                    predicted_result TEXT,
                    actual_result TEXT,
                    prediction_correct INTEGER,
                    goals_error REAL,
                    league_id INTEGER,
                    season INTEGER
                )
            """)
            
            # Value bet tracking
            c.execute("""
                CREATE TABLE IF NOT EXISTS value_bets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fixture_id INTEGER,
                    timestamp TEXT,
                    market_type TEXT,
                    outcome TEXT,
                    predicted_prob REAL,
                    market_odds REAL,
                    edge REAL,
                    market_efficiency REAL,
                    bet_won INTEGER,
                    profit_loss REAL,
                    league_id INTEGER,
                    season INTEGER,
                    UNIQUE(fixture_id, market_type, outcome)
                )
            """)
            
            # Market efficiency tracking
            c.execute("""
                CREATE TABLE IF NOT EXISTS market_efficiency (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fixture_id INTEGER,
                    timestamp TEXT,
                    efficiency REAL,
                    margin REAL,
                    league_id INTEGER,
                    season INTEGER,
                    UNIQUE(fixture_id)
                )
            """)
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Database initialization error: {e}")

    def track_prediction(self, fixture_id: int, prediction: Dict[str, Any], 
                        actual_result: Dict[str, Any], league_id: int, season: int):
        """Track prediction accuracy for a fixture"""
        try:
            # Extract predicted values
            pred_home = prediction.get("predicted_home_goals", 0)
            pred_away = prediction.get("predicted_away_goals", 0)
            
            # Extract actual values
            actual_home = actual_result.get("goals", {}).get("home", 0)
            actual_away = actual_result.get("goals", {}).get("away", 0)
            
            # Calculate prediction results
            pred_result = "H" if pred_home > pred_away else "D" if pred_home == pred_away else "A"
            actual_result_str = "H" if actual_home > actual_away else "D" if actual_home == actual_away else "A"
            prediction_correct = 1 if pred_result == actual_result_str else 0
            
            # Calculate goal prediction error
            goals_error = ((pred_home - actual_home) ** 2 + (pred_away - actual_away) ** 2) ** 0.5
            
            # Store in database
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                INSERT OR REPLACE INTO prediction_accuracy 
                (fixture_id, timestamp, predicted_home_goals, predicted_away_goals,
                 actual_home_goals, actual_away_goals, predicted_result,
                 actual_result, prediction_correct, goals_error, league_id, season)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                fixture_id, datetime.now().isoformat(), pred_home, pred_away,
                actual_home, actual_away, pred_result, actual_result_str,
                prediction_correct, goals_error, league_id, season
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error tracking prediction: {e}")

    def track_value_bets(self, fixture_id: int, prediction: Dict[str, Any], 
                        odds_data: Dict[str, Any], actual_result: Dict[str, Any], 
                        league_id: int, season: int):
        """Track value bet performance"""
        try:
            value_opps = self.odds_analyzer.get_value_opportunities(
                fixture_id, prediction, min_edge=2.0, min_efficiency=0.90
            )
            
            if not value_opps:
                return
                
            # Get market efficiency
            market_analysis = value_opps.get("market_analysis", {})
            efficiency = market_analysis.get("efficiency", 0)
            margin = market_analysis.get("margin", 0)
            
            # Store market efficiency
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            c.execute("""
                INSERT OR REPLACE INTO market_efficiency
                (fixture_id, timestamp, efficiency, margin, league_id, season)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                fixture_id, datetime.now().isoformat(), efficiency, margin,
                league_id, season
            ))
            
            # Process each market type
            for market_type, outcomes in value_opps.items():
                if market_type == "market_analysis":
                    continue
                    
                for outcome, data in outcomes.items():
                    edge = data.get("edge", 0)
                    our_prob = data.get("our_prob", 0)
                    market_odds = data.get("market_odds", 0)
                    
                    # Determine if bet won
                    bet_won = self._check_bet_outcome(
                        market_type, outcome, actual_result
                    )
                    
                    # Calculate profit/loss (1 unit stakes)
                    profit_loss = (market_odds - 1) if bet_won else -1
                    
                    # Store value bet
                    c.execute("""
                        INSERT OR REPLACE INTO value_bets
                        (fixture_id, timestamp, market_type, outcome, predicted_prob,
                         market_odds, edge, market_efficiency, bet_won, profit_loss,
                         league_id, season)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        fixture_id, datetime.now().isoformat(), market_type,
                        outcome, our_prob, market_odds, edge, efficiency,
                        bet_won, profit_loss, league_id, season
                    ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error tracking value bets: {e}")

    def _check_bet_outcome(self, market_type: str, outcome: str, 
                          actual_result: Dict[str, Any]) -> int:
        """Check if a bet won based on actual result"""
        try:
            if market_type == "match_result":
                home_goals = actual_result.get("goals", {}).get("home", 0)
                away_goals = actual_result.get("goals", {}).get("away", 0)
                
                actual = "home" if home_goals > away_goals else "draw" if home_goals == away_goals else "away"
                return 1 if outcome == actual else 0
                
            elif market_type == "goals":
                total_goals = actual_result.get("goals", {}).get("home", 0) + \
                            actual_result.get("goals", {}).get("away", 0)
                            
                if "over" in outcome:
                    line = float(outcome.split("_")[1])
                    return 1 if total_goals > line else 0
                else:
                    line = float(outcome.split("_")[1])
                    return 1 if total_goals < line else 0
                    
            elif market_type == "corners":
                total_corners = actual_result.get("statistics", {}).get("corners", {}).get("total", 0)
                
                if "over" in outcome:
                    line = float(outcome.split("_")[1])
                    return 1 if total_corners > line else 0
                else:
                    line = float(outcome.split("_")[1])
                    return 1 if total_corners < line else 0
                    
            return 0
            
        except Exception as e:
            logger.error(f"Error checking bet outcome: {e}")
            return 0

    def get_prediction_accuracy(self, days: int = 30, 
                              league_id: Optional[int] = None) -> Dict[str, Any]:
        """Get prediction accuracy metrics for specified period"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT COUNT(*) as total,
                       SUM(prediction_correct) as correct,
                       AVG(goals_error) as avg_goals_error,
                       league_id,
                       season
                FROM prediction_accuracy
                WHERE timestamp > ?
            """
            
            if league_id:
                query += f" AND league_id = {league_id}"
                
            query += " GROUP BY league_id, season"
            
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            
            df = pd.read_sql_query(query, conn, params=(cutoff,))
            
            results = []
            for _, row in df.iterrows():
                accuracy = row["correct"] / row["total"] if row["total"] > 0 else 0
                results.append({
                    "league_id": row["league_id"],
                    "season": row["season"],
                    "total_predictions": int(row["total"]),
                    "correct_predictions": int(row["correct"]),
                    "accuracy": round(accuracy * 100, 2),
                    "avg_goals_error": round(row["avg_goals_error"], 2)
                })
                
            conn.close()
            return {"prediction_accuracy": results}
            
        except Exception as e:
            logger.error(f"Error getting prediction accuracy: {e}")
            return {"prediction_accuracy": []}

    def get_value_bet_performance(self, days: int = 30,
                                league_id: Optional[int] = None) -> Dict[str, Any]:
        """Get value betting performance metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT market_type,
                       COUNT(*) as total_bets,
                       SUM(bet_won) as winning_bets,
                       SUM(profit_loss) as total_profit,
                       AVG(edge) as avg_edge,
                       AVG(market_efficiency) as avg_efficiency,
                       league_id,
                       season
                FROM value_bets
                WHERE timestamp > ?
            """
            
            if league_id:
                query += f" AND league_id = {league_id}"
                
            query += " GROUP BY market_type, league_id, season"
            
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            
            df = pd.read_sql_query(query, conn, params=(cutoff,))
            
            results = []
            for _, row in df.iterrows():
                win_rate = row["winning_bets"] / row["total_bets"] if row["total_bets"] > 0 else 0
                roi = (row["total_profit"] / row["total_bets"]) * 100 if row["total_bets"] > 0 else 0
                
                results.append({
                    "market_type": row["market_type"],
                    "league_id": row["league_id"],
                    "season": row["season"],
                    "total_bets": int(row["total_bets"]),
                    "winning_bets": int(row["winning_bets"]),
                    "win_rate": round(win_rate * 100, 2),
                    "profit_loss": round(row["total_profit"], 2),
                    "roi": round(roi, 2),
                    "avg_edge": round(row["avg_edge"], 2),
                    "avg_efficiency": round(row["avg_efficiency"], 2)
                })
                
            conn.close()
            return {"value_bet_performance": results}
            
        except Exception as e:
            logger.error(f"Error getting value bet performance: {e}")
            return {"value_bet_performance": []}