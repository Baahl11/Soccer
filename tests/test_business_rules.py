import pytest
from typing import Dict, Any, Optional
from business_rules import adjust_prediction_based_on_lineup
from business_rules import (
    adjust_prediction_for_weather,
    adjust_single_pred_for_weather,
    is_derby_match,
    WEATHER_FACTORS
)

# Weather Test Cases
class TestWeatherAdjustments:
    @pytest.fixture
    def base_prediction(self) -> Dict[str, Any]:
        return {
            "goals": {"home": 2.0, "away": 1.5, "total": 3.5},
            "corners": {"predicted_corners_mean": 10.0},
            "cards": {"total": 4.0}
        }

    def test_rain_adjustments(self, base_prediction: Dict[str, Any]):
        weather = {"condition": "rain", "intensity": "high", "wind": 0, "temperature": 15}
        result = adjust_prediction_for_weather(base_prediction.copy(), weather)
        
        assert result["goals"]["home"] < base_prediction["goals"]["home"]
        assert result["goals"]["away"] < base_prediction["goals"]["away"]
        assert result["corners"]["predicted_corners_mean"] > base_prediction["corners"]["predicted_corners_mean"]

    def test_wind_thresholds(self, base_prediction: Dict[str, Any]):
        # Test each wind threshold
        for speed, expected_impact in [
            (35, "strong"),
            (25, "moderate"),
            (17, "light"),
            (10, None)
        ]:
            weather = {"condition": "clear", "wind": speed, "temperature": 15}
            result = adjust_single_pred_for_weather(10.0, weather)
            
            if expected_impact:
                factor = WEATHER_FACTORS["wind"]["adjustments"][expected_impact]
                assert abs(result - 10.0) > 0
            else:
                assert result == 10.0

    def test_temperature_extremes(self, base_prediction: Dict[str, Any]):
        # Very cold
        cold_weather = {"condition": "clear", "wind": 0, "temperature": -5}
        result = adjust_single_pred_for_weather(10.0, cold_weather)
        assert result < 10.0
        
        # Very hot
        hot_weather = {"condition": "clear", "wind": 0, "temperature": 36}
        result = adjust_single_pred_for_weather(10.0, hot_weather)
        assert result < 10.0

    def test_combined_conditions(self, base_prediction: Dict[str, Any]):
        weather = {
            "condition": "rain",
            "intensity": "high",
            "wind": 25,
            "temperature": 2
        }
        result = adjust_prediction_for_weather(base_prediction.copy(), weather)
        
        # Multiple factors should compound
        assert result != base_prediction
        assert all(result["goals"][k] < base_prediction["goals"][k] for k in ["home", "away"])

    def test_edge_cases(self, base_prediction: Dict[str, Any]):
        # Empty weather
        assert adjust_prediction_for_weather(base_prediction.copy(), {}) == base_prediction
        
        # None values
        weather = {"condition": None, "wind": None, "temperature": None}
        result = adjust_prediction_for_weather(base_prediction.copy(), weather)
        assert isinstance(result, dict)
        
        # Invalid types
        weather = {"condition": 123, "wind": "fast", "temperature": "hot"}
        result = adjust_prediction_for_weather(base_prediction.copy(), weather)
        assert isinstance(result, dict)

# Derby Detection Tests
class TestDerbyDetection:
    @pytest.mark.parametrize("home,away,league_id,expected", [
        ("Manchester United", "Manchester City", 524, True),
        ("Real Madrid", "Atletico Madrid", 564, True),
        ("Chelsea", "Arsenal", 524, False),
        ("Barcelona", "Real Madrid", 564, False),
        ("Manchester United", "Liverpool", 524, False),
    ])
    def test_known_derbies(self, home: str, away: str, league_id: int, expected: bool):
        assert is_derby_match(home, away, league_id) == expected

    def test_city_detection(self):
        # Same city, different suffixes
        assert is_derby_match("Milan", "Inter Milan", 100)
        assert is_derby_match("Real Betis", "Sevilla FC", 100)
        
        # False positives check
        assert not is_derby_match("Real Madrid", "Real Sociedad", 100)
        assert not is_derby_match("Sporting Lisbon", "Sporting Gijon", 100)

    def test_edge_cases(self):
        # Empty/invalid inputs
        assert not is_derby_match("", "", 0)
        assert not is_derby_match(None, None, None)
        
        # Special characters
        assert is_derby_match("Manchester United F.C.", "Manchester City", 524)
        assert is_derby_match("A.C. Milan", "Inter Milan", 100)

    def test_invalid_inputs(self):
        """Test handling of invalid input types"""
        # None values
        assert not is_derby_match(None, "Team B", 123)
        assert not is_derby_match("Team A", None, 123)
        assert not is_derby_match("Team A", "Team B", None)
        assert not is_derby_match(None, None, None)
        
        # Empty strings
        assert not is_derby_match("", "", 123)
        assert not is_derby_match("Team A", "", 123)
        assert not is_derby_match("", "Team B", 123)
        
        # Wrong types - convert numeric values to strings
        assert not is_derby_match("123", "Team B", 123)
        assert not is_derby_match("Team A", "123", 123)
        assert not is_derby_match("Team A", "Team B", 123)

class TestLineupAdjustments:
    @pytest.fixture
    def base_prediction(self) -> Dict[str, Any]:
        return {
            "goals": {
                "predicted_home_goals": 2.0,
                "predicted_away_goals": 1.5
            }
        }

    @pytest.fixture
    def valid_lineup_data(self) -> Dict[str, Any]:
        return {
            "home": {
                "formation": "4-4-2",
                "players": [
                    {"id": 1, "position": "GK"},
                    {"id": 2, "position": "DEF"}
                ]
            },
            "away": {
                "formation": "4-3-3",
                "players": [
                    {"id": 3, "position": "GK"},
                    {"id": 4, "position": "DEF"}
                ]
            }
        }

    def test_basic_adjustment(self, base_prediction: Dict[str, Any], valid_lineup_data: Dict[str, Any]):
        """Test basic prediction adjustment with valid data"""
        result = adjust_prediction_based_on_lineup(base_prediction, valid_lineup_data)
        
        assert isinstance(result, dict)
        assert "goals" in result
        assert "formation_analysis" in result
        assert result != base_prediction  # Should be modified
        
        # Check formation analysis structure
        analysis = result["formation_analysis"]
        assert "home" in analysis
        assert "away" in analysis
        assert "strength" in analysis
        assert "impact_factor" in analysis

    def test_missing_inputs(self, base_prediction: Dict[str, Any]):
        """Test handling of missing/empty inputs"""
        # Test with empty dict instead of None
        assert adjust_prediction_based_on_lineup({}, {}) == {}
        assert adjust_prediction_based_on_lineup(base_prediction, {}) == base_prediction.copy()

    def test_formation_impact(self, base_prediction: Dict[str, Any], valid_lineup_data: Dict[str, Any]):
        """Test that formation impacts are correctly applied"""
        result = adjust_prediction_based_on_lineup(base_prediction, valid_lineup_data)
        
        # Check that goals were adjusted
        original_home = base_prediction["goals"]["predicted_home_goals"]
        original_away = base_prediction["goals"]["predicted_away_goals"]
        
        assert result["goals"]["predicted_home_goals"] != original_home
        assert result["goals"]["predicted_away_goals"] != original_away
        
        # Check impact factor structure
        assert isinstance(result["formation_analysis"]["impact_factor"], dict)
        assert "home" in result["formation_analysis"]["impact_factor"]
        assert "away" in result["formation_analysis"]["impact_factor"]

    def test_prediction_immutability(self, base_prediction: Dict[str, Any], valid_lineup_data: Dict[str, Any]):
        """Test that original prediction isn't modified"""
        original = base_prediction.copy()
        _ = adjust_prediction_based_on_lineup(base_prediction, valid_lineup_data)
        
        assert base_prediction == original
        
    def test_edge_cases(self, base_prediction: Dict[str, Any]):
        """Test edge cases and error handling"""
        # Invalid formation
        invalid_lineup = {
            "home": {"formation": "invalid"},
            "away": {"formation": "5-5-5"}  # Invalid
        }
        result = adjust_prediction_based_on_lineup(base_prediction, invalid_lineup)
        assert isinstance(result, dict)
        
        # Missing keys
        partial_lineup = {
            "home": {"formation": "4-4-2"}
            # Missing away team
        }
        result = adjust_prediction_based_on_lineup(base_prediction, partial_lineup)
        assert result == base_prediction.copy()
        
        # Invalid types
        invalid_types = {
            "home": {"formation": 442},  # Number instead of string
            "away": {"formation": None}
        }
        result = adjust_prediction_based_on_lineup(base_prediction, invalid_types)
        assert isinstance(result, dict)

if __name__ == "__main__":
    pytest.main([__file__])