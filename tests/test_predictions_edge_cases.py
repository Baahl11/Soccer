import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.soccer.core.predictions import make_global_prediction

class TestPredictionEdgeCases:
    @pytest.fixture
    def base_fixture_data(self):
        return {
            "fixture_id": 123,
            "home_team": {"id": 1, "name": "Home Team"},
            "away_team": {"id": 2, "name": "Away Team"},
            "league": {"id": 1},
            "date": "2025-04-28",
            "form": {
                "home_team": {
                    "matches_analyzed": 5,
                    "goals_scored": 8,
                    "goals_conceded": 5,
                    "goals_scored_per_game": 1.6,
                    "goals_conceded_per_game": 1.0,
                    "form_trend": 0.2,
                    "clean_sheet_rate": 0.4
                },
                "away_team": {
                    "matches_analyzed": 5,
                    "goals_scored": 8,
                    "goals_conceded": 5,
                    "goals_scored_per_game": 1.6,
                    "goals_conceded_per_game": 1.0,
                    "form_trend": 0.2,
                    "clean_sheet_rate": 0.4
                }
            }
        }

    def test_weather_intensity_levels(self, base_fixture_data):
        """Test weather impact with different intensities"""
        intensities = ["light", "medium", "high"]
        for intensity in intensities:
            weather = {
                "condition": "rain",
                "intensity": intensity,
                "wind": 10,
                "temperature": 15
            }
            base_fixture_data["weather"] = weather
            result = make_global_prediction(base_fixture_data["fixture_id"])
            assert "predicted_home_goals" in result
            assert "predicted_away_goals" in result

    def test_weather_conditions(self, base_fixture_data):
        """Test different weather conditions"""
        conditions = ["clear", "rain", "snow"]
        for condition in conditions:
            weather = {
                "condition": condition,
                "intensity": "medium",
                "wind": 15,
                "temperature": 10
            }
            base_fixture_data["weather"] = weather
            result = make_global_prediction(base_fixture_data["fixture_id"])
            assert "predicted_home_goals" in result
            assert "predicted_away_goals" in result

    def test_invalid_fixture_id(self):
        """Test prediction with invalid fixture id"""
        result = make_global_prediction(-9999)
        assert isinstance(result, dict)
        assert "predicted_home_goals" in result
        assert "predicted_away_goals" in result

    def test_missing_weather_data(self, base_fixture_data):
        """Test prediction when weather data is missing"""
        if "weather" in base_fixture_data:
            del base_fixture_data["weather"]
        result = make_global_prediction(base_fixture_data["fixture_id"])
        assert "predicted_home_goals" in result
        assert "predicted_away_goals" in result
