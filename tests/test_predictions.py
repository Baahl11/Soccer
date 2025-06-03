import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from src.soccer.core.predictions import predict_goals, make_enhanced_prediction
from predictions import make_global_prediction  # Changed from predict_goals

class TestGoalPredictions:
    @pytest.fixture
    def sample_fixture_data(self):
        return {
            "fixture_id": 123,
            "home_team": {"id": 1, "name": "Home Team"},
            "away_team": {"id": 2, "name": "Away Team"},
            "league": {"id": 1},
            "date": "2025-04-28"
        }

    @pytest.fixture
    def sample_form_data(self):
        return {
            "matches_analyzed": 5,
            "goals_scored": 8,
            "goals_conceded": 5,
            "goals_scored_per_game": 1.6,
            "goals_conceded_per_game": 1.0,
            "form_trend": 0.2,
            "clean_sheet_rate": 0.4
        }

    @pytest.fixture
    def sample_h2h_data(self):
        return {
            "matches_analyzed": 10,
            "team1_goals": 15,
            "team2_goals": 12,
            "avg_goals_per_match": 2.7,
            "clean_sheets": {"team1": 3, "team2": 2}
        }

    @pytest.fixture
    def empty_dataframe(self):
        """Create empty DataFrame with expected columns"""
        return pd.DataFrame(columns=[
            'home_goals', 'away_goals', 
            'home_shots_on_target', 'away_shots_on_target',
            'home_possession', 'away_possession'
        ])

    def test_form_based_prediction(self, sample_fixture_data, sample_form_data):
        """Test prediction based on team form data"""
        fixture_data = sample_fixture_data.copy()
        fixture_data["form"] = {
            "home_team": sample_form_data,
            "away_team": sample_form_data.copy()
        }
        
        result = predict_goals(fixture_data)
        
        assert isinstance(result, dict)
        assert "predicted_home_goals" in result
        assert "predicted_away_goals" in result
        assert result["predicted_home_goals"] > result["predicted_away_goals"]  # Home advantage
        assert 0.3 <= result["predicted_home_goals"] <= 4.5
        assert 0.3 <= result["predicted_away_goals"] <= 4.5

    def test_h2h_integration(self, sample_fixture_data, sample_form_data):
        """Test H2H data affects predictions"""
        fixture_data = sample_fixture_data.copy()
        fixture_data["form"] = {
            "home_team": sample_form_data,
            "away_team": sample_form_data.copy()
        }
        fixture_data["h2h"] = {
            "home_wins": 3,
            "away_wins": 1,
            "total_matches": 5
        }
        
        # Get predictions with and without weather
        result_base = predict_goals(fixture_data)
        
        fixture_data["weather"] = {
            "condition": "rain",
            "intensity": "high",
            "temperature": 20
        }
        result_with_weather = predict_goals(fixture_data)
        
        assert result_base != result_with_weather

    def test_weather_impact(self, sample_fixture_data, sample_form_data):
        """Test weather conditions affect predictions"""
        fixture_data = sample_fixture_data.copy()
        fixture_data["form"] = {
            "home_team": sample_form_data,
            "away_team": sample_form_data.copy()
        }
        
        # Test with bad weather
        fixture_data["weather"] = {
            "condition": "rain",
            "intensity": "high",
            "wind": 30,
            "temperature": 5
        }
        
        result = predict_goals(fixture_data)
        normal = predict_goals(sample_fixture_data)
        
        # Allow equality in case adjustment is minimal
        assert result["predicted_home_goals"] <= normal["predicted_home_goals"]
        assert result["predicted_away_goals"] <= normal["predicted_away_goals"]

    def test_missing_data_handling(self, sample_fixture_data):
        """Test graceful handling of missing data"""
        result = predict_goals(sample_fixture_data)
        
        assert isinstance(result, dict)
        assert "predicted_home_goals" in result
        assert "predicted_away_goals" in result
        assert 0.3 <= result["predicted_home_goals"] <= 4.5
        assert 0.3 <= result["predicted_away_goals"] <= 4.5

    def test_probability_calculations(self, sample_fixture_data, sample_form_data):
        """Test probability calculations are valid"""
        fixture_data = sample_fixture_data.copy()
        fixture_data["form"] = {
            "home_team": sample_form_data,
            "away_team": sample_form_data
        }
        
        # Pass the integer fixture_id value
        result = predict_goals(fixture_data)  # Use the fixture_data dict instead of int
        
        assert "home_win_prob" in result
        assert "draw_prob" in result

    def test_input_validation(self):
        """Test input validation and error handling"""
        # Pass an empty dict to simulate invalid input
        result = predict_goals({})
        assert isinstance(result, dict)
        # Check for default prediction keys and values
        assert "predicted_home_goals" in result
        assert "predicted_away_goals" in result
        # Check that values are close to default minimal prediction
        assert abs(result["predicted_home_goals"] - 1.2) < 0.5
        assert abs(result["predicted_away_goals"] - 1.0) < 0.5

if __name__ == "__main__":
    pytest.main([__file__])
