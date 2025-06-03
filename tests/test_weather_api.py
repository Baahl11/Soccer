import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Adjust sys.path to import weather_api module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from weather_api import WeatherConditions, get_weather_forecast, get_weather_impact

@pytest.fixture
def weather_conditions():
    return WeatherConditions()

def test_get_match_conditions_returns_default_on_api_error(weather_conditions):
    with patch.object(weather_conditions.api, '_make_request', side_effect=Exception("API error")):
        conditions = weather_conditions.get_match_conditions(12345)
        assert isinstance(conditions, dict)
        assert 'temperature' in conditions
        assert 'humidity' in conditions

def test_get_match_conditions_parses_valid_response(weather_conditions):
    mock_response = {
        'response': [{
            'fixture': {
                'venue': {'id': 1},
                'weather': {
                    'temperature': '20°C',
                    'description': 'Clear',
                    'humidity': '50%',
                    'wind': '5 m/s'
                }
            }
        }]
    }
    with patch.object(weather_conditions.api, '_make_request', return_value=mock_response):
        with patch.object(weather_conditions, '_get_stadium_info', return_value={'name': 'Stadium'}):
            conditions = weather_conditions.get_match_conditions(1)
            assert conditions['temperature'] == 20.0
            assert conditions['humidity'] == 50
            assert conditions['wind'] == 18.0  # 5 m/s to km/h
            assert conditions['stadium_info']['name'] == 'Stadium'

def test_get_weather_impact_returns_dict(weather_conditions):
    with patch.object(weather_conditions, 'get_match_conditions', return_value={
        'temperature': 10,
        'wind': 10,
        'precipitation': 0,
        'stadium_info': {'altitude': 0},
        'possession_impact': 0.0,
        'passing_impact': 0.0,
        'physical_impact': 0.0,
        'technical_impact': 0.0
    }):
        impact = weather_conditions.get_weather_impact(1)
        assert isinstance(impact, dict)
        assert 'overall_impact' in impact or 'confidence' in impact

def test_get_weather_forecast_returns_expected():
    forecast = get_weather_forecast("London", "GB", "2024-06-01")
    assert forecast['city'] == "London"
    assert forecast['country'] == "GB"
    assert forecast['date'] == "2024-06-01"
    assert 'temperature' in forecast

def test_get_weather_impact_function_returns_expected():
    weather_data = {
        "condition": "Clear",
        "temperature": 25,
        "wind_speed": 5,
        "precipitation": 0
    }
    impact = get_weather_impact(weather_data)
    assert isinstance(impact, dict)
    assert 'overall_impact' in impact
    assert impact['overall_impact'] == "neutral"
