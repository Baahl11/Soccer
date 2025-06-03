"""
Formation analysis and tactical constants.
Centralized module for formation-related logic.
"""

import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

# Formation impact factors
FORMATION_IMPACT: Dict[str, List[float]] = {
    '3-4-3': [1.25, True],   # [impact_factor, is_offensive]
    '4-3-3': [1.30, True],
    '4-2-3-1': [1.15, True],
    '3-5-2': [1.20, True],
    '4-4-2': [1.10, False],
    '5-3-2': [0.85, False],
    '4-5-1': [0.90, False]
}

# Formation style characteristics
FORMATION_STYLES: Dict[str, Dict[str, float]] = {
    '4-3-3': {'attacking': 0.7, 'possession': 0.6, 'wingers': 0.8},
    '3-4-3': {'attacking': 0.8, 'wingers': 0.7, 'high_press': 0.6},
    '5-3-2': {'defensive': 0.7, 'counter_attack': 0.6},
    '4-2-3-1': {'possession': 0.7, 'high_press': 0.5},
}

def get_formation_strength(
    home_formation: str,
    away_formation: str
) -> Dict[str, Any]:
    """
    Calculates tactical strength based on formation matchup
    Returns dict with strength factor and offensive classification
    """
    try:
        # Get impact factors with defaults
        home_impact, home_offensive = FORMATION_IMPACT.get(
            home_formation, [1.0, False]
        )
        away_impact, away_offensive = FORMATION_IMPACT.get(
            away_formation, [1.0, False]
        )

        strength = home_impact * away_impact
        is_offensive = home_offensive and away_offensive

        return {
            'formation_strength': strength,
            'is_offensive_match': is_offensive,
            'home_impact': home_impact,
            'away_impact': away_impact
        }

    except Exception as e:
        logger.error(f"Formation strength error: {e}")
        return {
            'formation_strength': 1.0,
            'is_offensive_match': False,
            'home_impact': 1.0,
            'away_impact': 1.0
        }

def analyze_formation_style(
    formation: str,
    lineup_data: Dict[str, Any] = None
) -> Dict[str, float]:
    """
    Analyzes formation and lineup to determine playing style
    Returns dict of style characteristics (0-1 scale)
    """
    try:
        # Initialize with defaults
        style = {
            'attacking': 0.0,
            'defensive': 0.0,
            'possession': 0.0,
            'counter_attack': 0.0,
            'high_press': 0.0,
            'wingers': 0.0
        }

        # Get base style from formation
        if formation in FORMATION_STYLES:
            style.update(FORMATION_STYLES[formation])

        # Analyze lineup if provided
        if lineup_data:
            lineup_str = str(lineup_data).lower()
            
            if "winger" in lineup_str:
                style['wingers'] += 0.2
                style['attacking'] += 0.1
                
            if "defensive midfielder" in lineup_str:
                style['defensive'] += 0.2
                style['possession'] += 0.1

        # Normalize values
        return {k: min(v, 1.0) for k, v in style.items()}

    except Exception as e:
        logger.error(f"Formation style analysis error: {e}")
        return style

def test_formation_functions():
    """Unit tests for formation analysis functions"""
    # Test formation strength
    strength = get_formation_strength('4-3-3', '4-4-2')
    assert isinstance(strength, dict)
    assert 'formation_strength' in strength
    assert 'is_offensive_match' in strength

    # Test style analysis
    style = analyze_formation_style('4-3-3')
    assert isinstance(style, dict)
    assert all(0 <= v <= 1 for v in style.values())