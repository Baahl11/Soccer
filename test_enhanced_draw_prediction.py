"""
Test Enhanced Draw Prediction Implementation

This script provides a simple test to validate that our enhanced prediction system
is working correctly, with a specific focus on draw prediction improvements.
"""

import logging
import json
from typing import Dict, Any, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import local modules
from match_winner import predict_match_winner, MatchOutcome
from enhanced_match_winner import predict_with_enhanced_system
from draw_prediction import DrawPredictor
from team_form import get_team_form, get_head_to_head_analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample matches for testing
# These are selected to include cases where draws are likely
TEST_MATCHES = [
    {
        "home_team_id": 33,  # Manchester United
        "away_team_id": 40,  # Liverpool
        "league_id": 39,     # Premier League
        "info": "Evenly matched rivals"
    },
    {
        "home_team_id": 50,  # Manchester City
        "away_team_id": 47,  # Tottenham
        "league_id": 39,
        "info": "Top teams, potential draw"
    },
    {
        "home_team_id": 42,  # Arsenal
        "away_team_id": 49,  # Chelsea
        "league_id": 39,
        "info": "London derby"
    },
    {
        "home_team_id": 46,  # Leicester
        "away_team_id": 39,  # Wolves
        "league_id": 39,
        "info": "Mid-table clash"
    },
    {
        "home_team_id": 45,  # Everton
        "away_team_id": 34,  # Newcastle
        "league_id": 39,
        "info": "Similar strength teams"
    }
]

def compare_predictions() -> List[Dict[str, Any]]:
    """
    Compare original and enhanced predictions for the test matches.
    
    Returns:
        List of result dictionaries with prediction comparisons
    """
    results = []
    
    draw_predictor = DrawPredictor()
    
    for match in TEST_MATCHES:
        home_team_id = match["home_team_id"]
        away_team_id = match["away_team_id"]
        league_id = match["league_id"]
        info = match["info"]
        
        # Get required data for predictions
        home_form = get_team_form(home_team_id, league_id, None)
        away_form = get_team_form(away_team_id, league_id, None)
        h2h = get_head_to_head_analysis(home_team_id, away_team_id)
        
        # Calculate expected goals based on form
        home_xg = home_form.get('expected_goals_avg', 1.3)
        away_xg = away_form.get('expected_goals_avg', 1.1)
        
        # Original prediction
        original_pred = predict_match_winner(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            league_id=league_id,
            home_xg=home_xg,
            away_xg=away_xg,
            home_form=home_form,
            away_form=away_form,
            h2h=h2h
        )
        
        # Enhanced prediction
        enhanced_pred = predict_with_enhanced_system(
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            league_id=league_id,
            home_xg=home_xg,
            away_xg=away_xg,
            home_form=home_form,
            away_form=away_form,
            h2h=h2h
        )
        
        # Compare draw probabilities
        original_draw_prob = original_pred.get('probabilities', {}).get(MatchOutcome.DRAW.value, 0)
        enhanced_draw_prob = enhanced_pred.get('probabilities', {}).get(MatchOutcome.DRAW.value, 0)
        
        # Compare outcomes
        original_outcome = original_pred.get('predicted_outcome')
        enhanced_outcome = enhanced_pred.get('predicted_outcome')
        
        result = {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'info': info,
            'original': {
                'outcome': original_outcome,
                'draw_prob': original_draw_prob
            },
            'enhanced': {
                'outcome': enhanced_outcome,
                'draw_prob': enhanced_draw_prob
            },
            'draw_prob_change': enhanced_draw_prob - original_draw_prob
        }
        
        results.append(result)
    
    # Print comparison
    print("\n========== PREDICTION COMPARISON ==========\n")
    for i, result in enumerate(results, 1):
        print(f"Test Match {i}: {result['info']}")
        print(f"  Original prediction: {result['original']['outcome'].replace('_', ' ').title()} (Draw prob: {result['original']['draw_prob']:.2f})")
        print(f"  Enhanced prediction: {result['enhanced']['outcome'].replace('_', ' ').title()} (Draw prob: {result['enhanced']['draw_prob']:.2f})")
        print(f"  Draw probability change: {result['draw_prob_change']:.2f}")
        print()
    
    # Visualize draw probability changes
    plt.figure(figsize=(12, 6))
    
    matches = [f"Match {i+1}" for i in range(len(results))]
    original_probs = [r['original']['draw_prob'] for r in results]
    enhanced_probs = [r['enhanced']['draw_prob'] for r in results]
    
    df = pd.DataFrame({
        'Match': matches + matches,
        'Model': ['Original'] * len(results) + ['Enhanced'] * len(results),
        'Draw Probability': original_probs + enhanced_probs
    })
    
    sns.barplot(x='Match', y='Draw Probability', hue='Model', data=df)
    plt.title('Draw Probability Comparison', fontsize=15)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig('draw_probability_comparison.png')
    print("Visualization saved as 'draw_probability_comparison.png'")
    
    return results

if __name__ == "__main__":
    print("Testing enhanced draw prediction system...")
    results = compare_predictions()
    
    # Save results to JSON for reference
    with open('prediction_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nTest results saved to 'prediction_comparison.json'")