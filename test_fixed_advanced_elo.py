"""
Test script for fresh_advanced_elo.py
"""
from fresh_advanced_elo import AdvancedEloAnalytics

def test_advanced_elo():
    """Test the AdvancedEloAnalytics class"""
    print("Testing AdvancedEloAnalytics...")
    analytics = AdvancedEloAnalytics()
    print("Successfully instantiated AdvancedEloAnalytics")
    
    # Test form-adjusted rating
    team_id = 42
    recent_results = ['W', 'W', 'D', 'L', 'W']
    try:
        adjusted_rating = analytics.get_form_adjusted_rating(team_id, recent_results)
        print(f"Form-adjusted rating for team {team_id}: {adjusted_rating}")
    except Exception as e:
        print(f"Error testing get_form_adjusted_rating: {e}")
    
    # Test matchup analysis
    try:
        analysis = analytics.analyze_team_matchup(39, 40, 39)
        print("Team matchup analysis successful:")
        print(f"- Home team rating: {analysis['raw_elo']['home']}")
        print(f"- Away team rating: {analysis['raw_elo']['away']}")
        print(f"- Upset potential: {analysis['upset_potential']['description']}")
    except Exception as e:
        print(f"Error testing analyze_team_matchup: {e}")
    
    print("Tests completed")

if __name__ == "__main__":
    test_advanced_elo()
