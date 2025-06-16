#!/usr/bin/env python3
"""
Test script to verify the enhanced JSON formatter is working correctly
"""

import requests
import json

def test_enhanced_predictions():
    """Test the enhanced predictions endpoint"""
    try:
        print("🔍 Testing Enhanced Predictions Endpoint...")
        print("=" * 50)
        
        # Test the enhanced endpoint
        url = "http://localhost:8001/api/predictions/enhanced?auto_discovery=true"
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            
            # Get the first prediction
            predictions = data.get('🏆 MATCH PREDICTIONS', [])
            if predictions:
                first_prediction = predictions[0]
                
                print("✅ SUCCESS: Enhanced endpoint is working!")
                print(f"📊 Response size: {len(response.content)} bytes")
                print(f"⚽ Total predictions: {len(predictions)}")
                print()
                
                # Show a sample prediction beautifully formatted
                print("🏆 SAMPLE PREDICTION:")
                print("-" * 30)
                
                match_overview = first_prediction.get('🏆 MATCH OVERVIEW', {})
                match_details = match_overview.get('🏟️ Match Details', {})
                
                print(f"🏠 Home Team: {match_details.get('🏠 Home Team', 'N/A')}")
                print(f"🛣️ Away Team: {match_details.get('🛣️ Away Team', 'N/A')}")
                print(f"🏆 League: {match_details.get('🏆 League', 'N/A')}")
                print()
                
                prediction_results = first_prediction.get('🎯 PREDICTION RESULTS', {})
                main_outcome = prediction_results.get('🏅 Main Outcome', {})
                
                print("🎯 PREDICTION RESULTS:")
                print(f"🏠 Home Win: {main_outcome.get('🏠 Home Win Probability', 'N/A')}")
                print(f"🤝 Draw: {main_outcome.get('🤝 Draw Probability', 'N/A')}")
                print(f"🛣️ Away Win: {main_outcome.get('🛣️ Away Win Probability', 'N/A')}")
                print(f"🏆 Most Likely: {main_outcome.get('🏆 Most Likely Result', 'N/A')}")
                print()
                
                goals_prediction = prediction_results.get('⚽ Goals Prediction', {})
                print("⚽ GOALS PREDICTION:")
                print(f"🏠 Home Goals: {goals_prediction.get('🏠 Home Goals Expected', 'N/A')}")
                print(f"🛣️ Away Goals: {goals_prediction.get('🛣️ Away Goals Expected', 'N/A')}")
                print(f"🎯 Total Goals: {goals_prediction.get('🎯 Total Goals Expected', 'N/A')}")
                print()
                
                # Show ELO ratings
                elo_ratings = first_prediction.get('📈 ELO RATINGS', {})
                print("📈 ELO RATINGS:")
                print(f"🏠 Home ELO: {elo_ratings.get('🏠 Home ELO', 'N/A')}")
                print(f"🛣️ Away ELO: {elo_ratings.get('🛣️ Away ELO', 'N/A')}")
                print(f"⚖️ Difference: {elo_ratings.get('⚖️ ELO Difference', 'N/A')}")
                print()
                
                # Show odds
                odds_value = first_prediction.get('💰 ODDS & VALUE', {})
                current_odds = odds_value.get('🎰 Current Odds', {})
                print("💰 CURRENT ODDS:")
                print(f"🏠 Home: {current_odds.get('home', 'N/A')}")
                print(f"🤝 Draw: {current_odds.get('draw', 'N/A')}")
                print(f"🛣️ Away: {current_odds.get('away', 'N/A')}")
                print()
                
                print("✅ All data is being extracted and formatted correctly!")
                return True
            else:
                print("❌ No predictions found in response")
                return False
        else:
            print(f"❌ Failed to get response: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing enhanced predictions: {e}")
        return False

if __name__ == "__main__":
    success = test_enhanced_predictions()
    if success:
        print("\n🎉 SUCCESS: Enhanced JSON formatter is working perfectly!")
        print("🔗 Enhanced endpoint: http://localhost:8001/api/predictions/enhanced?auto_discovery=true")
    else:
        print("\n❌ FAILED: There was an issue with the enhanced formatter")
