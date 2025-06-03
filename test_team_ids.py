#!/usr/bin/env python3

from predictions import make_global_prediction

def test_team_ids():
    try:
        print("Testing make_global_prediction for team IDs...")
        prediction = make_global_prediction(1208382)
        
        print(f"home_team_id: {prediction.get('home_team_id')}")
        print(f"away_team_id: {prediction.get('away_team_id')}")
        print(f"confidence: {prediction.get('confidence')}")
        
        # Test if team IDs are available for confidence calculation
        from app import calculate_dynamic_confidence
        
        print("\nTesting dynamic confidence calculation...")
        dynamic_conf = calculate_dynamic_confidence(prediction)
        print(f"Dynamic confidence: {dynamic_conf}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_team_ids()
