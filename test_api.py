#!/usr/bin/env python3

import requests
import json
import sys

def test_predictions_endpoint():
    """Test the main prediction endpoint to verify confidence values"""
    try:
        print('Testing /api/predictions endpoint...')
        response = requests.get('http://127.0.0.1:5000/api/predictions', timeout=30)
        print(f'Status Code: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()
            print(f'Found {len(data)} predictions')
            
            # Check first few predictions for confidence values
            confidence_values = []
            for i, pred in enumerate(data[:5]):  # Check first 5
                confidence = pred.get('score', 'N/A')
                confidence_values.append(confidence)
                
                print(f'Prediction {i+1}:')
                print(f'  Teams: {pred.get("home_team", "Unknown")} vs {pred.get("away_team", "Unknown")}')
                print(f'  Confidence: {confidence}')
                print(f'  Home Win Prob: {pred.get("home_win_prob", "N/A")}')
                print(f'  Draw Prob: {pred.get("draw_prob", "N/A")}')
                print(f'  Away Win Prob: {pred.get("away_win_prob", "N/A")}')
                print(f'  League: {pred.get("league", "Unknown")}')
                print()
            
            # Analyze confidence values
            print("CONFIDENCE ANALYSIS:")
            print(f"Confidence values found: {confidence_values}")
            
            # Check if we're still getting hardcoded values
            hardcoded_count = sum(1 for c in confidence_values if c in [0.5, 0.7])
            dynamic_count = len(confidence_values) - hardcoded_count
            
            print(f"Hardcoded values (0.5 or 0.7): {hardcoded_count}")
            print(f"Dynamic values: {dynamic_count}")
            
            if dynamic_count > 0:
                print("✅ SUCCESS: Dynamic confidence values detected!")
            else:
                print("❌ ISSUE: All values appear to be hardcoded defaults")
                
        else:
            print(f'Error: {response.text}')
            return False
            
    except Exception as e:
        print(f'Error testing endpoint: {e}')
        return False
    
    return True

def test_specific_match_endpoint():
    """Test a specific match endpoint"""
    try:
        print('\nTesting /api/predict endpoint...')
        # Try to get a specific prediction
        response = requests.post('http://127.0.0.1:5000/api/predict', 
                               json={'team1': 'Real Madrid', 'team2': 'Barcelona'},
                               timeout=30)
        print(f'Status Code: {response.status_code}')
        
        if response.status_code == 200:
            data = response.json()
            print(f'Prediction data: {json.dumps(data, indent=2)}')
            return True
        else:
            print(f'Error: {response.text}')
            
    except Exception as e:
        print(f'Error testing specific match endpoint: {e}')
        
    return False

if __name__ == "__main__":
    print("=== SOCCER PREDICTION API TESTING ===\n")
    
    success1 = test_predictions_endpoint()
    success2 = test_specific_match_endpoint()
    
    print(f"\n=== SUMMARY ===")
    print(f"Predictions endpoint: {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Specific match endpoint: {'✅ PASS' if success2 else '❌ FAIL'}")
