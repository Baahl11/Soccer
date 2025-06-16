#!/usr/bin/env python3

import requests
import json

def comprehensive_validation():
    print('=== COMPREHENSIVE SYSTEM VALIDATION ===')
    
    try:
        # Test 1: Multiple predictions with confidence variety
        response = requests.get('http://127.0.0.1:5000/api/upcoming_predictions?limit=5')
        data = response.json()
        
        print(f'✅ API Response: Found {data.get("count", 0)} predictions')
        
        confidence_values = []
        confidence_scores = []
        
        for i, pred in enumerate(data.get('match_predictions', [])[:5]):
            top_confidence = pred.get('confidence')
            nested_pred = pred.get('prediction', {})
            nested_confidence = nested_pred.get('confidence', {}).get('score')
            
            confidence_values.append(top_confidence)
            confidence_scores.append(nested_confidence)
            
            home_name = pred.get("home_team", {}).get("name", "?")[:15]
            away_name = pred.get("away_team", {}).get("name", "?")[:15]
            
            probs = nested_pred.get("probabilities", {})
            home_prob = probs.get("home_win", 0)
            draw_prob = probs.get("draw", 0)
            away_prob = probs.get("away_win", 0)
            
            print(f'\nMatch {i+1}: {home_name} vs {away_name}')
            print(f'  Top-level confidence: {top_confidence}')
            print(f'  Nested confidence score: {nested_confidence}')
            print(f'  Win probabilities: H={home_prob:.3f} D={draw_prob:.3f} A={away_prob:.3f}')
        
        print(f'\n=== CONFIDENCE ANALYSIS ===')
        print(f'Top-level confidences: {confidence_values}')
        print(f'Nested confidence scores: {confidence_scores}')
        print(f'Unique top-level values: {len(set(confidence_values))}')
        print(f'Range: {min(confidence_values):.3f} - {max(confidence_values):.3f}')
        
        # Check if we're getting proper dynamic values
        hardcoded_count = sum(1 for c in confidence_values if c in [0.5, 0.7])
        print(f'Hardcoded values detected: {hardcoded_count}/5')
        
        if hardcoded_count == 0:
            print('🎉 SUCCESS: All confidence values are dynamic!')
            return True
        else:
            print(f'⚠️  Still found {hardcoded_count} hardcoded values')
            return False
            
    except Exception as e:
        print(f'❌ Error during validation: {e}')
        return False

if __name__ == "__main__":
    success = comprehensive_validation()
    print(f"\n=== FINAL RESULT ===")
    if success:
        print("✅ SYSTEM VALIDATION PASSED")
    else:
        print("❌ SYSTEM VALIDATION FAILED")
