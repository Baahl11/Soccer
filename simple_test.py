#!/usr/bin/env python3
# Script simple para probar predicciones

from enhanced_match_winner import predict_with_enhanced_system

print("Testing enhanced predictions...")

try:
    # Test 1: Manchester United vs Liverpool
    result1 = predict_with_enhanced_system(33, 40, 39)
    probs1 = result1.get('probabilities', {})
    print(f"Man Utd vs Liverpool: {probs1.get('home_win', 0)}% / {probs1.get('draw', 0)}% / {probs1.get('away_win', 0)}%")
    
    # Test 2: Real Madrid vs Barcelona
    result2 = predict_with_enhanced_system(541, 529, 140)
    probs2 = result2.get('probabilities', {})
    print(f"Real Madrid vs Barcelona: {probs2.get('home_win', 0)}% / {probs2.get('draw', 0)}% / {probs2.get('away_win', 0)}%")
    
    # Test 3: Different teams
    result3 = predict_with_enhanced_system(157, 165, 78)
    probs3 = result3.get('probabilities', {})
    print(f"Bayern vs Dortmund: {probs3.get('home_win', 0)}% / {probs3.get('draw', 0)}% / {probs3.get('away_win', 0)}%")
    
    # Check if probabilities are different
    prob1_tuple = (probs1.get('home_win', 0), probs1.get('draw', 0), probs1.get('away_win', 0))
    prob2_tuple = (probs2.get('home_win', 0), probs2.get('draw', 0), probs2.get('away_win', 0))
    prob3_tuple = (probs3.get('home_win', 0), probs3.get('draw', 0), probs3.get('away_win', 0))
    
    if prob1_tuple == prob2_tuple == prob3_tuple:
        print("❌ PROBLEM: All probabilities are identical!")
    else:
        print("✅ SUCCESS: Probabilities are different!")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
