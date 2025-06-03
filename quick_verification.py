#!/usr/bin/env python3
"""Quick integration verification"""

import prediction_integration as pi

def main():
    print("🔍 Integration Verification Starting...")
    
    # Test fixture detection
    print("\n1️⃣ Testing fixture detection:")
    tests = [(1000000, True), (999999, False), (1000001, True), (123456, False)]
    detection_ok = True
    
    for fid, expected in tests:
        result = pi.is_synthetic_fixture(fid)
        status = "✅" if result == expected else "❌"
        print(f"   Fixture {fid}: {status} ({'Synthetic' if expected else 'Real'})")
        if result != expected:
            detection_ok = False
    
    # Test mock prediction
    print("\n2️⃣ Testing mock prediction:")
    try:
        mock_data = {
            'fixture_id': 1000123,
            'home_team_id': 51, 'away_team_id': 52,
            'home_team_name': 'Brighton', 'away_team_name': 'Leicester',
            'league_id': 39, 'match_date': '2025-05-30T15:00:00Z', 'season': 2024
        }
        prediction = pi.make_mock_integrated_prediction(mock_data)
        
        if prediction and prediction.get('mock_data_used'):
            print("   Mock prediction: ✅ Generated successfully")
            print(f"   Score: {prediction.get('predicted_home_goals', 0):.2f} - {prediction.get('predicted_away_goals', 0):.2f}")
            mock_ok = True
        else:
            print("   Mock prediction: ❌ Failed - no prediction generated")
            mock_ok = False
    except Exception as e:
        print(f"   Mock prediction: ❌ Error: {e}")
        mock_ok = False
    
    # Test integration routing
    print("\n3️⃣ Testing integration routing:")
    try:
        # Test synthetic fixture routing
        synthetic_pred = pi.make_integrated_prediction(1000456, mock_data)
        if synthetic_pred and synthetic_pred.get('mock_data_used'):
            print("   Synthetic routing: ✅ Using mock data correctly")
            routing_ok = True
        else:
            print("   Synthetic routing: ❌ Not using mock data")
            routing_ok = False
    except Exception as e:
        print(f"   Synthetic routing: ❌ Error: {e}")
        routing_ok = False
    
    # Final result
    print("\n📊 Final Results:")
    all_passed = detection_ok and mock_ok and routing_ok
    
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("   ✅ Fixture detection working")
        print("   ✅ Mock prediction working")
        print("   ✅ Integration routing working")
        print("\n✅ Integration fix is VERIFIED and COMPLETE!")
    else:
        print("❌ Some tests failed")
        if not detection_ok:
            print("   ❌ Fixture detection issues")
        if not mock_ok:
            print("   ❌ Mock prediction issues")
        if not routing_ok:
            print("   ❌ Integration routing issues")
    
    return all_passed

if __name__ == "__main__":
    main()
