#!/usr/bin/env python3
"""
Final System Integration Test
Tests all key components after the fixes to ensure everything is working correctly.
"""

import sys
import os
import requests
import time
import json
from typing import Dict, Any

def test_api_connectivity():
    """Test if the API server is running and accessible."""
    try:
        response = requests.get("http://127.0.0.1:5000", timeout=5)
        return True
    except:
        return False

def test_confidence_calculation():
    """Test that confidence calculation is working without hardcoded values."""
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from app import calculate_dynamic_confidence
        
        # Create test prediction data
        test_prediction = {
            "home_team_id": 40,
            "away_team_id": 50,
            "league_id": 39,
            "fixture_id": 12345,
            "home_win_probability": 0.65,
            "away_win_probability": 0.20,
            "draw_probability": 0.15
        }
        
        # Test multiple calculations to ensure variation
        confidences = []
        for i in range(5):
            test_prediction["fixture_id"] = 12345 + i
            confidence = calculate_dynamic_confidence(test_prediction)
            confidences.append(confidence)
            
        # Check for variation (not all the same value)
        unique_values = len(set(confidences))
        has_variation = unique_values > 1
        
        # Check that values are in reasonable range
        in_range = all(0.4 <= c <= 0.9 for c in confidences)
        
        return {
            "success": has_variation and in_range,
            "confidences": confidences,
            "unique_values": unique_values,
            "in_range": in_range
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def test_footballapi_compatibility():
    """Test that FootballAPI class works correctly with the new system."""
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        from data import FootballAPI, ApiClient
        
        # Test that FootballAPI is properly aliased to ApiClient
        api = FootballAPI()
        
        # Check that the API has the required methods
        has_rate_limit = hasattr(api, '_respect_rate_limit')
        has_make_request = hasattr(api, '_make_request')
        has_team_stats = hasattr(api, 'get_team_statistics')
        
        return {
            "success": has_rate_limit and has_make_request and has_team_stats,
            "has_rate_limit": has_rate_limit,
            "has_make_request": has_make_request,
            "has_team_stats": has_team_stats,
            "type": str(type(api))
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def test_hardcoded_values():
    """Check for remaining hardcoded confidence values in key files."""
    # Focus on problematic patterns that indicate hardcoded confidence defaults
    hardcoded_patterns = [
        "confidence = 0.7",
        "confidence = 0.5", 
        "'confidence': 0.7",
        "'confidence': 0.5",
        "return 0.7  # Default",
        "return 0.5  # Default"
    ]
    
    files_to_check = [
        "app.py"
    ]
    
    found_hardcoded = []
    
    for filename in files_to_check:
        filepath = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                for pattern in hardcoded_patterns:
                    if pattern in content:
                        # Check if it's in a comment or string
                        lines = content.split('\n')
                        for i, line in enumerate(lines, 1):
                            if pattern in line and not line.strip().startswith('#'):
                                found_hardcoded.append(f"{filename}:{i}: {line.strip()}")
    
    return {
        "success": len(found_hardcoded) == 0,
        "found_hardcoded": found_hardcoded
    }

def test_imports():
    """Test that all critical imports work correctly."""
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Test core imports
        from data import FootballAPI, ApiClient
        from app import calculate_dynamic_confidence, get_or_calculate_confidence
        
        return {"success": True, "imports": "All critical imports successful"}
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def run_comprehensive_test():
    """Run all tests and provide a comprehensive report."""
    print("=" * 60)
    print("FINAL SYSTEM INTEGRATION TEST")
    print("=" * 60)
    
    tests = [
        ("API Connectivity", test_api_connectivity),
        ("Confidence Calculation", test_confidence_calculation),
        ("FootballAPI Compatibility", test_footballapi_compatibility),
        ("Hardcoded Values Check", test_hardcoded_values),
        ("Import Tests", test_imports)
    ]
    
    results = {}
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}...")
        try:
            result = test_func()
            if isinstance(result, dict):
                success = result.get("success", False)
                results[test_name] = result
            else:
                success = bool(result)
                results[test_name] = {"success": success}
                
            if success:
                print(f" ✅ PASS")
                passed += 1
            else:
                print(f" ❌ FAIL")
                if isinstance(result, dict) and "error" in result:
                    print(f"    Error: {result['error']}")
                    
        except Exception as e:
            print(f" ❌ ERROR: {e}")
            results[test_name] = {"success": False, "error": str(e)}
    
    print("\n" + "=" * 60)
    print(f"FINAL RESULT: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! System is working correctly.")
    else:
        print("⚠️  Some issues remain")
        
    print("=" * 60)
    
    # Print detailed results
    for test_name, result in results.items():
        if not result.get("success", False):
            print(f"\n❌ {test_name} Details:")
            for key, value in result.items():
                if key != "success":
                    print(f"  {key}: {value}")
    
    return passed == total

if __name__ == "__main__":
    run_comprehensive_test()
