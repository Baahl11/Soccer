#!/usr/bin/env python3
"""
Simple test to verify both synthetic and real fixture detection.
"""

from prediction_integration import is_synthetic_fixture

def test_fixture_detection():
    """Test fixture ID detection logic"""
    
    print("Testing fixture ID detection...")
    
    # Test synthetic fixture IDs
    synthetic_ids = [1000000, 1000001, 1500000, 2000000]
    for fixture_id in synthetic_ids:
        is_synthetic = is_synthetic_fixture(fixture_id)
        print(f"Fixture {fixture_id}: {'Synthetic' if is_synthetic else 'Real'} - {'✅' if is_synthetic else '❌'}")
    
    # Test real fixture IDs  
    real_ids = [123456, 999999, 500000, 1]
    for fixture_id in real_ids:
        is_synthetic = is_synthetic_fixture(fixture_id)
        print(f"Fixture {fixture_id}: {'Synthetic' if is_synthetic else 'Real'} - {'✅' if not is_synthetic else '❌'}")
    
    print("\n✅ Fixture detection test complete!")

if __name__ == "__main__":
    test_fixture_detection()
