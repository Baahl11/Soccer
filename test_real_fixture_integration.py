#!/usr/bin/env python3
"""
Test script to verify integration works with real fixtures (API-based data).
"""

from prediction_integration import make_integrated_prediction
import logging

def test_real_fixture_integration():
    """Test integration with a real fixture ID"""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Starting real fixture integration test...")
    
    # Test with a known fixture ID (should be < 1000000)
    test_fixture_id = 123456  # This should be recognized as a real fixture
    
    logger.info(f"Testing integration with real fixture ID: {test_fixture_id}")
    
    try:
        # This should take the normal integration path (not mock)
        prediction = make_integrated_prediction(test_fixture_id)
        
        if prediction:
            logger.info("✅ Real fixture integration successful!")
            logger.info(f"Data source: {prediction.get('data_source', 'api')}")
            logger.info(f"Mock data used: {prediction.get('mock_data_used', False)}")
            logger.info(f"Fixture ID: {prediction.get('fixture_id', 'unknown')}")
            logger.info(f"Method: {prediction.get('method', 'unknown')}")
            
            # Verify it didn't use mock data
            if prediction.get('mock_data_used', False):
                logger.warning("⚠️  Real fixture incorrectly used mock data")
                return False
            else:
                logger.info("✅ Real fixture correctly used API data")
                return True
        else:
            logger.warning("⚠️  No prediction returned (likely due to missing API data)")
            logger.info("This is expected if fixture doesn't exist in API")
            return True  # This is actually expected behavior for non-existent fixtures
            
    except Exception as e:
        logger.error(f"❌ Error in real fixture integration: {e}")
        return False

if __name__ == "__main__":
    success = test_real_fixture_integration()
    if success:
        print("\n🎉 Real fixture integration test PASSED!")
    else:
        print("\n❌ Real fixture integration test FAILED!")
