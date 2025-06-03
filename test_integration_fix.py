#!/usr/bin/env python3
"""
Test script to verify the integration fix between ELO workflow and prediction integration.
"""

from elo_prediction_workflow import ELOEnhancedPredictionWorkflow
import logging

def test_integration_fix():
    """Test the integration fix"""
    
    # Set up logging to see the integration flow
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("Starting integration fix test...")
    
    # Create workflow instance
    workflow = ELOEnhancedPredictionWorkflow()
    
    # Test with Premier League (ID 39)
    logger.info("Testing ELO workflow integration...")
    fixtures = workflow.get_upcoming_matches(39, days_ahead=2)
    logger.info(f"Generated {len(fixtures)} fixtures")
    
    # Make predictions for first fixture
    if fixtures:
        test_fixture = fixtures[0]
        logger.info(f"Testing prediction for: {test_fixture['home_team_name']} vs {test_fixture['away_team_name']}")
        logger.info(f"Fixture ID: {test_fixture['fixture_id']}")
        
        predictions = workflow.make_predictions_for_matches([test_fixture])
        
        if predictions:
            prediction = predictions[0]
            logger.info("✅ Prediction successful!")
            logger.info(f"Data source: {prediction.get('data_source', 'unknown')}")
            logger.info(f"Mock data used: {prediction.get('mock_data_used', False)}")
            logger.info(f"Predicted goals: {prediction.get('predicted_home_goals', 0):.2f} - {prediction.get('predicted_away_goals', 0):.2f}")
            logger.info(f"Prediction method: {prediction.get('method', 'unknown')}")
            logger.info(f"Total goals: {prediction.get('total_goals', 0):.2f}")
            logger.info(f"1X2 probabilities: {prediction.get('prob_1', 0):.2f} / {prediction.get('prob_X', 0):.2f} / {prediction.get('prob_2', 0):.2f}")
            
            # Check if integration features are present
            integration_features = []
            if 'elo_ratings' in prediction:
                integration_features.append("ELO ratings")
            if 'tactical_analysis' in prediction:
                integration_features.append("Tactical analysis")
            if 'weather_adjustment' in prediction:
                integration_features.append("Weather adjustments")
            if 'adjustments_applied' in prediction:
                integration_features.append(f"Adjustments: {prediction['adjustments_applied']}")
            
            if integration_features:
                logger.info(f"Integration features present: {', '.join(integration_features)}")
            else:
                logger.warning("No integration features found in prediction")
                
            return True
        else:
            logger.error("❌ No predictions generated")
            return False
    else:
        logger.error("❌ No fixtures generated")
        return False

if __name__ == "__main__":
    success = test_integration_fix()
    if success:
        print("\n🎉 Integration fix test PASSED!")
    else:
        print("\n❌ Integration fix test FAILED!")
