"""
Test script to verify the JSON structure of the /api/upcoming_predictions endpoint.
This script checks that:
1. tactical_analysis is at the root level (not in additional_data)
2. odds_analysis is at the root level
3. Field naming for stats (corners, cards, fouls) is consistent
"""

import requests
import json
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('api_structure_test.log')
    ]
)

logger = logging.getLogger(__name__)

def test_api_structure(base_url="http://localhost:5000"):
    """Test the API structure of the /api/upcoming_predictions endpoint."""
    
    # Define the endpoint URL with parameters
    endpoint = f"{base_url}/api/upcoming_predictions"
    params = {
        "league_id": 619,  # Use the league ID from the prompt
        "season": 2025,    # Use the season from the prompt
        "include_additional_data": "true",
        "limit": 1         # Request just one prediction for simplicity
    }
    
    try:
        logger.info(f"Testing endpoint: {endpoint}")
        logger.info(f"Parameters: {params}")
        
        # Make the API request
        response = requests.get(endpoint, params=params)
        
        # Check if request was successful
        if response.status_code != 200:
            logger.error(f"API request failed with status code {response.status_code}: {response.text}")
            return False
        
        # Parse the response
        data = response.json()
        
        # Save the raw response for inspection
        with open("api_response.json", "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Response saved to api_response.json")
        
        # Check if the response has the expected structure
        if "match_predictions" not in data:
            logger.error("Response missing 'match_predictions' field")
            return False
        
        predictions = data["match_predictions"]
        if not predictions:
            logger.error("No predictions returned")
            return False
        
        # For each prediction, check the structure
        for i, pred in enumerate(predictions):
            logger.info(f"Checking prediction {i+1}: {pred.get('home_team', 'Unknown')} vs {pred.get('away_team', 'Unknown')}")
            
            # Check 1: tactical_analysis should be at root level
            if "tactical_analysis" not in pred:
                logger.error(f"Prediction {i+1}: Missing tactical_analysis at root level")
                
                # Check if tactical_analysis is in additional_data
                if "additional_data" in pred and "tactical_analysis" in pred["additional_data"]:
                    logger.error(f"Prediction {i+1}: tactical_analysis found in additional_data instead of root level")
                    return False
            else:
                logger.info(f"Prediction {i+1}: tactical_analysis found at root level ✓")
            
            # Check 2: odds_analysis should be at root level
            if "odds_analysis" not in pred:
                logger.error(f"Prediction {i+1}: Missing odds_analysis at root level")
                
                # Check if odds_analysis is in additional_data
                if "additional_data" in pred and "odds_analysis" in pred["additional_data"]:
                    logger.error(f"Prediction {i+1}: odds_analysis found in additional_data instead of root level")
                    return False
            else:
                logger.info(f"Prediction {i+1}: odds_analysis found at root level ✓")
            
            # Check 3: Field naming for stats should be consistent
            if "corners" in pred:
                corners = pred["corners"]
                expected_keys = ["predicted_total", "home_corners", "away_corners"]
                missing_keys = [k for k in expected_keys if k not in corners]
                if missing_keys:
                    logger.error(f"Prediction {i+1}: corners missing fields: {missing_keys}")
                    logger.error(f"Actual corners structure: {corners}")
                    return False
                else:
                    logger.info(f"Prediction {i+1}: corners structure is correct ✓")
            
            if "cards" in pred:
                cards = pred["cards"]
                expected_keys = ["predicted_total", "home_cards", "away_cards"]
                missing_keys = [k for k in expected_keys if k not in cards]
                if missing_keys:
                    logger.error(f"Prediction {i+1}: cards missing fields: {missing_keys}")
                    logger.error(f"Actual cards structure: {cards}")
                    return False
                else:
                    logger.info(f"Prediction {i+1}: cards structure is correct ✓")
            
            if "fouls" in pred:
                fouls = pred["fouls"]
                expected_keys = ["predicted_total", "home_fouls", "away_fouls"]
                missing_keys = [k for k in expected_keys if k not in fouls]
                if missing_keys:
                    logger.error(f"Prediction {i+1}: fouls missing fields: {missing_keys}")
                    logger.error(f"Actual fouls structure: {fouls}")
                    return False
                else:
                    logger.info(f"Prediction {i+1}: fouls structure is correct ✓")
        
        logger.info("All structure checks passed! ✓")
        return True
    
    except Exception as e:
        logger.exception(f"Error testing API structure: {e}")
        return False

if __name__ == "__main__":
    success = test_api_structure()
    sys.exit(0 if success else 1)
