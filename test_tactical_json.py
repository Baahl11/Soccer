#!/usr/bin/env python3
"""
Script para verificar que las predicciones incluyan correctamente los datos de análisis táctico.
"""

import json
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tactical_json_integration():
    """Tests if tactical analysis is correctly included in prediction JSON."""
    try:
        # Import the necessary functions
        from tactical_integration import enrich_prediction_with_tactical_analysis
        
        # Create test data
        prediction = {
            "home_team": {"id": 123, "name": "Home Team"},
            "away_team": {"id": 456, "name": "Away Team"},
            "predicted_home_goals": 2.1,
            "predicted_away_goals": 1.3
        }
        
        home_matches = [{"score": {"home": 2, "away": 1}}]
        away_matches = [{"score": {"home": 1, "away": 2}}]
        
        # Enrich the prediction with tactical analysis
        enriched = enrich_prediction_with_tactical_analysis(prediction, home_matches, away_matches)
        
        # Check if tactical_analysis is in the result
        if "tactical_analysis" not in enriched:
            print("❌ ERROR: tactical_analysis not found in enriched prediction")
            return False
        
        # Save the result to a file for inspection
        with open("tactical_json_test.json", "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2, ensure_ascii=False)
        
        print(f"✅ SUCCESS: tactical_analysis found in enriched prediction")
        print(f"Results saved to tactical_json_test.json")
        return True
    
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n--- Testing tactical analysis JSON integration ---\n")
    test_tactical_json_integration()
