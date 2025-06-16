#!/usr/bin/env python3
"""
Cache Clear and Debug Script for Soccer Prediction API

This script will:
1. Clear all cached predictions
2. Force fresh prediction generation
3. Debug the prediction pipeline
4. Identify why predictions return 0% values
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
import json

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clear_all_caches():
    """Clear all cache directories and files"""
    print("\n🧹 CLEARING ALL CACHES")
    print("=" * 50)
    
    cache_directories = [
        "cache",
        "odds_cache", 
        "data_cache",
        "weather_cache",
        "predictions_cache"
    ]
    
    cleared_count = 0
    
    for cache_dir in cache_directories:
        cache_path = Path(cache_dir)
        if cache_path.exists():
            try:
                if cache_path.is_dir():
                    shutil.rmtree(cache_path)
                    print(f"✅ Cleared directory: {cache_dir}")
                    cleared_count += 1
                else:
                    cache_path.unlink()
                    print(f"✅ Cleared file: {cache_dir}")
                    cleared_count += 1
            except Exception as e:
                print(f"❌ Error clearing {cache_dir}: {e}")
        else:
            print(f"⚠️ Cache directory not found: {cache_dir}")
    
    # Clear any individual cache files
    cache_files = [
        "fixtures_cache.json",
        "predictions_cache.json",
        "team_data_cache.json"
    ]
    
    for cache_file in cache_files:
        cache_path = Path(cache_file)
        if cache_path.exists():
            try:
                cache_path.unlink()
                print(f"✅ Cleared file: {cache_file}")
                cleared_count += 1
            except Exception as e:
                print(f"❌ Error clearing {cache_file}: {e}")
    
    print(f"\n✅ Total items cleared: {cleared_count}")
    return cleared_count

def clear_in_memory_cache():
    """Clear any in-memory cache stores"""
    print("\n🧠 CLEARING IN-MEMORY CACHES")
    print("=" * 50)
    
    try:
        # Clear the in-memory cache store from predictions.py
        from predictions import _cache_store
        if _cache_store:
            cache_count = len(_cache_store)
            _cache_store.clear()
            print(f"✅ Cleared {cache_count} items from predictions cache store")
        else:
            print("ℹ️ In-memory cache store is already empty")
            
    except ImportError as e:
        print(f"⚠️ Could not import predictions cache store: {e}")
    except Exception as e:
        print(f"❌ Error clearing in-memory cache: {e}")

def debug_prediction_generation():
    """Debug the prediction generation process"""
    print("\n🔍 DEBUGGING PREDICTION GENERATION")
    print("=" * 50)
    
    try:
        # Import the main prediction function
        from predictions import make_global_prediction
        
        # Test fixture ID (use a known fixture)
        test_fixture_id = 1208382
        
        print(f"Testing prediction generation for fixture {test_fixture_id}...")
        
        # Generate fresh prediction (cache should be empty now)
        prediction = make_global_prediction(test_fixture_id)
        
        print("\n📊 PREDICTION RESULT:")
        print(f"  Home goals: {prediction.get('predicted_home_goals', 'N/A')}")
        print(f"  Away goals: {prediction.get('predicted_away_goals', 'N/A')}")
        print(f"  Total goals: {prediction.get('total_goals', 'N/A')}")
        print(f"  Over 2.5: {prediction.get('prob_over_2_5', 'N/A')}")
        print(f"  BTTS: {prediction.get('prob_btts', 'N/A')}")
        
        # Check if probabilities are present
        if 'prob_home_win' in prediction:
            print(f"  Home win: {prediction['prob_home_win']}")
        if 'prob_draw' in prediction:
            print(f"  Draw: {prediction['prob_draw']}")
        if 'prob_away_win' in prediction:
            print(f"  Away win: {prediction['prob_away_win']}")
            
        # Check method used
        print(f"  Method: {prediction.get('method', 'Unknown')}")
        print(f"  Confidence: {prediction.get('confidence', 'N/A')}")
        
        # Determine if this is a valid prediction or fallback
        if prediction.get('predicted_home_goals', 0) == 0 and prediction.get('predicted_away_goals', 0) == 0:
            print("\n❌ ISSUE DETECTED: Prediction contains zero values!")
            return False
        else:
            print("\n✅ Prediction generated successfully with non-zero values")
            return True
            
    except Exception as e:
        print(f"❌ Error during prediction generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_predictions():
    """Test predictions for multiple fixtures to check variability"""
    print("\n🎯 TESTING MULTIPLE PREDICTIONS")
    print("=" * 50)
    
    test_fixture_ids = [1208382, 1208374, 1208373, 1208380, 1208375]
    
    try:
        from predictions import make_global_prediction
        
        results = []
        
        for fixture_id in test_fixture_ids:
            print(f"Testing fixture {fixture_id}...")
            
            try:
                prediction = make_global_prediction(fixture_id)
                
                result = {
                    'fixture_id': fixture_id,
                    'home_goals': prediction.get('predicted_home_goals', 0),
                    'away_goals': prediction.get('predicted_away_goals', 0),
                    'prob_home_win': prediction.get('prob_home_win', 0),
                    'prob_draw': prediction.get('prob_draw', 0),
                    'prob_away_win': prediction.get('prob_away_win', 0),
                    'method': prediction.get('method', 'unknown')
                }
                
                results.append(result)
                print(f"  ✅ {result['home_goals']:.1f} - {result['away_goals']:.1f} | {result['prob_home_win']:.1%} - {result['prob_draw']:.1%} - {result['prob_away_win']:.1%}")
                
            except Exception as e:
                print(f"  ❌ Error with fixture {fixture_id}: {e}")
        
        # Check for variability
        if len(results) > 1:
            print(f"\n📈 VARIABILITY CHECK:")
            
            # Check if all predictions are identical
            first_result = results[0]
            all_identical = all(
                r['home_goals'] == first_result['home_goals'] and 
                r['away_goals'] == first_result['away_goals'] 
                for r in results
            )
            
            if all_identical:
                print("❌ ALL PREDICTIONS ARE IDENTICAL! This indicates a problem with the prediction logic.")
                return False
            else:
                print("✅ Predictions show variability - this is expected.")
                return True
        
        return len(results) > 0
        
    except Exception as e:
        print(f"❌ Error testing multiple predictions: {e}")
        return False

def check_prediction_components():
    """Check individual prediction components"""
    print("\n🔧 CHECKING PREDICTION COMPONENTS")
    print("=" * 50)
    
    try:
        # Test team form retrieval
        print("Testing team form retrieval...")
        from team_form import get_team_form, get_head_to_head_analysis
        
        test_home_team = 33  # Manchester United
        test_away_team = 40  # Liverpool
        test_league = 39     # Premier League
        
        home_form = get_team_form(test_home_team, test_league, None)
        away_form = get_team_form(test_away_team, test_league, None)
        h2h = get_head_to_head_analysis(test_home_team, test_away_team)
        
        print(f"  Home form: {'✅ Found' if home_form else '❌ Empty'}")
        print(f"  Away form: {'✅ Found' if away_form else '❌ Empty'}")
        print(f"  H2H data: {'✅ Found' if h2h else '❌ Empty'}")
        
        if home_form:
            print(f"    Home goals/game: {home_form.get('goals_scored_pg', 'N/A')}")
        if away_form:
            print(f"    Away goals/game: {away_form.get('goals_scored_pg', 'N/A')}")
            
        # Test if form data is reasonable
        if home_form and away_form:
            home_gpg = home_form.get('goals_scored_pg', 0)
            away_gpg = away_form.get('goals_scored_pg', 0)
            
            if home_gpg > 0 and away_gpg > 0:
                print("  ✅ Form data contains reasonable values")
                return True
            else:
                print("  ❌ Form data contains zero/invalid values")
                return False
        else:
            print("  ❌ No form data available")
            return False
            
    except Exception as e:
        print(f"❌ Error checking prediction components: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to clear cache and debug predictions"""
    print("🏆 SOCCER PREDICTION CACHE CLEAR & DEBUG")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print()
    
    # Step 1: Clear all caches
    cleared_count = clear_all_caches()
    
    # Step 2: Clear in-memory caches
    clear_in_memory_cache()
    
    # Step 3: Test prediction generation
    prediction_works = debug_prediction_generation()
    
    # Step 4: Test multiple predictions for variability
    variability_works = test_multiple_predictions()
    
    # Step 5: Check prediction components
    components_work = check_prediction_components()
    
    # Summary
    print("\n📋 SUMMARY")
    print("=" * 30)
    print(f"Cache cleared: ✅ ({cleared_count} items)")
    print(f"Prediction generation: {'✅' if prediction_works else '❌'}")
    print(f"Prediction variability: {'✅' if variability_works else '❌'}")
    print(f"Prediction components: {'✅' if components_work else '❌'}")
    
    if all([prediction_works, variability_works, components_work]):
        print("\n🎉 SUCCESS: Cache cleared and predictions are working!")
        print("Try making API requests again - they should now show fresh predictions.")
    else:
        print("\n⚠️ ISSUES DETECTED: The prediction system may have underlying problems.")
        print("Recommendations:")
        if not prediction_works:
            print("  - Check the make_global_prediction function")
        if not variability_works:
            print("  - Check xG calculation logic for team-specific values")
        if not components_work:
            print("  - Check team form data retrieval")
            
    print(f"\nScript completed at: {datetime.now()}")

if __name__ == "__main__":
    main()
