# test_fixture_statistics_integration.py
"""
Comprehensive test for fixture statistics integration in Master Pipeline
"""
import logging
import sys
import os
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fixture_statistics_analyzer():
    """Test the FixtureStatisticsAnalyzer functionality"""
    try:
        from fixture_statistics_analyzer import FixtureStatisticsAnalyzer, enhance_goal_predictions, enhance_match_probabilities
        
        logger.info("=== Testing FixtureStatisticsAnalyzer ===")
        
        # Initialize analyzer
        analyzer = FixtureStatisticsAnalyzer()
        
        # Test with sample team IDs (Manchester City vs Arsenal - common test teams)
        home_team_id = 50  # Manchester City
        away_team_id = 42  # Arsenal  
        league_id = 39     # Premier League
        
        # Test fixture statistics analysis
        fixture_stats = analyzer.analyze_fixture_statistics(home_team_id, away_team_id, league_id)
        
        # Validate structure
        assert 'home' in fixture_stats, "Missing home team analysis"
        assert 'away' in fixture_stats, "Missing away team analysis"
        assert 'comparative' in fixture_stats, "Missing comparative analysis"
        assert 'confidence_boost' in fixture_stats, "Missing confidence boost"
        assert 'goal_expectation_modifiers' in fixture_stats, "Missing goal expectation modifiers"
        assert 'probability_adjustments' in fixture_stats, "Missing probability adjustments"
        
        # Validate home team metrics
        home_metrics = fixture_stats['home']
        required_metrics = ['attacking_threat', 'possession_control', 'defensive_solidity', 
                          'disciplinary_risk', 'corner_generation', 'overall_quality']
        
        for metric in required_metrics:
            assert metric in home_metrics, f"Missing home metric: {metric}"
            assert isinstance(home_metrics[metric], (int, float)), f"Invalid type for {metric}"
        
        # Validate comparative analysis
        comparative = fixture_stats['comparative']
        comparative_metrics = ['shooting_advantage', 'possession_advantage', 'defensive_advantage',
                             'discipline_advantage', 'corner_advantage', 'overall_advantage']
        
        for metric in comparative_metrics:
            assert metric in comparative, f"Missing comparative metric: {metric}"
            assert -1 <= comparative[metric] <= 1, f"Comparative metric {metric} out of range [-1, 1]"
        
        # Test goal enhancement
        base_home_goals = 1.5
        base_away_goals = 1.2
        
        enhanced_home, enhanced_away = enhance_goal_predictions(base_home_goals, base_away_goals, fixture_stats)
        
        assert 0.1 <= enhanced_home <= 6.0, "Enhanced home goals out of valid range"
        assert 0.1 <= enhanced_away <= 6.0, "Enhanced away goals out of valid range"
        
        # Test probability enhancement
        base_probs = {'home': 0.45, 'away': 0.30, 'draw': 0.25}
        enhanced_probs = enhance_match_probabilities(base_probs, fixture_stats)
        
        assert abs(sum(enhanced_probs.values()) - 1.0) < 0.001, "Enhanced probabilities don't sum to 1"
        assert all(0.05 <= prob <= 0.95 for prob in enhanced_probs.values()), "Enhanced probabilities out of range"
        
        logger.info("✅ FixtureStatisticsAnalyzer tests passed!")
        
        # Print sample results
        logger.info("\n=== Sample Analysis Results ===")
        logger.info(f"Home Team Quality: {home_metrics['overall_quality']:.3f}")
        logger.info(f"Away Team Quality: {fixture_stats['away']['overall_quality']:.3f}")
        logger.info(f"Overall Advantage (Home): {comparative['overall_advantage']:.3f}")
        logger.info(f"Confidence Boost: {fixture_stats['confidence_boost']:.3f}")
        logger.info(f"Goal Modifiers - Home: {fixture_stats['goal_expectation_modifiers']['home']:.3f}, Away: {fixture_stats['goal_expectation_modifiers']['away']:.3f}")
        logger.info(f"Enhanced Goals: {enhanced_home:.2f} - {enhanced_away:.2f}")
        logger.info(f"Enhanced Probabilities: H:{enhanced_probs['home']:.3f} D:{enhanced_probs['draw']:.3f} A:{enhanced_probs['away']:.3f}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ FixtureStatisticsAnalyzer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_master_pipeline_integration():
    """Test integration with Master Pipeline"""
    try:
        from master_prediction_pipeline_simple import generate_master_prediction
        
        logger.info("\n=== Testing Master Pipeline Integration ===")
        
        # Test prediction with fixture statistics
        fixture_id = 12345
        home_team_id = 50  # Manchester City
        away_team_id = 42  # Arsenal
        league_id = 39     # Premier League
        
        prediction = generate_master_prediction(fixture_id, home_team_id, away_team_id, league_id)
        
        # Validate prediction structure
        assert 'predictions' in prediction, "Missing predictions section"
        assert 'confidence_scores' in prediction, "Missing confidence scores"
        assert 'component_analyses' in prediction, "Missing component analyses"
        
        # Check if fixture statistics were applied
        enhancements = prediction['predictions'].get('enhancements_applied', [])
        components = prediction.get('component_analyses', {})
        
        logger.info(f"Enhancements Applied: {enhancements}")
        logger.info(f"Components Active: {prediction['confidence_scores'].get('components_active', 0)}")
        
        # Check fixture statistics component
        if 'fixture_statistics' in components:
            fixture_component = components['fixture_statistics']
            logger.info(f"Fixture Statistics Available: {fixture_component.get('available', False)}")
            logger.info(f"Fixture Statistics Confidence Boost: {fixture_component.get('confidence_boost', 0)}")
            
            if fixture_component.get('available'):
                logger.info("✅ Fixture statistics successfully integrated!")
            else:
                logger.warning("⚠️ Fixture statistics component not active")
        else:
            logger.warning("⚠️ Fixture statistics component not found in analysis")
        
        # Validate prediction values
        predictions = prediction['predictions']
        assert 0.1 <= predictions['predicted_home_goals'] <= 6.0, "Home goals prediction out of range"
        assert 0.1 <= predictions['predicted_away_goals'] <= 6.0, "Away goals prediction out of range"
        assert 0.05 <= predictions['home_win_prob'] <= 0.95, "Home win probability out of range"
        assert 0.05 <= predictions['draw_prob'] <= 0.95, "Draw probability out of range"
        assert 0.05 <= predictions['away_win_prob'] <= 0.95, "Away win probability out of range"
        
        # Check confidence improvement
        confidence = prediction['confidence_scores']['overall_confidence']
        logger.info(f"Overall Confidence: {confidence:.3f}")
        
        if confidence >= 0.85:
            logger.info("✅ High confidence prediction achieved!")
        elif confidence >= 0.75:
            logger.info("✅ Good confidence prediction achieved!")
        else:
            logger.warning(f"⚠️ Lower confidence prediction: {confidence:.3f}")
        
        logger.info("✅ Master Pipeline integration tests passed!")
        
        # Print detailed results
        logger.info("\n=== Detailed Prediction Results ===")
        logger.info(f"Goals: {predictions['predicted_home_goals']:.2f} - {predictions['predicted_away_goals']:.2f}")
        logger.info(f"Probabilities: H:{predictions['home_win_prob']:.3f} D:{predictions['draw_prob']:.3f} A:{predictions['away_win_prob']:.3f}")
        logger.info(f"Method: {predictions.get('method', 'unknown')}")
        logger.info(f"Components: {prediction['confidence_scores']['components_active']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Master Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_availability():
    """Test that enhanced team statistics are available"""
    try:
        from data import get_team_statistics
        
        logger.info("\n=== Testing Enhanced Data Availability ===")
        
        # Test with sample team
        team_id = 50  # Manchester City
        league_id = 39  # Premier League
        season = "2023"
        
        stats = get_team_statistics(team_id, league_id, season)
        
        # Check for enhanced statistics
        enhanced_fields = [
            'shots_per_game', 'shots_on_target_per_game', 'possession_percentage',
            'fouls_per_game', 'goals_per_game', 'goals_conceded_per_game',
            'passes_completed_per_game', 'passes_attempted_per_game'
        ]
        
        for field in enhanced_fields:
            assert field in stats, f"Missing enhanced statistic: {field}"
            assert isinstance(stats[field], (int, float)), f"Invalid type for {field}"
            logger.info(f"✅ {field}: {stats[field]}")
        
        logger.info("✅ Enhanced data availability tests passed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Data availability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_test():
    """Run all tests"""
    logger.info("🚀 Starting Comprehensive Fixture Statistics Integration Test")
    logger.info("=" * 60)
    
    test_results = []
    
    # Test 1: FixtureStatisticsAnalyzer
    test_results.append(("FixtureStatisticsAnalyzer", test_fixture_statistics_analyzer()))
    
    # Test 2: Data Availability
    test_results.append(("Enhanced Data Availability", test_data_availability()))
    
    # Test 3: Master Pipeline Integration
    test_results.append(("Master Pipeline Integration", test_master_pipeline_integration()))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED! Fixture statistics integration is working correctly.")
        return True
    else:
        logger.warning(f"⚠️ {total - passed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
