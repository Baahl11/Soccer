# demonstration_fixture_statistics.py
"""
Demonstration of the enhanced Master Pipeline with Fixture Statistics integration
"""
import logging
import sys
import os
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def demonstrate_fixture_statistics():
    """Demonstrate the fixture statistics integration with real examples."""
    
    try:
        from master_prediction_pipeline_simple import generate_master_prediction
        from fixture_statistics_analyzer import FixtureStatisticsAnalyzer
        
        print("🏆 FIXTURE STATISTICS INTEGRATION DEMONSTRATION")
        print("=" * 60)
        
        # Test cases with different team matchups
        test_cases = [
            {
                'name': 'Premier League: Manchester City vs Arsenal',
                'fixture_id': 12001,
                'home_team_id': 50,
                'away_team_id': 42,
                'league_id': 39
            },
            {
                'name': 'Premier League: Liverpool vs Chelsea',
                'fixture_id': 12002,
                'home_team_id': 40,
                'away_team_id': 49,
                'league_id': 39
            },
            {
                'name': 'La Liga: Real Madrid vs Barcelona',
                'fixture_id': 12003,
                'home_team_id': 541,
                'away_team_id': 529,
                'league_id': 140
            }
        ]
        
        analyzer = FixtureStatisticsAnalyzer()
        
        for i, case in enumerate(test_cases, 1):
            print(f"\n📊 TEST CASE {i}: {case['name']}")
            print("-" * 50)
            
            # Generate prediction with fixture statistics
            prediction = generate_master_prediction(
                fixture_id=case['fixture_id'],
                home_team_id=case['home_team_id'],
                away_team_id=case['away_team_id'],
                league_id=case['league_id']
            )
            
            # Extract key information
            predictions = prediction['predictions']
            confidence = prediction['confidence_scores']
            components = prediction.get('component_analyses', {})
            
            # Display core predictions
            print(f"🎯 PREDICTIONS:")
            print(f"   Goals: {predictions['predicted_home_goals']:.2f} - {predictions['predicted_away_goals']:.2f}")
            print(f"   Total Goals: {predictions['predicted_total_goals']:.2f}")
            print(f"   Probabilities: H:{predictions['home_win_prob']:.3f} D:{predictions['draw_prob']:.3f} A:{predictions['away_win_prob']:.3f}")
            
            # Display confidence information
            print(f"\n📈 CONFIDENCE ANALYSIS:")
            print(f"   Overall Confidence: {confidence['overall_confidence']:.1%}")
            print(f"   Components Active: {confidence['components_active']}")
            print(f"   Reliability: {prediction.get('quality_indicators', {}).get('confidence_reliability', 'unknown')}")
            
            # Display enhancement information
            enhancements = predictions.get('enhancements_applied', [])
            print(f"\n🔧 ENHANCEMENTS APPLIED:")
            for enhancement in enhancements:
                status = "✅" if enhancement != "fixture_statistics_analysis" else "🆕"
                print(f"   {status} {enhancement.replace('_', ' ').title()}")
            
            # Display fixture statistics component if available
            if 'fixture_statistics' in components:
                fixture_stats = components['fixture_statistics']
                print(f"\n📊 FIXTURE STATISTICS ANALYSIS:")
                print(f"   Status: {'✅ Active' if fixture_stats.get('available') else '❌ Inactive'}")
                
                if fixture_stats.get('available'):
                    print(f"   Confidence Boost: +{fixture_stats.get('confidence_boost', 0):.3f}")
                    
                    modifiers = fixture_stats.get('goal_modifiers', {})
                    if modifiers:
                        print(f"   Goal Modifiers: Home x{modifiers.get('home', 1):.3f}, Away x{modifiers.get('away', 1):.3f}")
                    
                    comparative = fixture_stats.get('comparative_analysis', {})
                    if comparative:
                        print(f"   Key Advantages:")
                        for advantage, value in comparative.items():
                            if abs(value) > 0.1:  # Only show significant advantages
                                team = "Home" if value > 0 else "Away"
                                strength = "Strong" if abs(value) > 0.3 else "Moderate" if abs(value) > 0.1 else "Slight"
                                print(f"     - {advantage.replace('_', ' ').title()}: {strength} {team} advantage ({value:+.3f})")
            
            # Demonstrate standalone fixture statistics analysis
            print(f"\n🔬 DETAILED STATISTICAL ANALYSIS:")
            detailed_stats = analyzer.analyze_fixture_statistics(
                case['home_team_id'], 
                case['away_team_id'], 
                case['league_id']
            )
            
            home_quality = detailed_stats['home']['overall_quality']
            away_quality = detailed_stats['away']['overall_quality']
            overall_advantage = detailed_stats['comparative']['overall_advantage']
            
            print(f"   Home Team Quality Score: {home_quality:.3f}")
            print(f"   Away Team Quality Score: {away_quality:.3f}")
            print(f"   Overall Advantage: {overall_advantage:+.3f} ({'Home' if overall_advantage > 0 else 'Away' if overall_advantage < 0 else 'Balanced'})")
            
            # Performance indicators
            accuracy_proj = prediction.get('quality_indicators', {}).get('accuracy_projection', {})
            if accuracy_proj:
                baseline = accuracy_proj.get('baseline', 0.75)
                enhanced = accuracy_proj.get('with_enhancements', baseline)
                improvement = accuracy_proj.get('improvement_percentage', 0)
                
                print(f"\n🎯 PERFORMANCE PROJECTION:")
                print(f"   Baseline Accuracy: {baseline:.1%}")
                print(f"   Enhanced Accuracy: {enhanced:.1%}")
                print(f"   Improvement: +{improvement:.1f}%")
        
        print(f"\n" + "=" * 60)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("\nKey Benefits Demonstrated:")
        print("• Enhanced prediction accuracy through fixture statistics")
        print("• Improved confidence scoring with statistical clarity")
        print("• Comprehensive team analysis and comparative advantages")
        print("• Seamless integration with existing Master Pipeline")
        print("• Transparent component reporting and analysis")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def demonstrate_statistical_impact():
    """Demonstrate the statistical impact of fixture analysis."""
    
    try:
        from fixture_statistics_analyzer import enhance_goal_predictions, enhance_match_probabilities
        
        print(f"\n🧮 STATISTICAL IMPACT DEMONSTRATION")
        print("=" * 60)
        
        # Sample scenarios
        scenarios = [
            {
                'name': 'Balanced Teams',
                'base_home_goals': 1.5,
                'base_away_goals': 1.3,
                'base_probs': {'home': 0.40, 'away': 0.35, 'draw': 0.25},
                'comparative_advantage': 0.0  # Balanced
            },
            {
                'name': 'Strong Home Advantage',
                'base_home_goals': 1.8,
                'base_away_goals': 1.0,
                'base_probs': {'home': 0.50, 'away': 0.25, 'draw': 0.25},
                'comparative_advantage': 0.3  # Home advantage
            },
            {
                'name': 'Away Team Dominance',
                'base_home_goals': 1.0,
                'base_away_goals': 2.2,
                'base_probs': {'home': 0.25, 'away': 0.55, 'draw': 0.20},
                'comparative_advantage': -0.4  # Away advantage
            }
        ]
        
        for scenario in scenarios:
            print(f"\n📈 SCENARIO: {scenario['name']}")
            print("-" * 40)
            
            # Create mock fixture statistics data
            mock_stats = {
                'goal_expectation_modifiers': {
                    'home': 1.0 + scenario['comparative_advantage'] * 0.2,
                    'away': 1.0 - scenario['comparative_advantage'] * 0.2
                },
                'probability_adjustments': {
                    'home': scenario['comparative_advantage'] * 0.1,
                    'away': -scenario['comparative_advantage'] * 0.1,
                    'draw': -abs(scenario['comparative_advantage']) * 0.05
                }
            }
            
            # Show original predictions
            print(f"Original Goals: {scenario['base_home_goals']:.2f} - {scenario['base_away_goals']:.2f}")
            print(f"Original Probs: H:{scenario['base_probs']['home']:.3f} D:{scenario['base_probs']['draw']:.3f} A:{scenario['base_probs']['away']:.3f}")
            
            # Apply enhancements
            enhanced_home, enhanced_away = enhance_goal_predictions(
                scenario['base_home_goals'], 
                scenario['base_away_goals'], 
                mock_stats
            )
            
            enhanced_probs = enhance_match_probabilities(scenario['base_probs'], mock_stats)
            
            # Show enhanced predictions
            print(f"Enhanced Goals: {enhanced_home:.2f} - {enhanced_away:.2f}")
            print(f"Enhanced Probs: H:{enhanced_probs['home']:.3f} D:{enhanced_probs['draw']:.3f} A:{enhanced_probs['away']:.3f}")
            
            # Calculate changes
            goal_change_home = enhanced_home - scenario['base_home_goals']
            goal_change_away = enhanced_away - scenario['base_away_goals']
            prob_change_home = enhanced_probs['home'] - scenario['base_probs']['home']
            
            print(f"Changes: Goals ({goal_change_home:+.2f}, {goal_change_away:+.2f}), Home Win Prob ({prob_change_home:+.3f})")
        
        print(f"\n✅ Statistical impact demonstration completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Statistical impact demonstration failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 MASTER PIPELINE WITH FIXTURE STATISTICS")
    print("Real-time Football Prediction Enhancement System")
    print("=" * 60)
    
    success1 = demonstrate_fixture_statistics()
    success2 = demonstrate_statistical_impact()
    
    if success1 and success2:
        print(f"\n🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("\nThe Master Pipeline now includes advanced fixture statistics")
        print("analysis for enhanced prediction accuracy and confidence.")
    else:
        print(f"\n⚠️ Some demonstrations encountered issues.")
    
    input("\nPress Enter to exit...")
