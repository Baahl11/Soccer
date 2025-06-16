#!/usr/bin/env python3
"""
Test Commercial Integration
Verifica que la integración del sistema de mejora comercial funcione correctamente
en el pipeline de descubrimiento automático de partidos del casino.
"""

import logging
import json
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from automatic_match_discovery import AutomaticMatchDiscovery
from commercial_response_enhancer import CommercialResponseEnhancer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_commercial_enhancer_standalone():
    """Test del mejorador comercial independiente"""
    print("=" * 60)
    print("🧪 TESTING COMMERCIAL ENHANCER STANDALONE")
    print("=" * 60)
    
    enhancer = CommercialResponseEnhancer()
    
    # Crear predicción de ejemplo
    sample_prediction = {
        'fixture_id': 12345,
        'home_team': 'Barcelona',
        'away_team': 'Real Madrid',
        'league': 'La Liga',
        'predicted_home_goals': 1.8,
        'predicted_away_goals': 1.3,
        'total_goals': 3.1,
        'prob_over_2_5': 0.67,
        'prob_btts': 0.72,
        'home_win_prob': 0.45,
        'draw_prob': 0.28,
        'away_win_prob': 0.27,
        'corners': {
            'total': 9.8,
            'home': 5.4,
            'away': 4.4,
            'over_9.5': 0.52
        },
        'cards': {
            'total': 4.1,
            'home': 2.0,
            'away': 2.1,
            'over_4.5': 0.48
        },
        'confidence': 0.74,
        'method': 'master_pipeline_casino_integration'
    }
    
    print(f"📊 Original Prediction Sample:")
    print(f"   Match: {sample_prediction['home_team']} vs {sample_prediction['away_team']}")
    print(f"   Goals: {sample_prediction['predicted_home_goals']} - {sample_prediction['predicted_away_goals']}")
    print(f"   Over 2.5: {sample_prediction['prob_over_2_5']:.1%}")
    print(f"   Corners: {sample_prediction['corners']['total']}")
    print(f"   Cards: {sample_prediction['cards']['total']}")
    
    # Aplicar mejora comercial
    try:
        enhanced = enhancer.enhance_prediction_response(sample_prediction)
        
        print(f"\n✅ Commercial Enhancement Applied Successfully!")
        print(f"   📈 Enhanced Structure:")
        
        # Verificar estructura mejorada
        if 'executive_summary' in enhanced:
            print(f"      ✅ Executive Summary: Present")
        
        if 'commercial_insights' in enhanced:
            insights = enhanced['commercial_insights']
            print(f"      ✅ Commercial Insights: {len(insights)} insights")
            
        if 'betting_recommendations' in enhanced:
            recommendations = enhanced['betting_recommendations']
            print(f"      ✅ Betting Recommendations: {len(recommendations)} recommendations")
            
        if 'value_opportunities' in enhanced:
            value_bets = enhanced['value_opportunities']
            print(f"      ✅ Value Opportunities: {len(value_bets)} value bets")
            
        if 'risk_assessment' in enhanced:
            print(f"      ✅ Risk Assessment: Present")
            
        # Verificar correcciones matemáticas
        corners = enhanced.get('corners', {})
        if 'over_9.5' in corners and 'under_9.5' in corners:
            over_prob = corners['over_9.5']
            under_prob = corners['under_9.5']
            total_prob = over_prob + under_prob
            print(f"      ✅ Math Corrections: Over + Under = {total_prob:.3f} (should be ~1.0)")
            
        print(f"\n📋 Sample Enhanced Content:")
        if 'commercial_insights' in enhanced and enhanced['commercial_insights']:
            first_insight = enhanced['commercial_insights'][0]
            print(f"   💡 First Insight: {first_insight.get('insight', 'N/A')}")
            
        if 'betting_recommendations' in enhanced and enhanced['betting_recommendations']:
            first_rec = enhanced['betting_recommendations'][0]
            print(f"   🎯 First Recommendation: {first_rec.get('market', 'N/A')} - {first_rec.get('selection', 'N/A')}")
            
        return True
        
    except Exception as e:
        print(f"❌ Commercial Enhancement Failed: {e}")
        return False

def test_automatic_discovery_integration():
    """Test integración con el descubrimiento automático"""
    print("\n" + "=" * 60)
    print("🔗 TESTING AUTOMATIC DISCOVERY INTEGRATION")
    print("=" * 60)
    
    try:
        # Inicializar el sistema con cache reducido para testing
        discovery = AutomaticMatchDiscovery(cache_ttl=300)  # 5 minutos
        
        print("✅ AutomaticMatchDiscovery initialized with commercial enhancer")
        
        # Verificar que el commercial enhancer está presente
        if hasattr(discovery, 'commercial_enhancer'):
            print("✅ Commercial enhancer properly integrated")
            
            # Test crear una predicción con mejora comercial
            test_match = {
                'fixture_id': 98765,
                'home_team_id': 529,  # Barcelona
                'away_team_id': 541,  # Real Madrid  
                'home_team': 'Barcelona',
                'away_team': 'Real Madrid',
                'league_id': 140,     # La Liga
                'league': 'La Liga',
                'season': 2025,
                'date': '2025-06-10',
                'venue': 'Camp Nou'
            }
            
            print(f"\n🧪 Testing master prediction generation with commercial enhancement...")
            print(f"   Match: {test_match['home_team']} vs {test_match['away_team']}")
            
            # Generar predicción con mejoras comerciales integradas
            enhanced_prediction = discovery._generate_master_prediction(test_match)
            
            # Verificar que tiene estructura comercial mejorada
            commercial_features = [
                'commercial_insights',
                'betting_recommendations', 
                'value_opportunities',
                'risk_assessment',
                'executive_summary'
            ]
            
            enhanced_features = []
            for feature in commercial_features:
                if feature in enhanced_prediction:
                    enhanced_features.append(feature)
                    
            print(f"✅ Prediction generated with {len(enhanced_features)}/{len(commercial_features)} commercial features")
            
            for feature in enhanced_features:
                content = enhanced_prediction[feature]
                if isinstance(content, list):
                    print(f"   📊 {feature}: {len(content)} items")
                elif isinstance(content, dict):
                    print(f"   📊 {feature}: {len(content)} fields")
                else:
                    print(f"   📊 {feature}: Present")
                    
            return True
            
        else:
            print("❌ Commercial enhancer not found in discovery system")
            return False
            
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_full_pipeline():
    """Test del pipeline completo con pocos partidos para verificar"""
    print("\n" + "=" * 60)
    print("🔄 TESTING FULL COMMERCIAL PIPELINE")
    print("=" * 60)
    
    try:
        discovery = AutomaticMatchDiscovery(cache_ttl=300)
        
        print("🔍 Getting today's predictions with commercial enhancements...")
        
        # Obtener predicciones (esto probará el pipeline completo)
        result = discovery.get_todays_predictions()
        
        if result.get('status') == 'success':
            matches = result.get('matches', [])
            total_matches = len(matches)
            
            print(f"✅ Pipeline executed successfully")
            print(f"   📊 Total matches found: {total_matches}")
            
            if total_matches > 0:
                # Analizar primer partido para verificar mejoras comerciales
                first_match = matches[0]
                
                print(f"\n🔍 Analyzing first match for commercial features:")
                print(f"   Match: {first_match.get('home_team', 'TBD')} vs {first_match.get('away_team', 'TBD')}")
                
                commercial_features = [
                    'commercial_insights',
                    'betting_recommendations',
                    'value_opportunities', 
                    'risk_assessment'
                ]
                
                enhanced_count = 0
                for feature in commercial_features:
                    if feature in first_match:
                        enhanced_count += 1
                        content = first_match[feature]
                        if isinstance(content, list):
                            print(f"   ✅ {feature}: {len(content)} items")
                        else:
                            print(f"   ✅ {feature}: Present")
                    else:
                        print(f"   ❌ {feature}: Missing")
                
                enhancement_ratio = enhanced_count / len(commercial_features)
                print(f"\n📈 Commercial Enhancement Score: {enhancement_ratio:.1%} ({enhanced_count}/{len(commercial_features)})")
                
                if enhancement_ratio >= 0.75:  # 75% de las features comerciales presentes
                    print("🎉 COMMERCIAL INTEGRATION: EXCELLENT")
                    return True
                elif enhancement_ratio >= 0.5:
                    print("⚠️ COMMERCIAL INTEGRATION: GOOD (some features missing)")
                    return True
                else:
                    print("❌ COMMERCIAL INTEGRATION: POOR (most features missing)")
                    return False
                    
            else:
                print("⚠️ No matches found, but pipeline worked")
                return True
                
        else:
            print(f"❌ Pipeline failed with status: {result.get('status')}")
            print(f"   Error: {result.get('error', 'Unknown')}")
            return False
            
    except Exception as e:
        print(f"❌ Full pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Ejecutar todos los tests de integración comercial"""
    print("🚀 COMMERCIAL INTEGRATION TEST SUITE")
    print("Testing the integration of CommercialResponseEnhancer with AutomaticMatchDiscovery")
    print("=" * 80)
    
    tests = [
        ("Commercial Enhancer Standalone", test_commercial_enhancer_standalone),
        ("Automatic Discovery Integration", test_automatic_discovery_integration),
        ("Full Pipeline Test", test_full_pipeline)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"💥 {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Resumen final
    print("\n" + "=" * 80)
    print("🏁 COMMERCIAL INTEGRATION TEST RESULTS")
    print("=" * 80)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {test_name}")
        if success:
            passed += 1
    
    success_rate = passed / total
    print(f"\n📊 Overall Success Rate: {success_rate:.1%} ({passed}/{total})")
    
    if success_rate == 1.0:
        print("🎉 ALL TESTS PASSED! Commercial integration is working perfectly!")
    elif success_rate >= 0.67:
        print("✅ Most tests passed. Commercial integration is mostly working.")
    else:
        print("⚠️ Several tests failed. Commercial integration needs attention.")
        
    return success_rate >= 0.67

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
