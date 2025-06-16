#!/usr/bin/env python3
"""
Team Name Verification Tool
============================
This script verifies that team names are being extracted correctly from cached predictions.
It analyzes the cache to ensure that our fix for the "Unknown" team names issue is working.
"""

import pickle
import os
from pathlib import Path
from datetime import datetime
from collections import Counter

def analyze_team_names():
    """Analyze cached predictions to verify team name extraction"""
    
    cache_dir = Path('cache')
    cache_files = list(cache_dir.glob('*.cache'))
    
    print("🔍 TEAM NAME VERIFICATION ANALYSIS")
    print("=" * 50)
    print(f"📁 Cache Directory: {cache_dir}")
    print(f"📊 Total Cache Files: {len(cache_files)}")
    print()
    
    # Statistics
    valid_predictions = 0
    unknown_teams = 0
    valid_team_names = []
    leagues = Counter()
    confidence_levels = []
    
    # Sample matches for display
    sample_matches = []
    
    for cache_file in cache_files:
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            content = data.get('data')
            timestamp = data.get('timestamp', 0)
            age_hours = (datetime.now().timestamp() - timestamp) / 3600
            
            # Only analyze individual match predictions (not lists)
            if isinstance(content, dict) and 'home_team' in content and 'away_team' in content:
                valid_predictions += 1
                
                home_team = content.get('home_team', 'Unknown')
                away_team = content.get('away_team', 'Unknown')
                league = content.get('league', 'Unknown League')
                confidence = content.get('confidence', 0)
                pred_home = content.get('predicted_home_goals', 0)
                pred_away = content.get('predicted_away_goals', 0)
                
                # Track statistics
                if home_team == 'Unknown' or away_team == 'Unknown':
                    unknown_teams += 1
                else:
                    valid_team_names.extend([home_team, away_team])
                    leagues[league] += 1
                    confidence_levels.append(confidence)
                    
                    # Add to samples (first 10)
                    if len(sample_matches) < 10:
                        sample_matches.append({
                            'home': home_team,
                            'away': away_team,
                            'league': league,
                            'confidence': confidence,
                            'prediction': f"{pred_home:.1f} - {pred_away:.1f}",
                            'age_hours': age_hours
                        })
        
        except Exception as e:
            continue
    
    # Calculate statistics
    valid_team_rate = ((valid_predictions - unknown_teams) / valid_predictions * 100) if valid_predictions > 0 else 0
    avg_confidence = sum(confidence_levels) / len(confidence_levels) if confidence_levels else 0
    unique_teams = len(set(valid_team_names))
    
    print("📈 STATISTICS:")
    print("-" * 30)
    print(f"✅ Valid Predictions: {valid_predictions}")
    print(f"❌ Unknown Team Names: {unknown_teams}")
    print(f"📊 Valid Team Rate: {valid_team_rate:.1f}%")
    print(f"🏟️  Unique Teams: {unique_teams}")
    print(f"🏆 Leagues Covered: {len(leagues)}")
    print(f"📈 Average Confidence: {avg_confidence:.1f}%")
    print()
    
    if valid_team_rate >= 95:
        print("🎉 EXCELLENT: Team name extraction is working perfectly!")
    elif valid_team_rate >= 80:
        print("✅ GOOD: Team name extraction is working well")
    elif valid_team_rate >= 50:
        print("⚠️  PARTIAL: Some team names are missing")
    else:
        print("❌ POOR: Team name extraction needs fixing")
    
    print()
    print("🏆 SAMPLE MATCHES:")
    print("-" * 30)
    for i, match in enumerate(sample_matches, 1):
        print(f"{i:2d}. {match['home']} vs {match['away']}")
        print(f"    Liga: {match['league']}")
        print(f"    Predicción: {match['prediction']} | Confianza: {match['confidence']:.1f}%")
        print(f"    Cache: {match['age_hours']:.1f}h ago")
        print()
    
    if leagues:
        print("🌍 TOP LEAGUES:")
        print("-" * 30)
        for league, count in leagues.most_common(5):
            print(f"  {league}: {count} matches")
    
    print()
    print("💡 NEXT STEPS:")
    if valid_team_rate < 100:
        print("  - Check API response structure for matches with 'Unknown' teams")
        print("  - Verify fixtures endpoint is providing team data")
    else:
        print("  - Team name extraction is working perfectly!")
        print("  - System ready for production use")
    
    return {
        'valid_predictions': valid_predictions,
        'unknown_teams': unknown_teams,
        'valid_team_rate': valid_team_rate,
        'unique_teams': unique_teams,
        'leagues_count': len(leagues),
        'avg_confidence': avg_confidence
    }

if __name__ == "__main__":
    analyze_team_names()
