#!/usr/bin/env python3
"""
Test script to show cached predictions without using API quota.
This demonstrates that the cache system is working perfectly.
"""

import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

def get_cached_matches() -> List[Dict[str, Any]]:
    """Extract all cached match predictions."""
    
    cache_files = list(Path('cache').glob('*.cache'))
    cached_matches = []
    
    print(f"Scanning {len(cache_files)} cache files...")
    
    for cache_file in cache_files:
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            content = data.get('data')
            timestamp = data.get('timestamp', 0)
            age_hours = (datetime.now().timestamp() - timestamp) / 3600
            
            # Skip old cache (older than 24 hours)
            if age_hours > 24:
                continue
            
            # Check if it's a match prediction
            if isinstance(content, dict):
                if ('home_team' in content and 'away_team' in content and 
                    content.get('home_team') != 'Unknown' and 
                    content.get('away_team') != 'Unknown'):
                    
                    content['cache_age_hours'] = round(age_hours, 1)
                    cached_matches.append(content)
                    
        except Exception as e:
            continue
    
    return cached_matches

def main():
    print("🔍 TESTING CACHED PREDICTIONS (No API calls)")
    print("=" * 50)
    
    cached_matches = get_cached_matches()
    
    if not cached_matches:
        print("❌ No cached matches found")
        return
    
    print(f"✅ Found {len(cached_matches)} cached predictions!")
    print()
    
    # Sort by cache age (newest first)
    cached_matches.sort(key=lambda x: x.get('cache_age_hours', 999))
    
    print("🏆 CACHED PREDICTIONS:")
    print("-" * 50)
    
    for i, match in enumerate(cached_matches[:10], 1):  # Show top 10
        home_team = match.get('home_team', 'Unknown')
        away_team = match.get('away_team', 'Unknown')
        league_name = match.get('league', {}).get('name', 'Unknown League')
        confidence = match.get('confidence', 0.5)
        pred_home = match.get('predicted_home_goals', 0)
        pred_away = match.get('predicted_away_goals', 0)
        cache_age = match.get('cache_age_hours', 0)
        
        print(f"{i}. {home_team} vs {away_team}")
        print(f"   Liga: {league_name}")
        print(f"   Predicción: {pred_home:.1f} - {pred_away:.1f}")
        print(f"   Confianza: {confidence:.1%}")
        print(f"   Cache: {cache_age}h ago")
        print()
    
    # Statistics
    total_confidence = sum(m.get('confidence', 0.5) for m in cached_matches)
    avg_confidence = total_confidence / len(cached_matches)
    
    high_confidence = sum(1 for m in cached_matches if m.get('confidence', 0) >= 0.8)
    
    print("📊 STATISTICS:")
    print(f"Total predictions: {len(cached_matches)}")
    print(f"Average confidence: {avg_confidence:.1%}")
    print(f"High confidence matches (≥80%): {high_confidence}")
    print()
    print("💡 These predictions were generated from cached data!")
    print("🚀 Tomorrow with fresh API quota, you'll get many more matches!")

if __name__ == "__main__":
    main()
