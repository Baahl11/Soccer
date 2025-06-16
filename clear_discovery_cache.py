#!/usr/bin/env python3
"""
Clear Discovery Cache Tool
==========================
This script clears only the main discovery cache while preserving individual match prediction caches.
Use this when you want to force a fresh API call for match discovery.
"""

import os
import json
from hashlib import md5
from datetime import datetime
from pathlib import Path

def clear_discovery_cache():
    """Clear only the discovery cache, keeping individual match predictions"""
    
    cache_dir = Path('cache')
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Generate the same cache key as used in automatic_match_discovery.py
    cache_key = {
        'method': 'discover_matches',
        'date_range': '2_days', 
        'date': today
    }
    
    # Create hash the same way the cache system does
    cache_key_str = json.dumps(cache_key, sort_keys=True)
    cache_hash = md5(cache_key_str.encode()).hexdigest()[:8]
    
    discovery_cache_file = cache_dir / f'{cache_hash}.cache'
    
    print("🧹 CLEAR DISCOVERY CACHE")
    print("=" * 40)
    print(f"📅 Date: {today}")
    print(f"🔑 Cache Key: {cache_key}")
    print(f"#️⃣  Cache Hash: {cache_hash}")
    print(f"📁 Cache File: {discovery_cache_file}")
    print()
    
    if discovery_cache_file.exists():
        try:
            os.remove(discovery_cache_file)
            print("✅ Discovery cache cleared successfully!")
            print("💡 Next run will make fresh API calls for match discovery")
        except Exception as e:
            print(f"❌ Error clearing cache: {e}")
    else:
        print("ℹ️  Discovery cache not found (may already be cleared)")
    
    # Show remaining cache stats
    total_cache_files = len(list(cache_dir.glob('*.cache')))
    print(f"📊 Remaining cache files: {total_cache_files}")
    print()
    print("🚀 Ready for fresh match discovery!")

if __name__ == "__main__":
    clear_discovery_cache()
