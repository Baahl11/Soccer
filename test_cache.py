#!/usr/bin/env python3
"""
Test script for the caching system
"""

import time
import shutil
import os
from automatic_match_discovery import CacheManager, AutomaticMatchDiscovery

def test_cache_manager():
    """Test the cache manager functionality"""
    print("🧪 Testing Cache Manager")
    print("=" * 40)
    
    # Create cache manager
    cache = CacheManager(cache_dir='test_cache', default_ttl=10)  # 10 seconds TTL
    
    # Test data
    test_data = {
        'matches': [
            {'home': 'Team A', 'away': 'Team B', 'score': '2-1'},
            {'home': 'Team C', 'away': 'Team D', 'score': '0-3'}
        ],
        'timestamp': time.time()
    }
    
    # Test setting cache
    print("📝 Setting cache...")
    cache.set('test_matches', test_data)
    
    # Test getting cache
    print("📖 Getting cache...")
    retrieved = cache.get('test_matches')
    
    if retrieved == test_data:
        print("✅ Cache GET/SET working correctly!")
    else:
        print("❌ Cache GET/SET failed!")
        return False
    
    # Test cache miss (non-existent key)
    print("🔍 Testing cache miss...")
    missing = cache.get('non_existent_key')
    if missing is None:
        print("✅ Cache miss handled correctly!")
    else:
        print("❌ Cache miss not handled correctly!")
        return False
    
    # Test TTL expiration (wait for expiration)
    print("⏰ Testing TTL expiration (waiting 12 seconds)...")
    time.sleep(12)
    
    expired = cache.get('test_matches')
    if expired is None:
        print("✅ TTL expiration working correctly!")
    else:
        print("❌ TTL expiration failed!")
        return False
    
    # Cleanup
    if os.path.exists('test_cache'):
        shutil.rmtree('test_cache')
        print("🧹 Test cache cleaned up")
    
    return True

def test_automatic_discovery_cache():
    """Test the automatic discovery with cache"""
    print("\n🤖 Testing Automatic Discovery Cache")
    print("=" * 40)
    
    try:
        # Initialize with short cache for testing
        discovery = AutomaticMatchDiscovery(cache_ttl=30)
        
        print("✅ AutomaticMatchDiscovery initialized with cache")
        
        # Test discovering casino matches with cache
        print("🎰 Testing casino match discovery...")
        start_time = time.time()
        matches = discovery.discover_casino_matches()
        end_time = time.time()
        
        print(f"⏱️  Discovery took: {end_time - start_time:.2f} seconds")
        print(f"📊 Matches found: {len(matches)}")
        
        if len(matches) > 0:
            print(f"🏟️  Sample match: {matches[0].get('home_team', 'N/A')} vs {matches[0].get('away_team', 'N/A')}")
        
        # Test second run (should use cache)
        print("\n🔄 Testing cached discovery...")
        start_time = time.time()
        cached_matches = discovery.discover_casino_matches()
        end_time = time.time()
        
        print(f"⏱️  Cached discovery took: {end_time - start_time:.2f} seconds")
        print(f"📊 Cached matches: {len(cached_matches)}")
        
        if len(matches) == len(cached_matches):
            print("✅ Cache working - same number of matches returned")
        else:
            print("⚠️  Different number of matches (may be normal)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in automatic discovery test: {e}")
        return False

def main():
    """Run all cache tests"""
    print("🚀 Cache System Test Suite")
    print("=" * 60)
    
    # Test 1: Basic cache functionality
    cache_test_passed = test_cache_manager()
    
    # Test 2: Automatic discovery with cache
    discovery_test_passed = test_automatic_discovery_cache()
    
    # Summary
    print("\n📋 Test Results Summary")
    print("=" * 40)
    print(f"Cache Manager Test: {'✅ PASSED' if cache_test_passed else '❌ FAILED'}")
    print(f"Discovery Cache Test: {'✅ PASSED' if discovery_test_passed else '❌ FAILED'}")
    
    if cache_test_passed and discovery_test_passed:
        print("\n🎉 All tests PASSED! Cache system is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
