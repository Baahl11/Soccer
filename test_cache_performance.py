#!/usr/bin/env python3
"""
Cache Performance Test
Tests the caching system performance and hit rates.
"""

from automatic_match_discovery import AutomaticMatchDiscovery
import time
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_cache_performance():
    """Test cache performance and hit rates."""
    
    print('=' * 60)
    print('CACHE PERFORMANCE TEST')
    print('=' * 60)
    
    discovery = AutomaticMatchDiscovery()
    
    # Test 1: First run (cache miss expected)
    print('\n1. First run (cache miss expected):')
    start_time = time.time()
    result1 = discovery.get_todays_predictions()
    first_run_time = time.time() - start_time
    
    print(f'Status: {result1.get("status")}')
    print(f'Matches found: {result1.get("total_matches", 0)}')
    print(f'Time taken: {first_run_time:.2f} seconds')
    
    # Test 2: Second run (cache hit expected)
    print('\n2. Second run (cache hit expected):')
    start_time = time.time()
    result2 = discovery.get_todays_predictions()
    second_run_time = time.time() - start_time
    
    print(f'Status: {result2.get("status")}')
    print(f'Matches found: {result2.get("total_matches", 0)}')
    print(f'Time taken: {second_run_time:.2f} seconds')
    
    # Performance metrics
    if first_run_time > 0 and second_run_time > 0:
        speedup = first_run_time / second_run_time
        time_saved = first_run_time - second_run_time
        improvement_pct = ((speedup - 1) * 100)
        
        print(f'\n' + '=' * 60)
        print('PERFORMANCE METRICS')
        print('=' * 60)
        print(f'First run time:  {first_run_time:.2f} seconds')
        print(f'Second run time: {second_run_time:.2f} seconds')
        print(f'Cache speedup:   {speedup:.1f}x faster')
        print(f'Time saved:      {time_saved:.2f} seconds')
        print(f'Performance improvement: {improvement_pct:.1f}%')
        
        # Cache efficiency assessment
        if speedup > 10:
            print(f'✅ EXCELLENT cache performance!')
        elif speedup > 5:
            print(f'✅ GOOD cache performance')
        elif speedup > 2:
            print(f'⚠️  MODERATE cache performance')
        else:
            print(f'❌ POOR cache performance - check implementation')
    
    # Test 3: Individual method caching
    print(f'\n' + '=' * 60)
    print('INDIVIDUAL METHOD CACHING TEST')
    print('=' * 60)
    
    # Test match discovery caching
    print('\n3. Testing match discovery cache:')
    start_time = time.time()
    matches1 = discovery.discover_casino_matches()
    discovery_time1 = time.time() - start_time
    print(f'First call: {len(matches1)} matches in {discovery_time1:.2f}s')
    
    start_time = time.time()
    matches2 = discovery.discover_casino_matches()
    discovery_time2 = time.time() - start_time
    print(f'Second call: {len(matches2)} matches in {discovery_time2:.2f}s')
    
    if discovery_time1 > 0 and discovery_time2 > 0:
        discovery_speedup = discovery_time1 / discovery_time2
        print(f'Discovery cache speedup: {discovery_speedup:.1f}x faster')
    
    return {
        'overall_speedup': speedup if 'speedup' in locals() else 0,
        'time_saved': time_saved if 'time_saved' in locals() else 0,
        'first_run_time': first_run_time,
        'second_run_time': second_run_time,
        'matches_found': result1.get("total_matches", 0),
        'cache_working': second_run_time < first_run_time
    }

def test_cache_cleanup():
    """Test cache cleanup functionality."""
    
    print(f'\n' + '=' * 60)
    print('CACHE CLEANUP TEST')
    print('=' * 60)
    
    discovery = AutomaticMatchDiscovery()
    
    # Get cache directory info
    cache_dir = discovery.cache.cache_dir
    cache_files_before = list(cache_dir.glob("*.cache"))
    print(f'Cache files before cleanup: {len(cache_files_before)}')
    
    # Run cleanup
    cleared = discovery.cache.clear_expired()
    print(f'Expired entries cleared: {cleared}')
    
    # Check after cleanup
    cache_files_after = list(cache_dir.glob("*.cache"))
    print(f'Cache files after cleanup: {len(cache_files_after)}')
    
    return {
        'files_before': len(cache_files_before),
        'files_cleared': cleared,
        'files_after': len(cache_files_after)
    }

if __name__ == "__main__":
    # Run performance tests
    perf_results = test_cache_performance()
    cleanup_results = test_cache_cleanup()
    
    print(f'\n' + '=' * 60)
    print('TEST SUMMARY')
    print('=' * 60)
    print(f'Cache system working: {"✅ YES" if perf_results["cache_working"] else "❌ NO"}')
    print(f'Overall speedup: {perf_results["overall_speedup"]:.1f}x')
    print(f'Matches processed: {perf_results["matches_found"]}')
    print(f'Cache files: {cleanup_results["files_after"]} active')
    print('=' * 60)
