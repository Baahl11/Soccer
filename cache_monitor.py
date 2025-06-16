#!/usr/bin/env python3
"""
Cache Performance Monitor
Real-time monitoring and testing of the caching system performance.
"""

import time
import json
from automatic_match_discovery import AutomaticMatchDiscovery
from cache_analytics import CacheAnalytics
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_cache_performance_test():
    """Run comprehensive cache performance tests."""
    
    print("=" * 80)
    print("SOCCER PREDICTIONS PLATFORM - CACHE PERFORMANCE TEST")
    print("=" * 80)
    
    # Initialize systems
    discovery = AutomaticMatchDiscovery()
    analytics = CacheAnalytics()
    
    print("\n1. INITIAL CACHE STATE")
    print("-" * 40)
    overview = analytics.get_cache_overview()
    print(f"Cache files: {overview['total_files']}")
    print(f"Cache size: {overview['total_size_mb']} MB")
    print(f"Valid entries: {overview['valid_entries']}")
    print(f"Expired entries: {overview['expired_entries']}")
    print(f"Cache health: {overview['cache_health']}")
    
    # Test 1: Cold start (cache miss)
    print("\n2. COLD START TEST (Cache Miss Expected)")
    print("-" * 40)
    start_time = time.time()
    result1 = discovery.get_todays_predictions()
    cold_start_time = time.time() - start_time
    
    print(f"Status: {result1.get('status', 'unknown')}")
    print(f"Matches found: {result1.get('total_matches', 0)}")
    print(f"Time taken: {cold_start_time:.2f} seconds")
    print(f"System: {result1.get('system', 'unknown')}")
    
    # Test 2: Warm start (cache hit)
    print("\n3. WARM START TEST (Cache Hit Expected)")
    print("-" * 40)
    start_time = time.time()
    result2 = discovery.get_todays_predictions()
    warm_start_time = time.time() - start_time
    
    print(f"Status: {result2.get('status', 'unknown')}")
    print(f"Matches found: {result2.get('total_matches', 0)}")
    print(f"Time taken: {warm_start_time:.2f} seconds")
    
    # Performance calculations
    if cold_start_time > 0 and warm_start_time > 0:
        speedup = cold_start_time / warm_start_time
        time_saved = cold_start_time - warm_start_time
        improvement_pct = ((speedup - 1) * 100)
        
        print("\n4. PERFORMANCE METRICS")
        print("-" * 40)
        print(f"Cold start time:     {cold_start_time:.2f} seconds")
        print(f"Warm start time:     {warm_start_time:.2f} seconds")
        print(f"Speedup factor:      {speedup:.1f}x faster")
        print(f"Time saved:          {time_saved:.2f} seconds")
        print(f"Performance gain:    {improvement_pct:.1f}%")
        
        # Performance assessment
        if speedup > 20:
            performance = "🚀 EXCELLENT"
        elif speedup > 10:
            performance = "✅ VERY GOOD"
        elif speedup > 5:
            performance = "✅ GOOD"
        elif speedup > 2:
            performance = "⚠️  MODERATE"
        else:
            performance = "❌ POOR"
        
        print(f"Performance rating:  {performance}")
    
    # Test 3: Individual component caching
    print("\n5. COMPONENT CACHING TEST")
    print("-" * 40)
    
    # Test match discovery caching
    print("Testing match discovery...")
    start_time = time.time()
    matches1 = discovery.discover_casino_matches()
    discovery_cold = time.time() - start_time
    
    start_time = time.time()
    matches2 = discovery.discover_casino_matches()
    discovery_warm = time.time() - start_time
    
    discovery_speedup = discovery_cold / discovery_warm if discovery_warm > 0 else 0
    
    print(f"Discovery cold: {discovery_cold:.2f}s ({len(matches1)} matches)")
    print(f"Discovery warm: {discovery_warm:.2f}s ({len(matches2)} matches)")
    print(f"Discovery speedup: {discovery_speedup:.1f}x")
    
    # Updated cache state
    print("\n6. UPDATED CACHE STATE")
    print("-" * 40)
    updated_overview = analytics.get_cache_overview()
    print(f"Cache files: {updated_overview['total_files']} (+{updated_overview['total_files'] - overview['total_files']})")
    print(f"Cache size: {updated_overview['total_size_mb']} MB")
    print(f"Valid entries: {updated_overview['valid_entries']}")
    print(f"Cache health: {updated_overview['cache_health']}")
    
    # Usage patterns
    print("\n7. CACHE USAGE ANALYSIS")
    print("-" * 40)
    patterns = analytics.analyze_cache_usage_patterns()
    print(f"Methods using cache: {len(patterns.get('method_usage', {}))}")
    print(f"Most cached method: {patterns.get('most_cached_method', 'unknown')}")
    print(f"Peak usage hour: {patterns.get('peak_hour', 'unknown')}")
    
    for insight in patterns.get('insights', []):
        print(f"• {insight}")
    
    # Recommendations
    print("\n8. OPTIMIZATION RECOMMENDATIONS")
    print("-" * 40)
    recommendations = analytics.recommend_optimizations()
    
    if not recommendations:
        print("✅ No optimizations needed - cache is performing well!")
    else:
        for i, rec in enumerate(recommendations, 1):
            priority_icon = "🔴" if rec['priority'] == 'high' else "🟡" if rec['priority'] == 'medium' else "🟢"
            print(f"{priority_icon} {rec['title']}")
            print(f"   {rec['description']}")
            print(f"   Action: {rec['action']}")
            print()
    
    # Summary
    print("\n9. TEST SUMMARY")
    print("-" * 40)
    cache_working = warm_start_time < cold_start_time if cold_start_time > 0 and warm_start_time > 0 else False
    print(f"✅ Cache system functional: {'YES' if cache_working else 'NO'}")
    print(f"✅ Matches processed: {result1.get('total_matches', 0)}")
    print(f"✅ Overall speedup: {speedup:.1f}x" if 'speedup' in locals() else "❌ Could not calculate speedup")
    print(f"✅ Cache health: {updated_overview['cache_health']}")
    print(f"✅ Total cache files: {updated_overview['total_files']}")
    
    return {
        'cache_functional': cache_working,
        'cold_start_time': cold_start_time,
        'warm_start_time': warm_start_time,
        'speedup': speedup if 'speedup' in locals() else 0,
        'matches_processed': result1.get('total_matches', 0),
        'cache_health': updated_overview['cache_health'],
        'recommendations_count': len(recommendations)
    }

def monitor_cache_in_realtime():
    """Monitor cache performance in real-time."""
    
    print("\n" + "=" * 80)
    print("REAL-TIME CACHE MONITORING")
    print("=" * 80)
    print("Press Ctrl+C to stop monitoring...")
    
    analytics = CacheAnalytics()
    
    try:
        while True:
            overview = analytics.get_cache_overview()
            patterns = analytics.analyze_cache_usage_patterns()
            
            print(f"\r[{time.strftime('%H:%M:%S')}] "
                  f"Files: {overview['total_files']} | "
                  f"Size: {overview['total_size_mb']:.1f}MB | "
                  f"Valid: {overview['valid_entries']} | "
                  f"Health: {overview['cache_health']}", end="")
            
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    # Run performance test
    test_results = run_cache_performance_test()
    
    # Optionally run real-time monitoring
    response = input("\nWould you like to start real-time monitoring? (y/n): ")
    if response.lower().startswith('y'):
        monitor_cache_in_realtime()
    
    print("\nCache performance testing completed!")
    print("=" * 80)
