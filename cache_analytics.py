#!/usr/bin/env python3
"""
Cache Analytics Module
Provides comprehensive cache analytics and monitoring for the Soccer Predictions Platform.
"""

import json
import pickle
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class CacheAnalytics:
    """Provides analytics and monitoring for the cache system."""
    
    def __init__(self, cache_dir: str = "cache", default_ttl: int = 3600):
        """
        Initialize cache analytics.
        
        Args:
            cache_dir: Directory containing cache files
            default_ttl: Default TTL for cache entries in seconds
        """
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        
    def get_cache_overview(self) -> Dict[str, Any]:
        """
        Get comprehensive cache overview and statistics.
        
        Returns:
            Dictionary with cache statistics and health metrics
        """
        cache_files = list(self.cache_dir.glob("*.cache"))
        total_files = len(cache_files)
        
        if total_files == 0:
            return {
                'status': 'empty',
                'total_files': 0,
                'total_size_mb': 0,
                'valid_entries': 0,
                'expired_entries': 0,
                'cache_health': 'empty'
            }
        
        # Calculate total size
        total_size = sum(f.stat().st_size for f in cache_files if f.exists())
        total_size_mb = total_size / (1024 * 1024)
        
        # Analyze cache entries
        current_time = datetime.now().timestamp()
        valid_entries = 0
        expired_entries = 0
        corrupted_entries = 0
        newest_entry = 0
        oldest_entry = float('inf')
        
        for cache_file in cache_files:
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                cache_time = cache_data.get('timestamp', 0)
                age_seconds = current_time - cache_time
                
                if age_seconds <= self.default_ttl:
                    valid_entries += 1
                else:
                    expired_entries += 1
                
                newest_entry = max(newest_entry, cache_time)
                oldest_entry = min(oldest_entry, cache_time)
                
            except Exception as e:
                logger.debug(f"Corrupted cache file {cache_file.name}: {e}")
                corrupted_entries += 1
        
        # Calculate cache health
        valid_ratio = valid_entries / total_files if total_files > 0 else 0
        cache_health = self._assess_cache_health(valid_ratio, corrupted_entries, total_files)
        
        # Calculate age metrics
        newest_age_hours = (current_time - newest_entry) / 3600 if newest_entry > 0 else 0
        oldest_age_hours = (current_time - oldest_entry) / 3600 if oldest_entry < float('inf') else 0
        
        return {
            'status': 'active',
            'total_files': total_files,
            'total_size_mb': round(total_size_mb, 2),
            'valid_entries': valid_entries,
            'expired_entries': expired_entries,
            'corrupted_entries': corrupted_entries,
            'valid_ratio': round(valid_ratio, 3),
            'cache_health': cache_health,
            'newest_entry_age_hours': round(newest_age_hours, 2),
            'oldest_entry_age_hours': round(oldest_age_hours, 2),
            'default_ttl_hours': self.default_ttl / 3600,
            'cache_directory': str(self.cache_dir)
        }
    
    def get_cache_performance_metrics(self, cache_manager) -> Dict[str, Any]:
        """
        Get performance metrics from cache manager.
        
        Args:
            cache_manager: Instance of CacheManager with analytics
            
        Returns:
            Performance metrics dictionary
        """
        if not hasattr(cache_manager, 'cache_hits'):
            return {
                'error': 'Cache manager does not have analytics enabled',
                'cache_hits': 0,
                'cache_misses': 0,
                'hit_rate_percent': 0
            }
        
        total_requests = cache_manager.cache_hits + cache_manager.cache_misses
        hit_rate = (cache_manager.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        # Calculate efficiency rating
        efficiency = self._calculate_efficiency_rating(hit_rate)
        
        return {
            'cache_hits': cache_manager.cache_hits,
            'cache_misses': cache_manager.cache_misses,
            'cache_sets': getattr(cache_manager, 'cache_sets', 0),
            'total_requests': total_requests,
            'hit_rate_percent': round(hit_rate, 2),
            'efficiency_rating': efficiency,
            'performance_status': self._assess_performance(hit_rate)
        }
    
    def analyze_cache_usage_patterns(self) -> Dict[str, Any]:
        """
        Analyze cache usage patterns and provide insights.
        
        Returns:
            Analysis of cache usage patterns
        """
        cache_files = list(self.cache_dir.glob("*.cache"))
        
        if not cache_files:
            return {'status': 'no_data', 'patterns': []}
        
        # Group by method and time patterns
        method_usage = {}
        hourly_usage = {}
        current_time = datetime.now().timestamp()
        
        for cache_file in cache_files:
            try:
                with open(cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                cache_time = cache_data.get('timestamp', 0)
                data = cache_data.get('data', {})
                
                # Extract method if available (from cache key patterns)
                method = 'unknown'
                if isinstance(data, dict):
                    method = data.get('method', 'unknown')
                
                # Count by method
                method_usage[method] = method_usage.get(method, 0) + 1
                
                # Count by hour
                cache_hour = datetime.fromtimestamp(cache_time).hour
                hourly_usage[cache_hour] = hourly_usage.get(cache_hour, 0) + 1
                
            except Exception:
                continue
        
        # Find peak usage patterns
        peak_hour = max(hourly_usage, key=hourly_usage.get) if hourly_usage else 0
        most_cached_method = max(method_usage, key=method_usage.get) if method_usage else 'unknown'
        
        return {
            'status': 'analyzed',
            'method_usage': method_usage,
            'hourly_usage': hourly_usage,
            'peak_hour': peak_hour,
            'most_cached_method': most_cached_method,
            'total_analyzed_files': len(cache_files),
            'insights': self._generate_usage_insights(method_usage, hourly_usage)
        }
    
    def recommend_optimizations(self) -> List[Dict[str, Any]]:
        """
        Recommend cache optimizations based on current state.
        
        Returns:
            List of optimization recommendations
        """
        overview = self.get_cache_overview()
        recommendations = []
        
        # Check for expired entries
        if overview['expired_entries'] > overview['valid_entries']:
            recommendations.append({
                'priority': 'high',
                'type': 'cleanup',
                'title': 'Clean up expired cache entries',
                'description': f"Found {overview['expired_entries']} expired entries vs {overview['valid_entries']} valid ones",
                'action': 'Run cache cleanup to remove expired entries and free up space'
            })
        
        # Check cache size
        if overview['total_size_mb'] > 100:  # > 100MB
            recommendations.append({
                'priority': 'medium',
                'type': 'size',
                'title': 'Large cache size detected',
                'description': f"Cache size is {overview['total_size_mb']:.1f}MB",
                'action': 'Consider implementing cache size limits or more aggressive cleanup'
            })
        
        # Check for corrupted entries
        if overview.get('corrupted_entries', 0) > 0:
            recommendations.append({
                'priority': 'high',
                'type': 'integrity',
                'title': 'Corrupted cache entries found',
                'description': f"Found {overview['corrupted_entries']} corrupted cache files",
                'action': 'Remove corrupted files and investigate causes'
            })
        
        # Check cache health
        if overview['cache_health'] in ['poor', 'critical']:
            recommendations.append({
                'priority': 'high',
                'type': 'health',
                'title': 'Poor cache health',
                'description': f"Cache health is rated as '{overview['cache_health']}'",
                'action': 'Perform comprehensive cache cleanup and monitoring'
            })
        
        # TTL recommendations
        if overview['oldest_entry_age_hours'] > overview['default_ttl_hours'] * 2:
            recommendations.append({
                'priority': 'low',
                'type': 'ttl',
                'title': 'Old cache entries detected',
                'description': f"Oldest entry is {overview['oldest_entry_age_hours']:.1f} hours old",
                'action': 'Consider adjusting TTL settings or cleanup frequency'
            })
        
        return recommendations
    
    def _assess_cache_health(self, valid_ratio: float, corrupted: int, total: int) -> str:
        """Assess overall cache health based on metrics."""
        if corrupted > total * 0.1:  # More than 10% corrupted
            return 'critical'
        elif valid_ratio < 0.3:  # Less than 30% valid
            return 'poor'
        elif valid_ratio < 0.6:  # Less than 60% valid
            return 'moderate'
        elif valid_ratio < 0.8:  # Less than 80% valid
            return 'good'
        else:
            return 'excellent'
    
    def _calculate_efficiency_rating(self, hit_rate: float) -> str:
        """Calculate efficiency rating based on hit rate."""
        if hit_rate >= 90:
            return 'excellent'
        elif hit_rate >= 75:
            return 'good'
        elif hit_rate >= 60:
            return 'moderate'
        elif hit_rate >= 40:
            return 'poor'
        else:
            return 'critical'
    
    def _assess_performance(self, hit_rate: float) -> str:
        """Assess performance status."""
        if hit_rate >= 80:
            return 'optimal'
        elif hit_rate >= 60:
            return 'good'
        elif hit_rate >= 40:
            return 'suboptimal'
        else:
            return 'poor'
    
    def _generate_usage_insights(self, method_usage: Dict, hourly_usage: Dict) -> List[str]:
        """Generate insights from usage patterns."""
        insights = []
        
        if method_usage:
            total_methods = len(method_usage)
            insights.append(f"Cache is used by {total_methods} different methods")
            
            # Find dominant method
            if total_methods > 1:
                max_usage = max(method_usage.values())
                total_usage = sum(method_usage.values())
                if max_usage > total_usage * 0.5:
                    dominant_method = max(method_usage, key=method_usage.get)
                    insights.append(f"Method '{dominant_method}' dominates cache usage ({max_usage}/{total_usage} entries)")
        
        if hourly_usage:
            peak_hour = max(hourly_usage, key=hourly_usage.get)
            peak_usage = hourly_usage[peak_hour]
            insights.append(f"Peak cache activity at hour {peak_hour} with {peak_usage} entries")
            
            # Check for usage distribution
            hours_with_activity = len(hourly_usage)
            if hours_with_activity < 6:
                insights.append("Cache usage is concentrated in few hours - consider TTL optimization")
        
        return insights

def main():
    """Demo of cache analytics functionality."""
    print("=" * 60)
    print("CACHE ANALYTICS DEMO")
    print("=" * 60)
    
    analytics = CacheAnalytics()
    
    # Get overview
    overview = analytics.get_cache_overview()
    print("\nCACHE OVERVIEW:")
    print(f"Status: {overview['status']}")
    print(f"Total files: {overview['total_files']}")
    print(f"Size: {overview['total_size_mb']} MB")
    print(f"Valid entries: {overview['valid_entries']}")
    print(f"Expired entries: {overview['expired_entries']}")
    print(f"Health: {overview['cache_health']}")
    
    # Analyze usage patterns
    patterns = analytics.analyze_cache_usage_patterns()
    print(f"\nUSAGE PATTERNS:")
    print(f"Most cached method: {patterns.get('most_cached_method', 'unknown')}")
    print(f"Peak hour: {patterns.get('peak_hour', 'unknown')}")
    print(f"Insights: {len(patterns.get('insights', []))}")
    
    # Get recommendations
    recommendations = analytics.recommend_optimizations()
    print(f"\nRECOMMENDATIONS: {len(recommendations)}")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. [{rec['priority'].upper()}] {rec['title']}")
        print(f"   {rec['description']}")

if __name__ == "__main__":
    main()
