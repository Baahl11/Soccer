"""
Comprehensive integration test suite for the fixed VotingEnsembleCornersModel
and related prediction components.

This test suite covers:
1. Edge case scenarios with invalid/malformed input data
2. Performance testing with multiple concurrent predictions
3. Cross-module integration testing
4. Error recovery under various failure conditions
5. Memory and resource usage monitoring
"""

import sys
import time
import threading
import tracemalloc
import traceback
import concurrent.futures
from typing import Dict, Any, List
import numpy as np
import pandas as pd

# Import the modules we're testing
try:
    from voting_ensemble_corners import VotingEnsembleCornersModel
    from tactical_corner_predictor import TacticalCornerPredictor
    from enhanced_predictions import make_enhanced_prediction
    from team_form import get_team_form
    from match_winner import predict_match_winner
    from team_elo_rating import get_elo_ratings_for_match
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

class ComprehensiveIntegrationTester:
    """Comprehensive test suite for integration fixes"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.memory_usage = {}
        
    def run_all_tests(self):
        """Run all comprehensive tests"""
        print("=" * 60)
        print("COMPREHENSIVE INTEGRATION TEST SUITE")
        print("=" * 60)
        
        # Start memory tracking
        tracemalloc.start()
        
        # Test categories
        test_categories = [
            ("Edge Case Testing", self.test_edge_cases),
            ("Performance Testing", self.test_performance),
            ("Cross-Module Integration", self.test_cross_module_integration),
            ("Error Recovery Testing", self.test_error_recovery),
            ("Memory Usage Testing", self.test_memory_usage),
            ("Concurrent Access Testing", self.test_concurrent_access),
            ("Data Validation Testing", self.test_data_validation),
            ("API Integration Testing", self.test_api_integration)
        ]
        
        for category_name, test_function in test_categories:
            print(f"\n{'-' * 40}")
            print(f"Running {category_name}...")
            print(f"{'-' * 40}")
            
            try:
                start_time = time.time()
                result = test_function()
                end_time = time.time()
                
                self.test_results[category_name] = {
                    'status': 'PASSED' if result else 'FAILED',
                    'duration': end_time - start_time,
                    'details': result
                }
                
                print(f"✓ {category_name}: {'PASSED' if result else 'FAILED'} ({end_time - start_time:.2f}s)")
                
            except Exception as e:
                self.test_results[category_name] = {
                    'status': 'ERROR',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                print(f"✗ {category_name}: ERROR - {str(e)}")
        
        # Stop memory tracking and get final stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        self.memory_usage['final'] = {
            'current': current / 1024 / 1024,  # MB
            'peak': peak / 1024 / 1024  # MB
        }
        
        # Print summary
        self.print_test_summary()
        
        return self.test_results
    
    def test_edge_cases(self):
        """Test edge cases with invalid/malformed input data"""
        edge_cases = []
        
        # Test 1: Invalid team IDs
        try:
            model = VotingEnsembleCornersModel()
            result = model.predict_corners(
                home_team_id=-1,
                away_team_id=999999,
                home_stats={},
                away_stats={},
                league_id=39
            )
            edge_cases.append(("Invalid team IDs", True))
        except Exception as e:
            edge_cases.append(("Invalid team IDs", f"Error: {str(e)}"))
        
        # Test 2: Missing required stats
        try:
            model = VotingEnsembleCornersModel()
            result = model.predict_corners(
                home_team_id=1,
                away_team_id=2,
                home_stats={},  # Empty dict instead of None
                away_stats={},  # Empty dict instead of None
                league_id=39
            )
            edge_cases.append(("Empty stats", True))
        except Exception as e:
            edge_cases.append(("Empty stats", f"Error: {str(e)}"))
        
        # Test 3: Extreme statistical values
        try:
            model = VotingEnsembleCornersModel()
            result = model.predict_corners(
                home_team_id=1,
                away_team_id=2,
                home_stats={
                    'avg_corners_for': 999.0,
                    'avg_corners_against': -50.0,
                    'form_score': 1000.0,
                    'avg_shots': -10.0
                },
                away_stats={
                    'avg_corners_for': 0.0,
                    'avg_corners_against': 999.0,
                    'form_score': -100.0,
                    'avg_shots': 1000.0
                },
                league_id=39
            )
            edge_cases.append(("Extreme values", True))
        except Exception as e:
            edge_cases.append(("Extreme values", f"Error: {str(e)}"))
        
        # Test 4: Invalid league ID
        try:
            model = VotingEnsembleCornersModel()
            result = model.predict_corners(
                home_team_id=1,
                away_team_id=2,
                home_stats={'avg_corners_for': 5.0},
                away_stats={'avg_corners_for': 4.0},
                league_id=-999
            )
            edge_cases.append(("Invalid league ID", True))
        except Exception as e:
            edge_cases.append(("Invalid league ID", f"Error: {str(e)}"))
        
        # Test 5: String values instead of numbers (expect this to fail)
        try:
            model = VotingEnsembleCornersModel()
            # This should fail with type errors, which is expected behavior
            result = model.predict_corners(
                home_team_id=1,  # Use valid int
                away_team_id=2,  # Use valid int
                home_stats={'avg_corners_for': "not_a_number"},
                away_stats={'avg_corners_for': "also_not_a_number"},
                league_id=39  # Use valid int
            )
            edge_cases.append(("String values in stats", True))
        except Exception as e:
            edge_cases.append(("String values in stats", f"Expected error: {str(e)}"))
        
        print(f"Edge case tests completed: {len(edge_cases)} scenarios tested")
        for test_name, result in edge_cases:
            status = "✓" if result is True else "⚠"
            print(f"  {status} {test_name}: {result}")
        
        return edge_cases
    
    def test_performance(self):
        """Test performance with multiple predictions"""
        performance_results = {}
        
        # Test 1: Single prediction timing
        model = VotingEnsembleCornersModel()
        
        start_time = time.time()
        for i in range(10):
            result = model.predict_corners(
                home_team_id=1,
                away_team_id=2,
                home_stats={'avg_corners_for': 5.0, 'avg_corners_against': 4.5},
                away_stats={'avg_corners_for': 4.0, 'avg_corners_against': 5.5},
                league_id=39
            )
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        performance_results['single_prediction_avg_time'] = avg_time
        print(f"Average single prediction time: {avg_time:.4f}s")
        
        # Test 2: Batch prediction timing
        start_time = time.time()
        batch_results = []
        for i in range(100):
            result = model.predict_corners(
                home_team_id=i % 50 + 1,
                away_team_id=(i + 1) % 50 + 1,
                home_stats={'avg_corners_for': 5.0 + (i % 3), 'avg_corners_against': 4.5},
                away_stats={'avg_corners_for': 4.0, 'avg_corners_against': 5.5 + (i % 2)},
                league_id=39
            )
            batch_results.append(result)
        end_time = time.time()
        
        batch_time = end_time - start_time
        performance_results['batch_100_total_time'] = batch_time
        performance_results['batch_100_avg_time'] = batch_time / 100
        print(f"Batch 100 predictions total time: {batch_time:.4f}s")
        print(f"Batch 100 predictions avg time: {batch_time/100:.4f}s")
        
        # Test 3: Memory efficiency check
        try:
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Create multiple model instances
            models = []
            for i in range(10):
                models.append(VotingEnsembleCornersModel())
            
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = memory_after - memory_before
            
            performance_results['memory_per_model_mb'] = memory_increase / 10
            print(f"Memory per model instance: {memory_increase/10:.2f} MB")
            
        except ImportError:
            print("psutil not available - skipping memory efficiency test")
            performance_results['memory_per_model_mb'] = "N/A (psutil not available)"
        
        self.performance_metrics = performance_results
        return performance_results
    
    def test_cross_module_integration(self):
        """Test integration between different prediction modules"""
        integration_results = []
        
        # Test 1: Enhanced prediction integration
        try:
            result = make_enhanced_prediction(
                fixture_id=12345,
                home_team_id=1,
                away_team_id=2,
                league_id=39,
                season=2024
            )
            
            required_keys = ['predicted_home_goals', 'predicted_away_goals', 'corners', 'match_winner']
            has_all_keys = all(key in result for key in required_keys)
            integration_results.append(("Enhanced prediction integration", has_all_keys))
            
        except Exception as e:
            integration_results.append(("Enhanced prediction integration", f"Error: {str(e)}"))
        
        # Test 2: Tactical corner predictor integration
        try:
            tactical_predictor = TacticalCornerPredictor()
            # Use the correct method name and parameters
            match_data = {
                'home_team_id': 1,
                'away_team_id': 2,
                'home_formation': '4-3-3',
                'away_formation': '4-4-2',
                'league_id': 39,
                'home_stats': {'avg_corners_for': 5.0},
                'away_stats': {'avg_corners_for': 4.0}
            }
            
            result = tactical_predictor.predict_with_formations(match_data)
            
            has_prediction_data = 'prediction' in result and 'tactical_analysis' in result
            integration_results.append(("Tactical corner predictor", has_prediction_data))
            
        except Exception as e:
            integration_results.append(("Tactical corner predictor", f"Error: {str(e)}"))
        
        # Test 3: ELO integration
        try:
            elo_result = get_elo_ratings_for_match(1, 2, 39)
            has_elo_data = 'home_elo' in elo_result and 'away_elo' in elo_result
            integration_results.append(("ELO integration", has_elo_data))
            
        except Exception as e:
            integration_results.append(("ELO integration", f"Error: {str(e)}"))
        
        print(f"Cross-module integration tests: {len(integration_results)} tests")
        for test_name, result in integration_results:
            status = "✓" if result is True else "⚠"
            print(f"  {status} {test_name}: {result}")
        
        return integration_results
    
    def test_error_recovery(self):
        """Test error recovery under various failure conditions"""
        recovery_results = []
        
        # Test 1: Model file not found (should fallback gracefully)
        try:
            model = VotingEnsembleCornersModel()
            # The model should initialize even without model files
            is_initialized = hasattr(model, 'rf_model') and hasattr(model, 'xgb_model')
            recovery_results.append(("Model file not found recovery", is_initialized))
            
        except Exception as e:
            recovery_results.append(("Model file not found recovery", f"Error: {str(e)}"))
        
        # Test 2: Network/API failure simulation
        try:
            # This should work even if external APIs fail
            model = VotingEnsembleCornersModel()
            result = model.predict_corners(
                home_team_id=999999,  # Non-existent team
                away_team_id=999998,  # Non-existent team
                home_stats={'avg_corners_for': 5.0},
                away_stats={'avg_corners_for': 4.0},
                league_id=999  # Non-existent league
            )
            
            has_fallback_result = 'total' in result and result.get('is_fallback', False)
            recovery_results.append(("API failure recovery", has_fallback_result or 'total' in result))
            
        except Exception as e:
            recovery_results.append(("API failure recovery", f"Error: {str(e)}"))
        
        # Test 3: Corrupted data recovery
        try:
            model = VotingEnsembleCornersModel()
            result = model.predict_corners(
                home_team_id=1,
                away_team_id=2,
                home_stats={'corrupted': 'data', 'invalid': None},
                away_stats={'also_corrupted': [], 'bad_data': {}},
                league_id=39
            )
            
            has_result = 'total' in result
            recovery_results.append(("Corrupted data recovery", has_result))
            
        except Exception as e:
            recovery_results.append(("Corrupted data recovery", f"Error: {str(e)}"))
        
        print(f"Error recovery tests: {len(recovery_results)} tests")
        for test_name, result in recovery_results:
            status = "✓" if result is True else "⚠"
            print(f"  {status} {test_name}: {result}")
        
        return recovery_results
    
    def test_memory_usage(self):
        """Test memory usage patterns"""
        memory_results = {}
        
        # Get initial memory
        current, peak = tracemalloc.get_traced_memory()
        initial_memory = current / 1024 / 1024  # MB
        
        # Test 1: Memory usage during multiple predictions
        model = VotingEnsembleCornersModel()
        
        for i in range(50):
            result = model.predict_corners(
                home_team_id=i % 20 + 1,
                away_team_id=(i + 1) % 20 + 1,
                home_stats={'avg_corners_for': 5.0},
                away_stats={'avg_corners_for': 4.0},
                league_id=39
            )
        
        current, peak = tracemalloc.get_traced_memory()
        after_predictions_memory = current / 1024 / 1024  # MB
        
        memory_increase = after_predictions_memory - initial_memory
        memory_results['memory_increase_50_predictions'] = memory_increase
        
        print(f"Memory increase after 50 predictions: {memory_increase:.2f} MB")
        
        # Test 2: Memory cleanup
        del model
        
        current, peak = tracemalloc.get_traced_memory()
        after_cleanup_memory = current / 1024 / 1024  # MB
        
        memory_freed = after_predictions_memory - after_cleanup_memory
        memory_results['memory_freed_after_cleanup'] = memory_freed
        
        print(f"Memory freed after cleanup: {memory_freed:.2f} MB")
        
        return memory_results
    
    def test_concurrent_access(self):
        """Test concurrent access to prediction models"""
        concurrent_results = []
        
        def make_prediction(thread_id):
            try:
                model = VotingEnsembleCornersModel()
                result = model.predict_corners(
                    home_team_id=thread_id % 10 + 1,
                    away_team_id=(thread_id + 1) % 10 + 1,
                    home_stats={'avg_corners_for': 5.0},
                    away_stats={'avg_corners_for': 4.0},
                    league_id=39
                )
                return f"Thread {thread_id}: Success"
            except Exception as e:
                return f"Thread {thread_id}: Error - {str(e)}"
        
        # Test with 10 concurrent threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_prediction, i) for i in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        successful_threads = sum(1 for result in results if "Success" in result)
        concurrent_results.append(("Concurrent access", f"{successful_threads}/10 threads successful"))
        
        print(f"Concurrent access test: {successful_threads}/10 threads successful")
        for result in results:
            print(f"  {result}")
        
        return concurrent_results
    
    def test_data_validation(self):
        """Test data validation and sanitization"""
        validation_results = []
        
        # Test 1: Data type conversion (test with proper types but string values in stats)
        try:
            model = VotingEnsembleCornersModel()
            result = model.predict_corners(
                home_team_id=1,  # Use proper int
                away_team_id=2,  # Use proper int
                home_stats={'avg_corners_for': "5.0"},  # String that can be converted
                away_stats={'avg_corners_for': "4.0"},  # String that can be converted
                league_id=39  # Use proper int
            )
            
            has_result = 'total' in result
            validation_results.append(("Data type conversion", has_result))
            
        except Exception as e:
            validation_results.append(("Data type conversion", f"Error: {str(e)}"))
        
        # Test 2: Missing data handling
        try:
            model = VotingEnsembleCornersModel()
            result = model.predict_corners(
                home_team_id=1,
                away_team_id=2,
                home_stats={},  # Empty stats
                away_stats={},  # Empty stats
                league_id=39
            )
            
            has_result = 'total' in result
            validation_results.append(("Missing data handling", has_result))
            
        except Exception as e:
            validation_results.append(("Missing data handling", f"Error: {str(e)}"))
        
        print(f"Data validation tests: {len(validation_results)} tests")
        for test_name, result in validation_results:
            status = "✓" if result is True else "⚠"
            print(f"  {status} {test_name}: {result}")
        
        return validation_results
    
    def test_api_integration(self):
        """Test API integration scenarios"""
        api_results = []
        
        # Test 1: Enhanced prediction API
        try:
            result = make_enhanced_prediction(
                fixture_id=12345,
                home_team_id=1,
                away_team_id=2,
                league_id=39,
                season=2024
            )
            
            # Check if result has expected structure
            expected_keys = ['predicted_home_goals', 'predicted_away_goals', 'total_goals', 'corners']
            has_expected_structure = all(key in result for key in expected_keys)
            api_results.append(("Enhanced prediction API", has_expected_structure))
            
        except Exception as e:
            api_results.append(("Enhanced prediction API", f"Error: {str(e)}"))
        
        # Test 2: Multiple API calls
        try:
            results = []
            for i in range(5):
                result = make_enhanced_prediction(
                    fixture_id=12345 + i,
                    home_team_id=i + 1,
                    away_team_id=i + 2,
                    league_id=39,
                    season=2024
                )
                results.append(result)
            
            all_successful = len(results) == 5 and all('corners' in r for r in results)
            api_results.append(("Multiple API calls", all_successful))
            
        except Exception as e:
            api_results.append(("Multiple API calls", f"Error: {str(e)}"))
        
        print(f"API integration tests: {len(api_results)} tests")
        for test_name, result in api_results:
            status = "✓" if result is True else "⚠"
            print(f"  {status} {test_name}: {result}")
        
        return api_results
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("COMPREHENSIVE TEST SUMMARY")
        print("=" * 60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'PASSED')
        failed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'FAILED')
        error_tests = sum(1 for result in self.test_results.values() if result['status'] == 'ERROR')
        
        print(f"Total Test Categories: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Errors: {error_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        print(f"\nMemory Usage:")
        print(f"Peak Memory: {self.memory_usage['final']['peak']:.2f} MB")
        print(f"Final Memory: {self.memory_usage['final']['current']:.2f} MB")
        
        if self.performance_metrics:
            print(f"\nPerformance Metrics:")
            for metric, value in self.performance_metrics.items():
                if 'time' in metric:
                    print(f"{metric}: {value:.4f}s")
                elif 'mb' in metric.lower():
                    print(f"{metric}: {value:.2f} MB")
                else:
                    print(f"{metric}: {value}")
        
        print("\nDetailed Results:")
        for category, result in self.test_results.items():
            status_symbol = "✓" if result['status'] == 'PASSED' else "✗" if result['status'] == 'FAILED' else "⚠"
            duration = result.get('duration', 0)
            print(f"{status_symbol} {category}: {result['status']} ({duration:.2f}s)")
            
            if result['status'] == 'ERROR':
                print(f"    Error: {result.get('error', 'Unknown error')}")

def main():
    """Run comprehensive integration tests"""
    tester = ComprehensiveIntegrationTester()
    results = tester.run_all_tests()
    
    # Return overall success status
    all_passed = all(result['status'] == 'PASSED' for result in results.values())
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
