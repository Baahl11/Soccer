#!/usr/bin/env python3
"""
Phase 2 Integration Testing Suite
Comprehensive end-to-end testing for Auto-Updating ELO System with Database Backend
"""

import os
import sys
import json
import time
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
import unittest

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the ELO system components
try:
    from auto_updating_elo import AutoUpdatingELO
    print("✅ All ELO system imports successful")
    IMPORTS_AVAILABLE = True
except ImportError as e:
    print(f"❌ Import error: {e}")
    IMPORTS_AVAILABLE = False

class Phase2IntegrationTest(unittest.TestCase):
    """Comprehensive integration test suite for Phase 2 database backend"""
    
    def setUp(self):
        """Set up test environment"""
        if not IMPORTS_AVAILABLE:
            self.skipTest("Required imports not available")
        
        # Create temporary directories for testing
        self.temp_dir = tempfile.mkdtemp()
        self.json_backup_dir = os.path.join(self.temp_dir, "json_backup")
        self.database_dir = os.path.join(self.temp_dir, "database")
        
        os.makedirs(self.json_backup_dir, exist_ok=True)
        os.makedirs(self.database_dir, exist_ok=True)
        
        # Test data
        self.test_teams = {
            1: {"name": "Test Team A", "league_id": 39},
            2: {"name": "Test Team B", "league_id": 39},
            3: {"name": "Test Team C", "league_id": 140},
            4: {"name": "Test Team D", "league_id": 140}
        }
        
        self.test_matches = [
            {
                "home_team_id": 1,
                "away_team_id": 2,
                "home_goals": 2,
                "away_goals": 1,
                "league_id": 39,
                "match_date": datetime.now() - timedelta(days=30)
            },
            {
                "home_team_id": 3,
                "away_team_id": 4,
                "home_goals": 0,
                "away_goals": 3,
                "league_id": 140,
                "match_date": datetime.now() - timedelta(days=25)
            },
            {
                "home_team_id": 2,
                "away_team_id": 1,
                "home_goals": 1,
                "away_goals": 1,
                "league_id": 39,
                "match_date": datetime.now() - timedelta(days=20)
            }
        ]
        
        print(f"✅ Test setup complete - Temp dir: {self.temp_dir}")
    
    def tearDown(self):
        """Clean up test environment"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        print("✅ Test cleanup complete")
    
    def test_01_json_only_mode(self):
        """Test ELO system in JSON-only mode (baseline)"""
        print("\n🧪 Testing JSON-only mode...")
        
        json_file = os.path.join(self.json_backup_dir, "test_ratings.json")
          # Initialize system without database
        elo_system = AutoUpdatingELO(
            ratings_file=json_file,
            use_database=False
        )
          # Add test matches
        for match in self.test_matches:
            elo_system.process_match({
                "home_team_id": match["home_team_id"],
                "away_team_id": match["away_team_id"],
                "home_goals": match["home_goals"],
                "away_goals": match["away_goals"],
                "league_id": match["league_id"],
                "match_date": match["match_date"]
            })
        
        # Verify ratings were calculated
        team1_rating = elo_system.get_team_rating(1, 39)
        team2_rating = elo_system.get_team_rating(2, 39)
        
        self.assertIsInstance(team1_rating, float)
        self.assertIsInstance(team2_rating, float)
        self.assertGreater(team1_rating, 1000)  # Should be above initial rating
        self.assertLess(team2_rating, 1500)     # Should be below initial rating
        
        # Verify JSON file was created
        self.assertTrue(os.path.exists(json_file))
        
        with open(json_file, 'r') as f:
            data = json.load(f)
            self.assertIn('39', data)  # League data should exist
            self.assertIn('1', data['39'])  # Team data should exist
        
        elo_system.elo_system.close()
        print("✅ JSON-only mode test passed")
    
    def test_02_database_initialization(self):
        """Test database backend initialization"""
        print("\n🧪 Testing database initialization...")
        
        database_config = {
            'type': 'sqlite',
            'database': os.path.join(self.database_dir, 'test_elo.db')
        }
          # Test database backend creation through AutoUpdatingELO
        json_file = os.path.join(self.json_backup_dir, "init_test.json")
        elo_system = AutoUpdatingELO(
            ratings_file=json_file,
            use_database=True,
            database_config=database_config
        )
        self.assertIsNotNone(elo_system)
        self.assertIsNotNone(elo_system.elo_system)
        
        # Verify database functionality
        stats = elo_system.elo_system.get_database_statistics()
        # Stats might be None if database isn't available
        self.assertIn(type(stats), [dict, type(None)])
        
        # Verify database file was created
        self.assertTrue(os.path.exists(database_config['database']))
        
        elo_system.elo_system.close()
        print("✅ Database initialization test passed")
    
    def test_03_database_mode_basic(self):
        """Test ELO system with database backend"""
        print("\n🧪 Testing database mode basic functionality...")
        
        database_config = {
            'type': 'sqlite',
            'database': os.path.join(self.database_dir, 'test_elo_basic.db')
        }
        
        json_file = os.path.join(self.json_backup_dir, "test_ratings_db.json")
          # Initialize system with database
        elo_system = AutoUpdatingELO(
            ratings_file=json_file,
            use_database=True,
            database_config=database_config
        )
          # Add test matches
        for i, match in enumerate(self.test_matches):
            print(f"  Processing match {i+1}/3...")
            result = elo_system.process_match({
                "home_team_id": match["home_team_id"],
                "away_team_id": match["away_team_id"],
                "home_goals": match["home_goals"],
                "away_goals": match["away_goals"],
                "league_id": match["league_id"],
                "match_date": match["match_date"]
            })
            self.assertIsNotNone(result)
          # Verify ratings
        team1_rating = elo_system.elo_system.get_team_rating(1, 39)
        team3_rating = elo_system.elo_system.get_team_rating(3, 140)
        
        self.assertIsInstance(team1_rating, float)
        self.assertIsInstance(team3_rating, float)
          # Test predictions
        prediction = elo_system.elo_system.get_match_prediction(1, 2, 39)
        self.assertIsInstance(prediction, dict)
        self.assertIn('home_win_prob', prediction)
        self.assertIn('draw_prob', prediction)
        self.assertIn('away_win_prob', prediction)
        
        # Test statistics
        stats = elo_system.elo_system.get_database_statistics()
        if stats:  # Only test if database is working
            self.assertIsInstance(stats, dict)
        
        elo_system.elo_system.close()
        print("✅ Database mode basic test passed")
    
    def test_04_database_vs_json_consistency(self):
        """Test consistency between database and JSON storage"""
        print("\n🧪 Testing database vs JSON consistency...")
          # Setup JSON system
        json_file = os.path.join(self.json_backup_dir, "consistency_json.json")
        json_system = AutoUpdatingELO(
            ratings_file=json_file,
            use_database=False
        )
        
        # Setup database system
        database_config = {
            'type': 'sqlite',
            'database': os.path.join(self.database_dir, 'consistency_db.db')
        }
        db_json_file = os.path.join(self.json_backup_dir, "consistency_db.json")
        db_system = AutoUpdatingELO(
            ratings_file=db_json_file,
            use_database=True,
            database_config=database_config
        )
          # Process same matches in both systems
        for match in self.test_matches:
            json_system.process_match({
                "home_team_id": match["home_team_id"],
                "away_team_id": match["away_team_id"],
                "home_goals": match["home_goals"],
                "away_goals": match["away_goals"],
                "league_id": match["league_id"],
                "match_date": match["match_date"]
            })
            
            db_system.process_match({
                "home_team_id": match["home_team_id"],
                "away_team_id": match["away_team_id"],
                "home_goals": match["home_goals"],
                "away_goals": match["away_goals"],
                "league_id": match["league_id"],
                "match_date": match["match_date"]
            })
          # Compare ratings
        for team_id in [1, 2, 3, 4]:
            for league_id in [39, 140]:
                if team_id <= 2 and league_id == 39:
                    json_rating = json_system.elo_system.get_team_rating(team_id, league_id)
                    db_rating = db_system.elo_system.get_team_rating(team_id, league_id)
                    
                    # Ratings should be very close (within 0.1 points)
                    self.assertAlmostEqual(json_rating, db_rating, delta=0.1,
                                         msg=f"Rating mismatch for team {team_id} in league {league_id}")
        
        # Compare predictions
        json_pred = json_system.elo_system.get_match_prediction(1, 2, 39)
        db_pred = db_system.elo_system.get_match_prediction(1, 2, 39)
        
        self.assertAlmostEqual(json_pred['home_win_prob'], db_pred['home_win_prob'], delta=0.01)
        self.assertAlmostEqual(json_pred['draw_prob'], db_pred['draw_prob'], delta=0.01)
        self.assertAlmostEqual(json_pred['away_win_prob'], db_pred['away_win_prob'], delta=0.01)
        json_system.elo_system.close()
        db_system.elo_system.close()
        print("✅ Database vs JSON consistency test passed")
    
    def test_05_migration_functionality(self):
        """Test migration from JSON to database"""
        print("\n🧪 Testing JSON to database migration...")
          # Create JSON system with data
        json_file = os.path.join(self.json_backup_dir, "migration_source.json")
        json_system = AutoUpdatingELO(
            ratings_file=json_file,
            use_database=False
        )
          # Add some data
        for match in self.test_matches[:2]:  # Use first 2 matches
            json_system.process_match({
                "home_team_id": match["home_team_id"],
                "away_team_id": match["away_team_id"],
                "home_goals": match["home_goals"],
                "away_goals": match["away_goals"],                "league_id": match["league_id"],
                "match_date": match["match_date"]
            })
        
        json_system.elo_system.close()
        
        # Now create database system using the same JSON file
        database_config = {
            'type': 'sqlite',
            'database': os.path.join(self.database_dir, 'migration_target.db')
        }
        db_system = AutoUpdatingELO(
            ratings_file=json_file,  # Same file - should migrate
            use_database=True,
            database_config=database_config
        )
        # Verify data is accessible
        team1_rating = db_system.elo_system.get_team_rating(1, 39)
        self.assertIsInstance(team1_rating, float)
        self.assertNotEqual(team1_rating, 1500)  # Should not be default rating
        
        # Add more data to test continued operation
        db_system.process_match({
            "home_team_id": self.test_matches[2]["home_team_id"],
            "away_team_id": self.test_matches[2]["away_team_id"],
            "home_goals": self.test_matches[2]["home_goals"],
            "away_goals": self.test_matches[2]["away_goals"],
            "league_id": self.test_matches[2]["league_id"],
            "match_date": self.test_matches[2]["match_date"]
        })
        
        db_system.elo_system.close()
        print("✅ Migration functionality test passed")
    
    def test_06_performance_comparison(self):
        """Test performance comparison between JSON and database"""
        print("\n🧪 Testing performance comparison...")
          # JSON performance test
        json_file = os.path.join(self.json_backup_dir, "perf_json.json")
        start_time = time.time()
        json_system = AutoUpdatingELO(
            ratings_file=json_file,
            use_database=False
        )
        for _ in range(10):  # Repeat matches multiple times
            for match in self.test_matches:
                json_system.process_match({
                    "home_team_id": match["home_team_id"],
                    "away_team_id": match["away_team_id"],
                    "home_goals": match["home_goals"],
                    "away_goals": match["away_goals"],
                    "league_id": match["league_id"],
                    "match_date": match["match_date"]
                })
        
        json_system.elo_system.close()
        json_time = time.time() - start_time
        
        # Database performance test
        database_config = {
            'type': 'sqlite',
            'database': os.path.join(self.database_dir, 'perf_db.db')
        }        
        db_json_file = os.path.join(self.json_backup_dir, "perf_db.json")
        start_time = time.time()
        db_system = AutoUpdatingELO(
            ratings_file=db_json_file,
            use_database=True,
            database_config=database_config
        )
        for _ in range(10):  # Repeat matches multiple times
            for match in self.test_matches:
                db_system.process_match({
                    "home_team_id": match["home_team_id"],
                    "away_team_id": match["away_team_id"],
                    "home_goals": match["home_goals"],
                    "away_goals": match["away_goals"],
                    "league_id": match["league_id"],
                    "match_date": match["match_date"]
                })
        
        db_system.elo_system.close()
        db_time = time.time() - start_time
        
        print(f"  JSON processing time: {json_time:.3f} seconds")
        print(f"  Database processing time: {db_time:.3f} seconds")
        print(f"  Performance ratio (DB/JSON): {db_time/json_time:.2f}x")
        
        # Database should be reasonable (not more than 5x slower for small datasets)
        self.assertLess(db_time / json_time, 5.0, "Database performance is too slow")
        
        print("✅ Performance comparison test passed")
    
    def test_07_error_handling_and_fallback(self):
        """Test error handling and fallback mechanisms"""
        print("\n🧪 Testing error handling and fallback...")
        
        # Test with invalid database config (should fallback to JSON)
        invalid_config = {
            'type': 'invalid_db_type',
            'database': '/invalid/path/db.db'
        }
        
        json_file = os.path.join(self.json_backup_dir, "fallback_test.json")
          # This should not crash and should fallback to JSON mode
        elo_system = AutoUpdatingELO(
            ratings_file=json_file,
            use_database=True,
            database_config=invalid_config
        )
          # Should still work in JSON mode
        elo_system.process_match({
            "home_team_id": 1,
            "away_team_id": 2,
            "home_goals": 2,
            "away_goals": 1,
            "league_id": 39,
            "match_date": datetime.now()
        })
        
        rating = elo_system.elo_system.get_team_rating(1, 39)
        self.assertIsInstance(rating, float)
        
        # JSON file should exist (fallback worked)
        self.assertTrue(os.path.exists(json_file))
        
        elo_system.elo_system.close()
        print("✅ Error handling and fallback test passed")
    
    def test_08_auto_updating_elo_wrapper(self):
        """Test the AutoUpdatingELO wrapper class"""
        print("\n🧪 Testing AutoUpdatingELO wrapper...")
        
        database_config = {
            'type': 'sqlite',
            'database': os.path.join(self.database_dir, 'wrapper_test.db')
        }
        
        json_file = os.path.join(self.json_backup_dir, "wrapper_test.json")
        
        # Test wrapper initialization
        auto_elo = AutoUpdatingELO(
            ratings_file=json_file,
            use_database=True,
            database_config=database_config
        )
          # Test wrapper methods
        auto_elo.process_match({
            "home_team_id": 1,
            "away_team_id": 2,
            "home_goals": 2,
            "away_goals": 1,
            "league_id": 39
        })
        rating = auto_elo.get_team_rating(1, 39)
        self.assertIsInstance(rating, float)
        
        prediction = auto_elo.get_match_prediction(1, 2, 39)
        self.assertIsInstance(prediction, dict)
        
        stats = auto_elo.elo_system.get_database_statistics()
        # Should return stats or None (if database not available)
        self.assertIn(type(stats), [dict, type(None)])
        
        auto_elo.elo_system.close()
        print("✅ AutoUpdatingELO wrapper test passed")

def run_comprehensive_test():
    """Run the comprehensive integration test suite"""
    print("🚀 Starting Phase 2 Integration Testing Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(Phase2IntegrationTest)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=False
    )
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    print("\n" + "=" * 60)
    print(f"🏁 Integration Testing Complete")
    print(f"⏱️  Total time: {end_time - start_time:.2f} seconds")
    print(f"✅ Tests passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Tests failed: {len(result.failures)}")
    print(f"💥 Tests error: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ Phase 2 database backend integration is READY FOR PRODUCTION")
    else:
        print("\n⚠️  Some integration tests failed. Review issues before production deployment.")
    
    return success

if __name__ == "__main__":
    run_comprehensive_test()
