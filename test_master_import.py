#!/usr/bin/env python3
"""Test script for Master Pipeline import"""

import sys
import traceback

def test_import():
    try:
        print("🚀 Testing Master Pipeline Import...")
        print("=" * 50)
        
        # Test basic dependencies first
        print("Testing basic dependencies...")
        import logging
        print("✅ logging")
        
        import numpy as np
        print("✅ numpy")
        
        from typing import Dict, Any, Optional, List, Tuple
        print("✅ typing")
        
        from datetime import datetime, timedelta
        print("✅ datetime")
        
        # Test main import
        print("\nTesting Master Pipeline import...")
        from master_prediction_pipeline import generate_master_prediction
        print("✅ generate_master_prediction imported successfully!")
        
        # Test instantiation
        print("\nTesting Master Pipeline instantiation...")
        result = generate_master_prediction(12345, 40, 50, 39)
        print(f"✅ Master Pipeline test successful!")
        print(f"📊 Prediction version: {result.get('prediction_version', 'unknown')}")
        print(f"🎯 Components active: {result.get('system_status', {}).get('components_active', 0)}/4")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_import()
    sys.exit(0 if success else 1)
