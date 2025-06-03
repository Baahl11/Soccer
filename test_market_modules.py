"""
Test script for market integration modules.
"""

import logging
from market_integration import MarketDataIntegrator
from odds_analyzer import OddsAnalyzer
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main test function."""
    print("Testing market integration modules...")
    
    # Test MarketDataIntegrator
    print("\nTesting MarketDataIntegrator...")
    integrator = MarketDataIntegrator()
    print(f"MarketDataIntegrator successfully initialized: {integrator is not None}")
    
    # Test OddsAnalyzer
    print("\nTesting OddsAnalyzer...")
    analyzer = OddsAnalyzer()
    print(f"OddsAnalyzer successfully initialized: {analyzer is not None}")
    
    # Test market_visualizations module
    print("\nTesting market_visualizations module...")
    try:
        import market_visualizations
        print("Successfully imported market_visualizations module")
        
        # List available functions
        print("Available functions/classes:")
        for name in dir(market_visualizations):
            if not name.startswith('_'):  # Skip private members
                print(f"  - {name}")
    except ImportError as e:
        print(f"Error importing market_visualizations: {e}")
    
    # Test real_time_market_monitor module
    print("\nTesting real_time_market_monitor module...")
    try:
        import real_time_market_monitor
        print("Successfully imported real_time_market_monitor module")
        
        # List available functions
        print("Available functions/classes:")
        for name in dir(real_time_market_monitor):
            if not name.startswith('_'):  # Skip private members
                print(f"  - {name}")
    except ImportError as e:
        print(f"Error importing real_time_market_monitor: {e}")
    
    print("\nTest completed.")

if __name__ == "__main__":
    main()
