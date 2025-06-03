# Test script to verify the fixed math_utils.py and stats_utils.py files
import numpy as np
import logging
import os
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("verification")

def test_math_utils():
    logger.info("Testing fixed_math_utils.py")
    
    # Import the fixed module
    import fixed_math_utils as math_utils
    
    # Test with regular numbers
    assert math_utils.safe_divide(10, 2) == 5.0
    assert math_utils.safe_divide(10, 0) == 0.0
    
    # Test with arrays
    test_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = math_utils.safe_array_std(test_array)
    assert isinstance(result, float)
    
    # Test with empty array
    empty_array = np.array([])
    assert math_utils.safe_array_std(empty_array) == 0.0
    
    # Test with problematic data types
    complex_array = np.array([1+2j, 2+3j, 3+4j])
    try:
        result = math_utils.safe_array_std(complex_array)
        logger.info(f"Complex array handling successful, result: {result}")
    except Exception as e:
        logger.error(f"Failed on complex array: {e}")
        return False
    
    # Test with NaN values
    nan_array = np.array([1.0, 2.0, np.nan, 4.0])
    try:
        result = math_utils.safe_array_std(nan_array)
        logger.info(f"NaN array handling successful, result: {result}")
    except Exception as e:
        logger.error(f"Failed on NaN array: {e}")
        return False
        
    logger.info("All math_utils tests passed!")
    return True

def test_stats_utils():
    logger.info("Testing fixed_stats_utils.py")
    
    # Import the fixed module
    import fixed_stats_utils as stats_utils
    
    # Test with regular arrays
    test_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = stats_utils.safe_array_std(test_array)
    assert isinstance(result, float)
    
    # Test with ddof parameter
    result_ddof0 = stats_utils.safe_array_std(test_array, ddof=0)
    result_ddof1 = stats_utils.safe_array_std(test_array, ddof=1)
    assert result_ddof0 != result_ddof1  # Should be different with different ddof
    
    # Test with problematic data types
    complex_array = np.array([1+2j, 2+3j, 3+4j])
    try:
        result = stats_utils.safe_array_std(complex_array)
        logger.info(f"Complex array handling successful, result: {result}")
    except Exception as e:
        logger.error(f"Failed on complex array: {e}")
        return False
    
    # Test with different axis
    multi_array = np.array([[1, 2, 3], [4, 5, 6]])
    try:
        result0 = stats_utils.safe_array_std(multi_array, axis=0)
        result1 = stats_utils.safe_array_std(multi_array, axis=1)
        logger.info(f"Multi-dimensional array handling successful, results: {result0}, {result1}")
    except Exception as e:
        logger.error(f"Failed on multi-dimensional array: {e}")
        return False
    
    logger.info("All stats_utils tests passed!")
    return True

def replace_files():
    """Replace original files with fixed versions if tests pass."""
    backup_ext = ".bak"
    
    # Create backups first
    shutil.copy2("math_utils.py", f"math_utils.py{backup_ext}")
    shutil.copy2("stats_utils.py", f"stats_utils.py{backup_ext}")
    logger.info("Backups created")
    
    # Replace files
    shutil.copy2("fixed_math_utils.py", "math_utils.py")
    shutil.copy2("fixed_stats_utils.py", "stats_utils.py")
    logger.info("Files replaced with fixed versions")

if __name__ == "__main__":
    logger.info("Starting verification tests")
    
    math_utils_ok = test_math_utils()
    stats_utils_ok = test_stats_utils()
    
    if math_utils_ok and stats_utils_ok:
        logger.info("All tests passed! Replacing files...")
        replace_files()
        logger.info("Done! The files have been successfully fixed and replaced.")
    else:
        logger.error("Tests failed. Files not replaced.")
