"""
Test script to verify that the fixed neural network model is working correctly
and producing varied predictions.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
import logging
from fnn_model import FeedforwardNeuralNetwork
from fnn_model_fixed import FeedforwardNeuralNetworkFixed

# Configure logging to both console and file
log_file_path = 'fnn_model_test.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Run tests on both the original and fixed neural network models."""
    logger.info("Testing neural network models for prediction variability...")
    
    # Check if models exist
    if not os.path.exists('models/fnn_model_fixed.pkl'):
        logger.error("models/fnn_model_fixed.pkl not found!")
        return
    
    if not os.path.exists('models/fnn_model.pkl'):
        logger.warning("Original model models/fnn_model.pkl not found. Will only test fixed model.")
    
    # Load scaler
    if not os.path.exists('models/scaler.pkl'):
        logger.error("models/scaler.pkl not found!")
        return
    
    scaler = joblib.load('models/scaler.pkl')
    logger.info("Scaler loaded successfully.")
    
    # Create test data
    logger.info("Creating test data...")
    test_data = []
    
    # Create some example feature vectors for testing
    # These should be the same feature vector repeated multiple times
    base_vector = [1.2, 1.1, 0.6, 0.4, 1.5, 1.2, 0.5, 0.3, 1.3, 1.4, 0.4, 2.5, 1.4, 1.3]
    
    # Create 10 identical copies to test for variability
    for _ in range(10):
        test_data.append(base_vector.copy())
    
    test_data = np.array(test_data)
    
    # Scale the data
    test_data_scaled = scaler.transform(test_data)
    
    # Test the original model if available
    if os.path.exists('models/fnn_model.pkl'):
        try:
            logger.info("\n=== TESTING ORIGINAL MODEL ===")
            model_dict = joblib.load('models/fnn_model.pkl')
            input_dim = model_dict.get('input_dim', 14)
            
            orig_model = FeedforwardNeuralNetwork(input_dim=input_dim)
            orig_model.model.set_weights(model_dict['weights'])
            
            # Get predictions from original model
            orig_preds = orig_model.predict(test_data_scaled)
            
            # Print the first few predictions
            logger.info("Original model predictions (should be identical or very similar):")
            for i, pred in enumerate(orig_preds[:5]):
                logger.info(f"  Prediction {i+1}: Home xG = {pred[0]:.3f}, Away xG = {pred[1]:.3f}")
            
            # Check for variability
            rounded_preds = [(round(p[0], 3), round(p[1], 3)) for p in orig_preds]
            unique_preds = set(rounded_preds)
            
            logger.info(f"Variability: {len(unique_preds)} unique values in {len(orig_preds)} predictions")
            if len(unique_preds) <= 3:
                logger.warning("ORIGINAL MODEL HAS LOW VARIABILITY - this is the expected issue")
            
        except Exception as e:
            logger.error(f"Error testing original model: {e}")
    
    # Test the fixed model
    try:
        logger.info("\n=== TESTING FIXED MODEL ===")
        fixed_model_dict = joblib.load('models/fnn_model_fixed.pkl')
        input_dim = fixed_model_dict.get('input_dim', 14)
        
        fixed_model = FeedforwardNeuralNetworkFixed(input_dim=input_dim)
        fixed_model.load_weights(fixed_model_dict['weights'])
        
        # Get predictions from fixed model
        fixed_preds = fixed_model.predict(test_data_scaled)
        
        # Print the first few predictions
        logger.info("Fixed model predictions (should show variability):")
        for i, pred in enumerate(fixed_preds[:5]):
            logger.info(f"  Prediction {i+1}: Home xG = {pred[0]:.3f}, Away xG = {pred[1]:.3f}")
        
        # Check for variability
        rounded_preds = [(round(p[0], 3), round(p[1], 3)) for p in fixed_preds]
        unique_preds = set(rounded_preds)
        
        logger.info(f"Variability: {len(unique_preds)} unique values in {len(fixed_preds)} predictions")
        if len(unique_preds) >= len(fixed_preds) * 0.8:
            logger.info("FIXED MODEL SHOWS GOOD VARIABILITY ✓")
        else:
            logger.warning("FIXED MODEL STILL HAS LIMITED VARIABILITY ⚠")
    
    except Exception as e:
        logger.error(f"Error testing fixed model: {e}")

if __name__ == "__main__":
    main()
