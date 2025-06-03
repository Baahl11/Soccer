"""
Helper module for loading FeedforwardNeuralNetwork models
"""
import os
import logging
from keras import Model, utils, models
from typing import Tuple, Any, Optional
from fnn_model import FeedforwardNeuralNetwork, custom_poisson_loss, poisson_loss_metric
import joblib

logger = logging.getLogger(__name__)

def load_fnn_model_safe(model_path: str, scaler_path: str, input_dim: int = 14) -> Tuple[Any, Optional[Any]]:
    """
    Safely load a FeedforwardNeuralNetwork model and scaler
    
    Args:
        model_path: Path to the model file (.h5 format)
        scaler_path: Path to the scaler file (.pkl format)
        input_dim: Input dimension for the model
    
    Returns:
        Tuple of (model, scaler)
    """
    try:
        # Load scaler first
        scaler = None
        try:
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                logger.info(f"Scaler loaded from {scaler_path}")
            else:
                logger.warning(f"Scaler file not found at {scaler_path}")
        except Exception as scaler_error:
            logger.warning(f"Failed to load scaler: {scaler_error}")
        
        # Create a new model instance 
        model = FeedforwardNeuralNetwork(input_dim=input_dim)            # Try to load the model weights if the file exists
        try:
            if os.path.exists(model_path):
                # Define custom objects dict with our custom functions
                custom_objects = {
                    'custom_poisson_loss': custom_poisson_loss,
                    'poisson_loss_metric': poisson_loss_metric
                }
                
                # Load the model with custom objects
                with utils.custom_object_scope(custom_objects):
                    loaded_model = models.load_model(model_path)
                
                # Assign the loaded model to our instance using proper type casting
                from typing import cast
                model.model = cast(Model, loaded_model)
                logger.info(f"Model loaded successfully with custom loss functions from {model_path}")
            else:
                logger.warning(f"Model file not found at {model_path}, using new model")
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")
            logger.info("Using newly initialized model")
        
        return model, scaler
    
    except Exception as e:
        logger.error(f"General error loading model and scaler: {e}")
        # Create a new model as fallback
        fallback_model = FeedforwardNeuralNetwork(input_dim=input_dim)
        return fallback_model, None
