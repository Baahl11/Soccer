import psutil
import os
import logging
import json
from train_voting_ensemble import (
    load_corner_training_data,
    preprocess_corner_data,
    train_random_forest_corners_model,
    train_xgboost_corners_model,
    DynamicWeightEnsemble
)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def log_memory_usage(message: str):
    """Log current memory usage"""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"{message} - Memory usage: {memory_mb:.2f} MB")

def test_optimizations():
    """Test memory optimizations in model training pipeline"""
    try:
        # Initial memory usage
        log_memory_usage("Initial state")

        # Load data
        logger.info("Loading training data...")
        data = load_corner_training_data()
        log_memory_usage("After data loading")

        if data.empty:
            logger.error("No training data available")
            return

        # Preprocess data with batch processing
        logger.info("Preprocessing data with batch processing...")
        batch_sizes = [1000, 500, 100]  # Test different batch sizes
        for batch_size in batch_sizes:
            logger.info(f"\nTesting batch size: {batch_size}")
            X_train, X_test, y_train, y_test = preprocess_corner_data(data, batch_size=batch_size)
            log_memory_usage(f"After preprocessing (batch_size={batch_size})")

            # Train models incrementally
            logger.info("Training Random Forest incrementally...")
            rf_model = train_random_forest_corners_model(
                X_train, y_train,
                batch_size=batch_size,
                tune_hyperparams=False
            )
            log_memory_usage("After RF training")

            logger.info("Training XGBoost incrementally...")
            xgb_model = train_xgboost_corners_model(
                X_train, y_train,
                tune_hyperparams=False
            )
            log_memory_usage("After XGBoost training")

            # Create dynamic ensemble
            logger.info("Creating dynamic ensemble...")
            ensemble = DynamicWeightEnsemble(rf_model, xgb_model, window_size=10)
            
            # Test prediction
            logger.info("Testing predictions...")
            y_pred = ensemble.predict(X_test, y_test)
            log_memory_usage("After predictions")

            # Save results
            results = {
                'batch_size': batch_size,
                'data_shape': X_train.shape,
                'predictions_shape': y_pred.shape
            }
            
            os.makedirs('results', exist_ok=True)
            with open(f'results/memory_test_batch_{batch_size}.json', 'w') as f:
                json.dump(results, f, indent=2)

        logger.info("\nOptimization tests completed successfully")

    except Exception as e:
        logger.error(f"Error in optimization tests: {e}", exc_info=True)

if __name__ == '__main__':
    test_optimizations()
