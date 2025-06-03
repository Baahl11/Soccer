import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from fnn_model import FeedforwardNeuralNetwork
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_synthetic_data(n_samples=3000, n_features=10, random_state=42):
    """Create synthetic data with overlapping classes and noise"""
    np.random.seed(random_state)
    
    # Generate balanced classes
    n_per_class = n_samples // 2
    
    # Class 0: Base distribution
    X0 = np.random.normal(0, 1, (n_per_class, n_features))
    y0 = np.zeros(n_per_class)
    
    # Class 1: Partially overlapping distribution
    X1 = np.random.normal(0.5, 1.2, (n_per_class, n_features))
    # Add some non-linear patterns
    X1[:, :3] += 0.3 * np.sin(X1[:, 3:6])  # Sinusoidal interaction
    X1[:, 3:6] += 0.2 * np.square(X1[:, :3])  # Quadratic interaction
    y1 = np.ones(n_per_class)
    
    # Add noise
    noise_factor = 0.1
    X0 += np.random.normal(0, noise_factor, X0.shape)
    X1 += np.random.normal(0, noise_factor, X1.shape)
    
    # Combine data
    X = np.vstack((X0, X1))
    y = np.hstack((y0, y1))
    
    # Shuffle the data
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    return X, y

def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history"""
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def evaluate_model(model, X, y, scaler, prefix="", output_dir="results"):
    """Evaluate model performance with comprehensive metrics and enhanced visualizations"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        predictions_prob = model.predict(X_scaled)
        predictions = (predictions_prob > 0.5).astype(int)
        
        # Calculate metrics
        conf_matrix = confusion_matrix(y, predictions)
        class_report = classification_report(y, predictions, output_dict=True)
        assert isinstance(class_report, dict)  # Type assertion for type checker
        auc_score = roc_auc_score(y, predictions_prob)
        
        # Calculate detailed metrics
        tn, fp, fn, tp = conf_matrix.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        # Calculate class balance
        class_balance = np.bincount(y.astype(int)) / len(y)
        
        # Log results
        logger.info(f"\n{prefix} Performance Metrics")
        logger.info("="*50)
        logger.info("Overall Performance:")
        logger.info(f"  Accuracy: {class_report['accuracy']:.4f}")
        logger.info(f"  AUC-ROC: {auc_score:.4f}")
        logger.info(f"  F1 Score: {f1:.4f}")
        
        logger.info("\nClass Distribution:")
        logger.info(f"  Class 0: {class_balance[0]:.2%}")
        logger.info(f"  Class 1: {class_balance[1]:.2%}")
        
        logger.info("\nDetailed Performance:")
        logger.info("Class 0 (Negative):")
        logger.info(f"  Specificity: {specificity:.4f}")
        logger.info(f"  NPV: {tn/(tn+fn):.4f}")
        logger.info(f"  True Negatives: {tn}, False Positives: {fp}")
        
        logger.info("Class 1 (Positive):")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  True Positives: {tp}, False Negatives: {fn}")
        
        # Save visualizations
        try:
            # Confusion Matrix
            plt.figure(figsize=(10, 8))
            cm_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues',
                       xticklabels=['Class 0', 'Class 1'],
                       yticklabels=['Class 0', 'Class 1'])
            plt.title(f'{prefix} Confusion Matrix (Normalized)')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(os.path.join(output_dir, f'{prefix.lower()}_confusion_matrix.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()
            
            # Performance Curves
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # ROC Curve
            fpr, tpr, _ = roc_curve(y, predictions_prob)
            ax1.plot(fpr, tpr, label=f'ROC (AUC = {auc_score:.3f})')
            ax1.plot([0, 1], [0, 1], 'k--', label='Random')
            ax1.set_xlabel('False Positive Rate')
            ax1.set_ylabel('True Positive Rate')
            ax1.set_title(f'{prefix} ROC Curve')
            ax1.legend()
            ax1.grid(True)
            
            # Precision-Recall Curve
            precision_curve, recall_curve, _ = precision_recall_curve(y, predictions_prob)
            avg_precision = average_precision_score(y, predictions_prob)
            ax2.plot(recall_curve, precision_curve,
                    label=f'PR (AP = {avg_precision:.3f})')
            ax2.axhline(y=class_balance[1], color='r', linestyle='--',
                       label=f'Baseline ({class_balance[1]:.3f})')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.set_title(f'{prefix} Precision-Recall Curve')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{prefix.lower()}_curves.png'),
                       bbox_inches='tight', dpi=300)
            plt.close()
            
        except Exception as viz_error:
            logger.warning(f"Error generating visualizations: {viz_error}")
        
        return {
            'accuracy': class_report['accuracy'],
            'auc': auc_score,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'confusion_matrix': conf_matrix,
            'predictions': predictions_prob
        }
        
    except Exception as e:
        logger.error(f"Error in model evaluation: {e}")
        raise

def test_model():
    try:
        # Generate more challenging data
        logger.info("Generating synthetic data...")
        X, y = create_synthetic_data(n_samples=5000)  # More samples
        
        # Split and scale data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Train and evaluate model
        logger.info("Training model...")
        model = FeedforwardNeuralNetwork(input_dim=10)
        history = model.train(
            X_train_scaled, y_train,
            X_val=X_val_scaled,
            y_val=y_val,
            epochs=100,  # More epochs for harder problem
            batch_size=32  # Smaller batch size for better generalization
        )
        
        # Plot training history
        plot_training_history(history)
        
        # Evaluate model
        logger.info("\nEvaluating model performance...")
        train_metrics = evaluate_model(model, X_train, y_train, scaler, "Training")
        val_metrics = evaluate_model(model, X_val, y_val, scaler, "Validation")
        
        return True
            
    except Exception as e:
        logger.error(f"Error in model testing: {e}")
        return False

if __name__ == "__main__":
    success = test_model()
    exit(0 if success else 1)