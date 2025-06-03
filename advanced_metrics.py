"""
Advanced Evaluation Metrics Module

This module implements comprehensive evaluation metrics for soccer match
prediction models, including Brier score, profit/loss simulation, and
specialized soccer betting metrics.

Key features:
1. Brier score for probability calibration assessment
2. Profit/loss simulation with betting strategies
3. Expected Value (EV) calculations
4. Kelly Criterion bet sizing
5. Sharpe ratio for betting performance
6. ROC curves for multiclass problems
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union
import logging
from sklearn.metrics import (
    roc_auc_score, log_loss, accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

class SoccerBettingMetrics:
    """
    Comprehensive evaluation metrics specifically designed for soccer betting
    and match outcome prediction assessment.
    """
    
    def __init__(self, outcome_names: Optional[List[str]] = None):
        """
        Initialize the metrics calculator.
        
        Args:
            outcome_names: Names for outcomes (e.g., ['Home Win', 'Draw', 'Away Win'])
        """
        self.outcome_names = outcome_names or ['Home Win', 'Draw', 'Away Win']
        self.n_classes = len(self.outcome_names)
        
    def brier_score_multiclass(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
        """
        Calculate Brier score for multiclass predictions.
        
        Args:
            y_true: True class labels (shape: n_samples)
            y_prob: Predicted probabilities (shape: n_samples, n_classes)
            
        Returns:
            brier_scores: Dictionary with overall and per-class Brier scores
        """
        n_samples, n_classes = y_prob.shape
        
        # Convert true labels to one-hot encoding
        y_true_onehot = np.zeros((n_samples, n_classes))
        y_true_onehot[np.arange(n_samples), y_true] = 1
        
        # Calculate overall Brier score
        overall_brier = np.mean(np.sum((y_prob - y_true_onehot) ** 2, axis=1))
        
        # Calculate per-class Brier scores
        per_class_brier = {}
        for i, outcome_name in enumerate(self.outcome_names):
            class_brier = np.mean((y_prob[:, i] - y_true_onehot[:, i]) ** 2)
            per_class_brier[outcome_name] = class_brier
        
        return {
            'overall': overall_brier,
            'per_class': per_class_brier,
            'decomposition': self._brier_score_decomposition(y_true, y_prob)
        }
    
    def _brier_score_decomposition(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
        """
        Decompose Brier score into reliability, resolution, and uncertainty components.
        
        Args:
            y_true: True class labels
            y_prob: Predicted probabilities
            
        Returns:
            decomposition: Dictionary with Brier score components
        """
        # Use the most probable class for decomposition
        y_pred_prob = np.max(y_prob, axis=1)
        y_pred_class = np.argmax(y_prob, axis=1)
        y_binary = (y_true == y_pred_class).astype(int)
        
        # Sort by predicted probability
        sorted_indices = np.argsort(y_pred_prob)
        sorted_probs = y_pred_prob[sorted_indices]
        sorted_outcomes = y_binary[sorted_indices]
        
        # Create bins
        n_bins = min(10, len(np.unique(sorted_probs)))
        bins = np.linspace(0, 1, n_bins + 1)
        
        reliability = 0
        resolution = 0
        n_total = len(y_true)
        base_rate = np.mean(y_binary)
        
        for i in range(n_bins):
            # Find samples in this bin
            if i == n_bins - 1:
                in_bin = (sorted_probs >= bins[i]) & (sorted_probs <= bins[i + 1])
            else:
                in_bin = (sorted_probs >= bins[i]) & (sorted_probs < bins[i + 1])
            
            if np.sum(in_bin) > 0:
                n_bin = np.sum(in_bin)
                prob_bin = np.mean(sorted_probs[in_bin])
                outcome_bin = np.mean(sorted_outcomes[in_bin])
                
                # Reliability: how far off are the probabilities?
                reliability += (n_bin / n_total) * (prob_bin - outcome_bin) ** 2
                
                # Resolution: how much do the conditional event rates vary?
                resolution += (n_bin / n_total) * (outcome_bin - base_rate) ** 2
        
        uncertainty = base_rate * (1 - base_rate)
        
        return {
            'reliability': reliability,
            'resolution': resolution, 
            'uncertainty': uncertainty,
            'brier_score': reliability - resolution + uncertainty
        }
    
    def profit_loss_simulation(self, y_true: np.ndarray, y_prob: np.ndarray,
                              odds_matrix: np.ndarray, stake: float = 10.0,
                              min_probability: float = 0.05,
                              max_probability: float = 0.95) -> Dict[str, Any]:
        """
        Simulate profit/loss using betting strategy based on predicted probabilities.
        
        Args:
            y_true: True outcomes
            y_prob: Predicted probabilities
            odds_matrix: Betting odds for each outcome (shape: n_samples, n_classes)
            stake: Base stake amount
            min_probability: Minimum probability to place a bet
            max_probability: Maximum probability to place a bet (avoid certainties)
            
        Returns:
            simulation_results: Detailed P&L simulation results
        """
        n_samples = len(y_true)
        
        # Calculate expected value for each bet
        expected_values = []
        bet_decisions = []
        profits = []
        
        for i in range(n_samples):
            sample_ev = []
            for j in range(self.n_classes):
                prob = y_prob[i, j]
                odds = odds_matrix[i, j]
                
                # Expected value = (prob * (odds - 1)) - ((1 - prob) * 1)
                # Simplified: prob * odds - 1
                ev = prob * odds - 1
                sample_ev.append(ev)
            
            expected_values.append(sample_ev)
            
            # Betting decision: bet on outcome with highest positive EV
            best_ev_idx = np.argmax(sample_ev)
            best_ev = sample_ev[best_ev_idx]
            best_prob = y_prob[i, best_ev_idx]
            
            # Only bet if EV is positive and probability is in reasonable range
            if (best_ev > 0 and 
                min_probability <= best_prob <= max_probability):
                bet_decisions.append((i, best_ev_idx, stake))
                
                # Calculate profit/loss
                if y_true[i] == best_ev_idx:
                    # Win: get back stake + winnings
                    profit = stake * (odds_matrix[i, best_ev_idx] - 1)
                else:
                    # Loss: lose stake
                    profit = -stake
                profits.append(profit)
            else:
                bet_decisions.append(None)
                profits.append(0)
        
        # Calculate statistics
        total_bets = sum(1 for bet in bet_decisions if bet is not None)
        total_profit = sum(profits)
        total_staked = total_bets * stake
        
        if total_bets > 0:
            win_rate = sum(1 for p in profits if p > 0) / total_bets
            avg_profit_per_bet = total_profit / total_bets
            roi = total_profit / total_staked if total_staked > 0 else 0
        else:
            win_rate = 0
            avg_profit_per_bet = 0
            roi = 0
        
        return {
            'total_profit': total_profit,
            'total_bets': total_bets,
            'total_staked': total_staked,
            'win_rate': win_rate,
            'roi': roi,
            'avg_profit_per_bet': avg_profit_per_bet,
            'sharpe_ratio': self._calculate_sharpe_ratio(profits) if profits else 0,
            'max_drawdown': self._calculate_max_drawdown(profits),
            'bet_distribution': self._analyze_bet_distribution(bet_decisions, y_true),
            'daily_profits': profits        }
    
    def _calculate_sharpe_ratio(self, profits: List[float]) -> float:
        """Calculate Sharpe ratio for betting performance."""
        if len(profits) < 2:
            return 0
        
        returns = np.array(profits)
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        return float(mean_return / std_return) if std_return > 0 else 0.0
    
    def _calculate_max_drawdown(self, profits: List[float]) -> float:
        """Calculate maximum drawdown from peak to trough."""
        if not profits:
            return 0
        
        cumulative = np.cumsum(profits)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = running_max - cumulative
        
        return np.max(drawdowns) if len(drawdowns) > 0 else 0
    
    def _analyze_bet_distribution(self, bet_decisions: List, y_true: np.ndarray) -> Dict[str, int]:
        """Analyze distribution of bets across outcomes."""
        distribution = {outcome: 0 for outcome in self.outcome_names}
        
        for bet in bet_decisions:
            if bet is not None:
                _, outcome_idx, _ = bet
                distribution[self.outcome_names[outcome_idx]] += 1
        
        return distribution
    
    def kelly_criterion_sizing(self, y_prob: np.ndarray, odds_matrix: np.ndarray,
                               bankroll: float = 1000.0) -> np.ndarray:
        """
        Calculate optimal bet sizes using Kelly Criterion.
        
        Args:
            y_prob: Predicted probabilities
            odds_matrix: Betting odds
            bankroll: Total bankroll amount
            
        Returns:
            bet_sizes: Optimal bet sizes for each prediction
        """
        n_samples, n_classes = y_prob.shape
        bet_sizes = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            for j in range(n_classes):
                prob = y_prob[i, j]
                odds = odds_matrix[i, j]
                
                if odds > 1:  # Valid odds
                    # Kelly formula: f = (bp - q) / b
                    # where b = odds - 1, p = probability, q = 1 - p
                    b = odds - 1
                    p = prob
                    q = 1 - p
                    
                    kelly_fraction = (b * p - q) / b
                    
                    # Ensure non-negative and reasonable maximum
                    kelly_fraction = max(0, min(kelly_fraction, 0.25))  # Cap at 25% of bankroll
                    
                    bet_sizes[i, j] = kelly_fraction * bankroll
        
        return bet_sizes
    
    def comprehensive_evaluation(self, y_true: np.ndarray, y_prob: np.ndarray,
                                odds_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of model performance.
        
        Args:
            y_true: True class labels
            y_prob: Predicted probabilities  
            odds_matrix: Optional betting odds matrix
            
        Returns:
            evaluation_results: Comprehensive evaluation metrics
        """
        # Basic classification metrics
        y_pred = np.argmax(y_prob, axis=1)
        
        basic_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'log_loss': log_loss(y_true, y_prob),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
          # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(self.n_classes)
        )
        
        # Ensure arrays are proper format
        if not isinstance(precision, np.ndarray):
            precision = np.array([precision])
        if not isinstance(recall, np.ndarray):
            recall = np.array([recall])
        if not isinstance(f1, np.ndarray):
            f1 = np.array([f1])
        if not isinstance(support, np.ndarray):
            support = np.array([support])
        
        per_class_metrics = {}
        for i, outcome_name in enumerate(self.outcome_names):
            per_class_metrics[outcome_name] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
          # ROC AUC (one-vs-rest)
        roc_auc: Optional[float] = None
        try:
            y_true_binarized = label_binarize(y_true, classes=range(self.n_classes))
            if self.n_classes == 2:
                roc_auc = float(roc_auc_score(y_true, y_prob[:, 1]))
            else:
                roc_auc = float(roc_auc_score(y_true_binarized, y_prob, multi_class='ovr', average='weighted'))
        except Exception as e:
            logger.warning(f"Could not calculate ROC AUC: {e}")
            roc_auc = None
        
        # Brier score
        brier_results = self.brier_score_multiclass(y_true, y_prob)
        
        # Compile results
        results = {
            'basic_metrics': basic_metrics,
            'per_class_metrics': per_class_metrics,
            'roc_auc': roc_auc,
            'brier_score': brier_results,
            'class_distribution': {
                'actual': np.bincount(y_true).tolist(),
                'predicted': np.bincount(y_pred).tolist()
            }
        }
        
        # Add betting simulation if odds provided
        if odds_matrix is not None:
            betting_results = self.profit_loss_simulation(y_true, y_prob, odds_matrix)
            results['betting_simulation'] = betting_results
        
        return results
    
    def plot_calibration_curves(self, y_true: np.ndarray, y_prob: np.ndarray,
                               save_path: Optional[str] = None) -> None:
        """
        Plot reliability diagrams for each outcome.
        
        Args:
            y_true: True class labels
            y_prob: Predicted probabilities
            save_path: Optional path to save the plot
        """
        from sklearn.calibration import calibration_curve
        
        fig, axes = plt.subplots(1, self.n_classes, figsize=(5 * self.n_classes, 5))
        if self.n_classes == 1:
            axes = [axes]
        
        for i, (outcome_name, ax) in enumerate(zip(self.outcome_names, axes)):
            # Create binary labels for current outcome
            y_binary = (y_true == i).astype(int)
            
            # Calculate calibration curve
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_binary, y_prob[:, i], n_bins=10
            )
            
            # Plot
            ax.plot(mean_predicted_value, fraction_of_positives, 
                   marker='o', linewidth=2, label=f'{outcome_name}')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray', 
                   label='Perfect calibration')
            
            ax.set_xlabel('Mean Predicted Probability')
            ax.set_ylabel('Fraction of Positives')
            ax.set_title(f'Calibration Curve - {outcome_name}')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Calibration curves saved to {save_path}")
        else:
            plt.show()
    
    def plot_profit_curve(self, profits: List[float], save_path: Optional[str] = None) -> None:
        """
        Plot cumulative profit curve over time.
        
        Args:
            profits: List of daily profits
            save_path: Optional path to save the plot
        """
        if not profits:
            logger.warning("No profits data to plot")
            return
        
        cumulative_profits = np.cumsum(profits)
        
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_profits, linewidth=2)
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        plt.xlabel('Bet Number')
        plt.ylabel('Cumulative Profit')
        plt.title('Betting Strategy Performance Over Time')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        final_profit = cumulative_profits[-1]
        max_profit = np.max(cumulative_profits)
        min_profit = np.min(cumulative_profits)
        
        plt.text(0.02, 0.98, 
                f'Final Profit: {final_profit:.2f}\n'
                f'Max Profit: {max_profit:.2f}\n'
                f'Min Profit: {min_profit:.2f}',
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Profit curve saved to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_classes = 3
    
    # Create predictions and true outcomes
    y_prob = np.random.dirichlet([2, 1, 2], n_samples)  # Somewhat realistic probabilities
    y_true = np.random.choice(n_classes, n_samples, p=[0.45, 0.25, 0.30])
    
    # Create sample odds matrix (typical soccer odds)
    odds_matrix = np.random.uniform(1.5, 5.0, (n_samples, n_classes))
    
    # Initialize metrics calculator
    metrics = SoccerBettingMetrics(['Home Win', 'Draw', 'Away Win'])
    
    # Comprehensive evaluation
    results = metrics.comprehensive_evaluation(y_true, y_prob, odds_matrix)
    
    print("=== COMPREHENSIVE EVALUATION RESULTS ===")
    print(f"Accuracy: {results['basic_metrics']['accuracy']:.4f}")
    print(f"Log Loss: {results['basic_metrics']['log_loss']:.4f}")
    print(f"ROC AUC: {results['roc_auc']:.4f}")
    print(f"Overall Brier Score: {results['brier_score']['overall']:.4f}")
    
    print("\n=== BETTING SIMULATION ===")
    betting = results['betting_simulation']
    print(f"Total Profit: {betting['total_profit']:.2f}")
    print(f"ROI: {betting['roi']:.2%}")
    print(f"Win Rate: {betting['win_rate']:.2%}")
    print(f"Sharpe Ratio: {betting['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {betting['max_drawdown']:.2f}")
    
    print("\n=== PER-CLASS METRICS ===")
    for outcome, metrics_dict in results['per_class_metrics'].items():
        print(f"{outcome}: Precision={metrics_dict['precision']:.3f}, "
              f"Recall={metrics_dict['recall']:.3f}, F1={metrics_dict['f1_score']:.3f}")
