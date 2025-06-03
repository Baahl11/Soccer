"""
Enhanced Prediction System Evaluator

This script evaluates the enhanced prediction system against the original model,
with a specific focus on draw prediction performance. It produces comparative
visualizations and metrics to demonstrate improvements.
"""

import os
import argparse
import logging
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

# Import local modules
from real_data_processor import RealDataProcessor
from match_winner import predict_match_winner, MatchOutcome
from enhanced_match_winner import EnhancedPredictionSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simple visualizer class since the actual one appears to be missing
class PredictionVisualizer:
    """Utility class for visualizing prediction results."""
    
    def __init__(self):
        pass
        
    def plot_confusion_matrix(self, cm, labels, title='Confusion Matrix'):
        """Plot a confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True')
        plt.xlabel('Predicted')
        plt.tight_layout()

class EnhancedSystemEvaluator:
    """
    Evaluator for comparing the enhanced prediction system against the original model.
    """
    def __init__(self, output_dir: str = "comparison_results"):
        """
        Initialize the evaluator.
        
        Args:
            output_dir: Directory for saving evaluation results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.data_processor = RealDataProcessor()
        self.visualizer = PredictionVisualizer()
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.enhanced_system = EnhancedPredictionSystem()
        
    def load_test_data(
        self, 
        league_id: int, 
        season: int, 
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Load test data for evaluation.
        
        Args:
            league_id: League ID
            season: Season year
            limit: Optional limit on number of matches
            
        Returns:
            DataFrame with test matches
        """
        logger.info(f"Loading test data for league {league_id}, season {season}")
        
        # Try to load data for the specified season
        df = self.data_processor.load_historical_matches(league_id, season, limit)
        
        # If no data is available for the specified season, try the previous season
        if df.empty:
            logger.warning(f"No data found for league {league_id}, season {season}. Trying previous season.")
            df = self.data_processor.load_historical_matches(league_id, season-1, limit)
            
        # If still no data, use a known good season as fallback
        if df.empty:
            fallback_season = 2022
            logger.warning(f"Still no data found. Using fallback season {fallback_season}.")
            df = self.data_processor.load_historical_matches(league_id, fallback_season, limit)
            
        # Add league_id column if missing
        if not df.empty and 'league_id' not in df.columns:
            df['league_id'] = league_id
            
        return df
        
    def evaluate_models(
        self,
        test_data: pd.DataFrame
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Evaluate both original and enhanced models on test data.
        
        Args:
            test_data: DataFrame with test matches
            
        Returns:
            Tuple of (original results, enhanced results)
        """
        logger.info(f"Evaluating models on {len(test_data)} matches")
        
        # Prepare structures for results
        original_results = {
            'predictions': [],
            'actuals': [],
            'match_details': []
        }
        
        enhanced_results = {
            'predictions': [],
            'actuals': [],
            'match_details': []
        }
        
        # Process each match
        for _, match in test_data.iterrows():
            try:
                home_team_id = int(match['home_team_id'])
                away_team_id = int(match['away_team_id'])
                league_id = int(match['league_id'])
                fixture_id = match.get('fixture_id', None)                # Get actual outcome
                home_goals = match['home_goals']
                away_goals = match['away_goals']
                
                if home_goals > away_goals:
                    actual_outcome = MatchOutcome.HOME_WIN.value
                elif away_goals > home_goals:
                    actual_outcome = MatchOutcome.AWAY_WIN.value
                else:
                    actual_outcome = MatchOutcome.DRAW.value
                
                # Get form data for both teams
                try:
                    # Using internal method with leading underscore from RealDataProcessor
                    home_form = self.data_processor._get_team_form_data(home_team_id)
                    away_form = self.data_processor._get_team_form_data(away_team_id)
                
                    # Get head-to-head data
                    h2h = self.data_processor._get_head_to_head_data(home_team_id, away_team_id)
                except AttributeError as e:
                    logger.warning(f"Error accessing form data methods: {e}. Using default values.")
                    # Create default form and h2h data
                    home_form = {
                        'form_trend': 0.0,
                        'matches_played': 5,
                        'consistency': 0.5,
                        'expected_goals_avg': 1.3
                    }
                    away_form = {
                        'form_trend': 0.0,
                        'matches_played': 5,
                        'consistency': 0.5,
                        'expected_goals_avg': 1.1
                    }
                    h2h = {
                        'matches_played': 0,
                        'home_win_pct': 0.5,
                        'draw_pct': 0.25,
                        'away_win_pct': 0.25
                    }
                
                # Calculate expected goals based on form
                home_xg = home_form.get('expected_goals_avg', 1.3)
                away_xg = away_form.get('expected_goals_avg', 1.1)
                
                # Get original prediction
                original_pred = predict_match_winner(
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    home_xg=home_xg,
                    away_xg=away_xg,
                    home_form=home_form,
                    away_form=away_form,
                    h2h=h2h,
                    league_id=league_id
                )
                
                # Get enhanced prediction
                enhanced_pred = self.enhanced_system.predict(
                    home_team_id=home_team_id,
                    away_team_id=away_team_id,
                    league_id=league_id,
                    home_xg=home_xg,
                    away_xg=away_xg,
                    home_form=home_form,
                    away_form=away_form,
                    h2h=h2h
                )
                
                # Store results
                original_results['predictions'].append(original_pred)
                original_results['actuals'].append(actual_outcome)
                
                enhanced_results['predictions'].append(enhanced_pred)
                enhanced_results['actuals'].append(actual_outcome)
                
                # Store match details for both
                match_detail = {
                    'fixture_id': fixture_id,
                    'home_team_id': home_team_id,
                    'away_team_id': int(match['away_team_id']),
                    'home_goals': int(home_goals),
                    'away_goals': int(away_goals),
                    'actual_outcome': actual_outcome,
                    'league_id': league_id,
                    'date': str(match.get('date', ''))
                }
                original_results['match_details'].append(match_detail)
                enhanced_results['match_details'].append(match_detail)
                
            except Exception as e:
                logger.error(f"Error processing match: {e}")
                continue
            
        return original_results, enhanced_results
    
    def calculate_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate evaluation metrics from results.
        
        Args:
            results: Results dictionary from evaluation
            
        Returns:
            Dictionary of metrics
        """
        if not results.get('predictions') or not results.get('actuals'):
            logger.warning("No predictions or actuals available for metric calculation")
            return {
                'overall_accuracy': 0.0,
                'confusion_matrix': [],
                'outcome_metrics': {},
                'outcome_labels': [o.value for o in MatchOutcome]
            }
            
        predictions = [p['predicted_outcome'] for p in results['predictions']]
        actuals = results['actuals']
        
        # Calculate confusion matrix
        from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
        
        outcome_labels = [o.value for o in MatchOutcome]
          # Check if predictions and actuals are not empty        
        if len(predictions) == 0 or len(actuals) == 0:
            logger.warning("Empty predictions or actuals list. Cannot calculate metrics.")
            return {
                'overall_accuracy': 0.0,
                'confusion_matrix': [],
                'outcome_metrics': {},
                'outcome_labels': outcome_labels
            }
            
        try:
            cm = confusion_matrix(
                actuals, 
                predictions,
                labels=outcome_labels
            )
        except Exception as e:
            logger.error(f"Error calculating confusion matrix: {e}")
            # Create a zero matrix as fallback
            cm = np.zeros((len(outcome_labels), len(outcome_labels)), dtype=int)
        
        # Calculate accuracy metrics
        overall_acc = accuracy_score(actuals, predictions)
        
        # Calculate accuracy for each outcome
        outcome_metrics = {}
        for i, outcome in enumerate(outcome_labels):
            # Count true occurrences of this outcome
            true_count = sum(1 for a in actuals if a == outcome)
            if true_count > 0:
                # Count correct predictions for this outcome
                correct_preds = sum(1 for p, a in zip(predictions, actuals) 
                                   if p == a and a == outcome)
                outcome_metrics[outcome] = {
                    'total': true_count,
                    'correct': correct_preds,
                    'accuracy': round(float(correct_preds) / float(true_count) * 100.0, 2)
                }
            else:
                outcome_metrics[outcome] = {'total': 0, 'correct': 0, 'accuracy': 0}
                
        # Get detailed classification report
        class_report = classification_report(
            actuals, 
            predictions,
            labels=outcome_labels,
            output_dict=True
        )
        
        # Calculate confidence correlation
        confidences = []
        correctness = []
        
        for pred, actual in zip(results['predictions'], actuals):
            pred_outcome = pred['predicted_outcome']
            confidence = pred['probabilities'].get(pred_outcome, 0)
            is_correct = 1 if pred_outcome == actual else 0
            
            confidences.append(confidence)
            correctness.append(is_correct)
            
        confidence_corr = np.corrcoef(confidences, correctness)[0, 1]
        
        # Compile metrics dictionary
        metrics = {
            'overall_accuracy': overall_acc * 100,
            'outcome_metrics': outcome_metrics,
            'confusion_matrix': cm.tolist(),
            'outcome_labels': outcome_labels,
            'confidence_correlation': confidence_corr,
            'classification_report': class_report
        }
        
        return metrics
    
    def compare_and_visualize(
        self,
        original_metrics: Dict[str, Any],
        enhanced_metrics: Dict[str, Any]
    ) -> None:
        """
        Generate comparative visualizations for the two models.
        
        Args:
            original_metrics: Metrics from original model
            enhanced_metrics: Metrics from enhanced model
        """
        # Set up the plot style
        sns.set_style("whitegrid")
        plt.rcParams.update({
            'figure.figsize': (14, 10),
            'font.size': 12
        })
        
        # 1. Compare overall accuracy
        plt.figure(figsize=(10, 6))
        models = ['Original Model', 'Enhanced Model']
        accuracies = [
            original_metrics['overall_accuracy'],
            enhanced_metrics['overall_accuracy']
        ]
        
        sns.barplot(x=models, y=accuracies)
        plt.title('Overall Accuracy Comparison', fontsize=16)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.ylim(0, 100)
        
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 1, f'{acc:.2f}%', ha='center', fontsize=13)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'comparison_{self.timestamp}_overall_acc.png'))
        plt.close()
        
        # 2. Compare outcome-specific accuracy
        plt.figure(figsize=(12, 7))
        
        outcome_labels = original_metrics['outcome_labels']
        original_accs = [original_metrics['outcome_metrics'][o]['accuracy'] for o in outcome_labels]
        enhanced_accs = [enhanced_metrics['outcome_metrics'][o]['accuracy'] for o in outcome_labels]
        
        width = 0.35
        x = np.arange(len(outcome_labels))
        
        plt.bar(x - width/2, original_accs, width, label='Original Model')
        plt.bar(x + width/2, enhanced_accs, width, label='Enhanced Model')
        
        plt.title('Accuracy by Outcome Type', fontsize=16)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.xlabel('Outcome', fontsize=14)
        plt.xticks(x, [o.replace('_', ' ').title() for o in outcome_labels])
        plt.ylim(0, 100)
        plt.legend()
        
        for i, acc in enumerate(original_accs):
            plt.text(i - width/2, acc + 1, f'{acc:.1f}%', ha='center', fontsize=11)
            
        for i, acc in enumerate(enhanced_accs):
            plt.text(i + width/2, acc + 1, f'{acc:.1f}%', ha='center', fontsize=11)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'comparison_{self.timestamp}_outcome_acc.png'))
        plt.close()
        
        # 3. Compare confusion matrices side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Original model confusion matrix
        sns.heatmap(
            original_metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[o.replace('_', ' ').title() for o in outcome_labels],
            yticklabels=[o.replace('_', ' ').title() for o in outcome_labels],
            ax=ax1
        )
        ax1.set_title('Original Model Confusion Matrix', fontsize=15)
        ax1.set_ylabel('True Outcome', fontsize=13)
        ax1.set_xlabel('Predicted Outcome', fontsize=13)
        
        # Enhanced model confusion matrix
        sns.heatmap(
            enhanced_metrics['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=[o.replace('_', ' ').title() for o in outcome_labels],
            yticklabels=[o.replace('_', ' ').title() for o in outcome_labels],
            ax=ax2
        )
        ax2.set_title('Enhanced Model Confusion Matrix', fontsize=15)
        ax2.set_ylabel('True Outcome', fontsize=13)
        ax2.set_xlabel('Predicted Outcome', fontsize=13)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'comparison_{self.timestamp}_confusion.png'))
        plt.close()
        
        # 4. Draw-specific improvement focus
        plt.figure(figsize=(10, 6))
        
        draw_idx = outcome_labels.index(MatchOutcome.DRAW.value)
        original_draw_cm = original_metrics['confusion_matrix'][draw_idx]
        enhanced_draw_cm = enhanced_metrics['confusion_matrix'][draw_idx]
        
        # Normalize to percentages
        original_draw_total = sum(original_draw_cm)
        enhanced_draw_total = sum(enhanced_draw_cm)
        
        if original_draw_total > 0 and enhanced_draw_total > 0:
            original_draw_pct = [100 * x / original_draw_total for x in original_draw_cm]
            enhanced_draw_pct = [100 * x / enhanced_draw_total for x in enhanced_draw_cm]
            
            labels = ['Predicted as\nHome Win', 'Predicted as\nDraw', 'Predicted as\nAway Win']
            x = np.arange(len(labels))
            
            plt.bar(x - width/2, original_draw_pct, width, label='Original Model')
            plt.bar(x + width/2, enhanced_draw_pct, width, label='Enhanced Model')
            
            plt.title('True Draw Outcomes: Prediction Distribution', fontsize=16)
            plt.ylabel('Percentage (%)', fontsize=14)
            plt.ylim(0, 100)
            plt.xticks(x, labels)
            plt.legend()
            
            for i, pct in enumerate(original_draw_pct):
                plt.text(i - width/2, pct + 1, f'{pct:.1f}%', ha='center', fontsize=11)
                
            for i, pct in enumerate(enhanced_draw_pct):
                plt.text(i + width/2, pct + 1, f'{pct:.1f}%', ha='center', fontsize=11)
                
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'comparison_{self.timestamp}_draw_improvement.png'))
            plt.close()
        
        # 5. Save detailed metrics in JSON format
        comparison = {
            'timestamp': self.timestamp,
            'original_model': original_metrics,
            'enhanced_model': enhanced_metrics,
            'draw_improvement': {
                'original_accuracy': original_metrics['outcome_metrics'].get(MatchOutcome.DRAW.value, {}).get('accuracy', 0),
                'enhanced_accuracy': enhanced_metrics['outcome_metrics'].get(MatchOutcome.DRAW.value, {}).get('accuracy', 0),
                'accuracy_change': enhanced_metrics['outcome_metrics'].get(MatchOutcome.DRAW.value, {}).get('accuracy', 0) - 
                                  original_metrics['outcome_metrics'].get(MatchOutcome.DRAW.value, {}).get('accuracy', 0)
            }
        }
        
        with open(os.path.join(self.output_dir, f'comparison_{self.timestamp}_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2)
            
        logger.info(f"Saved comparison results to {self.output_dir}")
        
    def run_evaluation(
        self, 
        league_id: int = 39,  # Premier League by default
        season: int = 2022,
        limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run full evaluation workflow.
        
        Args:
            league_id: League ID to evaluate
            season: Season year
            limit: Optional limit on number of matches
            
        Returns:
            Comparison metrics dictionary
        """
        # Load test data
        test_data = self.load_test_data(league_id, season, limit)
        
        if test_data.empty:
            logger.error("No test data available. Aborting evaluation.")
            return {}
            
        # Evaluate both models
        original_results, enhanced_results = self.evaluate_models(test_data)
        
        # Calculate metrics
        original_metrics = self.calculate_metrics(original_results)
        enhanced_metrics = self.calculate_metrics(enhanced_results)
        
        # Generate comparison visualizations
        self.compare_and_visualize(original_metrics, enhanced_metrics)
        
        comparison = {
            'timestamp': self.timestamp,
            'original_metrics': original_metrics,
            'enhanced_metrics': enhanced_metrics,
            'draw_improvement': {
                'original_accuracy': original_metrics['outcome_metrics'].get(MatchOutcome.DRAW.value, {}).get('accuracy', 0),
                'enhanced_accuracy': enhanced_metrics['outcome_metrics'].get(MatchOutcome.DRAW.value, {}).get('accuracy', 0),
                'accuracy_change': enhanced_metrics['outcome_metrics'].get(MatchOutcome.DRAW.value, {}).get('accuracy', 0) - 
                                  original_metrics['outcome_metrics'].get(MatchOutcome.DRAW.value, {}).get('accuracy', 0)
            }
        }
        
        return comparison

def main():
    parser = argparse.ArgumentParser(description='Enhanced Prediction System Evaluator')
    parser.add_argument('--league', type=int, default=39, help='League ID (default: Premier League)')
    parser.add_argument('--season', type=int, default=2022, help='Season year')
    parser.add_argument('--limit', type=int, help='Maximum matches to evaluate')
    parser.add_argument('--output', type=str, default='comparison_results', help='Output directory')
    
    args = parser.parse_args()
    
    evaluator = EnhancedSystemEvaluator(output_dir=args.output)
    results = evaluator.run_evaluation(
        league_id=args.league,
        season=args.season,
        limit=args.limit
    )
    
    # Print summary of improvements
    if results:
        draw_improvement = results.get('draw_improvement', {})
        original_acc = draw_improvement.get('original_accuracy', 0)
        enhanced_acc = draw_improvement.get('enhanced_accuracy', 0)
        
        print("\n=== EVALUATION SUMMARY ===")
        print(f"Original model overall accuracy: {results['original_metrics']['overall_accuracy']:.2f}%")
        print(f"Enhanced model overall accuracy: {results['enhanced_metrics']['overall_accuracy']:.2f}%")
        print(f"\nDraw prediction accuracy:")
        print(f"  - Original model: {original_acc:.2f}%")
        print(f"  - Enhanced model: {enhanced_acc:.2f}%")
        print(f"  - Improvement: {enhanced_acc - original_acc:.2f}%")
        print(f"\nResults saved to: {args.output}")

if __name__ == "__main__":
    main()