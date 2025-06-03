"""
Match Winner Prediction Demo with Real Data

This script demonstrates how to use the match winner prediction model with real data from 
previous matches, evaluating its performance using historical match results.
"""

import logging
import pandas as pd
import json
from typing import Optional, Dict, Any
import os
import argparse
from datetime import datetime

from real_data_processor import RealDataProcessor, run_real_data_evaluation
from match_winner import predict_match_winner, MatchOutcome

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("match_winner_demo.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def save_evaluation_results(evaluation: Dict[str, Any], output_path: Optional[str] = None) -> str:
    """
    Save evaluation results to a JSON file.
    
    Args:
        evaluation: Evaluation results dictionary
        output_path: Optional output path, if None, generates a default path
        
    Returns:
        Path to the saved file
    """
    if output_path is None:
        # Generate default path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"match_winner_evaluation_{timestamp}.json"
    
    # Convert datetime objects to strings for JSON serialization
    def convert_datetime(obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        return obj
    
    try:
        # Process predictions to make them JSON serializable
        processed_evaluation = evaluation.copy()
        if 'predictions' in processed_evaluation:
            for pred in processed_evaluation['predictions']:
                if 'date' in pred and pred['date'] is not None:
                    pred['date'] = convert_datetime(pred['date'])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(processed_evaluation, f, indent=2, default=convert_datetime)
            
        logger.info(f"Evaluation results saved to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
        return ""

def run_demo_for_league(league_id: int, season: int, limit: int = 50) -> None:
    """
    Run a demonstration for a specific league and season.
    
    Args:
        league_id: League ID to evaluate
        season: Season year to evaluate
        limit: Number of matches to evaluate
    """
    logger.info(f"Running match winner prediction demo for league {league_id}, season {season}")
    
    # Get league name for better output
    league_names = {
        39: "Premier League",
        140: "La Liga",
        78: "Bundesliga",
        135: "Serie A",
        61: "Ligue 1"
    }
    league_name = league_names.get(league_id, f"League {league_id}")
    
    # Run evaluation
    evaluation = run_real_data_evaluation(league_id, season, limit)
    
    if 'error' in evaluation:
        logger.error(f"Evaluation failed: {evaluation['error']}")
        return
    
    # Print summary statistics
    print(f"\n{'=' * 50}")
    print(f"MATCH WINNER PREDICTION RESULTS: {league_name} {season}")
    print(f"{'=' * 50}")
    print(f"Total matches evaluated: {evaluation.get('total_matches', 0)}")
    print(f"Overall accuracy: {evaluation.get('overall_accuracy', 0)}%")
    
    # Print outcome-specific statistics
    print("\nAccuracy by outcome type:")
    for outcome, stats in evaluation.get('outcome_statistics', {}).items():
        print(f"  {outcome.upper()}: {stats.get('accuracy', 0)}% ({stats.get('correct', 0)}/{stats.get('total', 0)})")
    
    # Save detailed results
    output_path = save_evaluation_results(evaluation, f"match_winner_{league_id}_{season}.json")
    if output_path:
        print(f"\nDetailed results saved to: {output_path}")
    
    print(f"{'=' * 50}\n")

def main():
    """Main entry point for the demo script."""
    parser = argparse.ArgumentParser(description="Match Winner Prediction Demo")
    parser.add_argument("--league", type=int, default=39, help="League ID (default: 39 - Premier League)")
    parser.add_argument("--season", type=int, default=2022, help="Season year (default: 2022)")
    parser.add_argument("--limit", type=int, default=50, help="Number of matches to evaluate (default: 50)")
    
    args = parser.parse_args()
    
    run_demo_for_league(args.league, args.season, args.limit)

if __name__ == "__main__":
    main()
