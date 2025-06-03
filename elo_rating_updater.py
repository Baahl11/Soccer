"""
Utility module for updating Elo ratings after soccer matches.

This module provides functions to batch update Elo ratings based on match results,
as well as track rating history for visualizations.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
from team_elo_rating import TeamEloRating, EloRatingTracker

# Set up logging
logger = logging.getLogger(__name__)

def update_elo_ratings_from_results(
    results_df: pd.DataFrame,
    elo_system: Optional[TeamEloRating] = None,
    save_history: bool = True
) -> TeamEloRating:
    """
    Update Elo ratings based on match results
    
    Args:
        results_df: DataFrame with match results, must contain:
                   - home_team_id, away_team_id
                   - home_goals, away_goals
                   - league_id (optional)
                   - match_date (optional)
        elo_system: Existing Elo system to update, creates new if None
        save_history: Whether to save history for visualization
        
    Returns:
        Updated Elo rating system
    """
    try:
        # Create or use existing Elo system
        if elo_system is None:
            elo_system = TeamEloRating()
        
        # Create tracker for history if needed
        tracker = None
        if save_history:
            tracker = EloRatingTracker()
        
        # Make sure we have the required columns
        required_cols = ['home_team_id', 'away_team_id', 'home_goals', 'away_goals']
        if not all(col in results_df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in results_df.columns]
            raise ValueError(f"Missing required columns in results data: {missing}")
        
        # Sort by date if available
        if 'match_date' in results_df.columns:
            results_df = results_df.sort_values('match_date')
        
        # Process each match
        for _, match in results_df.iterrows():
            try:
                # Get match data
                home_id = int(match['home_team_id'])
                away_id = int(match['away_team_id'])
                home_goals = int(match['home_goals'])
                away_goals = int(match['away_goals'])
                
                # Get optional fields
                league_id = int(match['league_id']) if 'league_id' in match else None
                match_date = match['match_date'] if 'match_date' in match else None
                
                # Calculate match importance based on league or competition
                match_importance = 1.0
                if 'competition_type' in match:
                    if match['competition_type'] == 'cup':
                        match_importance = 1.25  # Cup matches have higher importance
                    elif match['competition_type'] == 'international':
                        match_importance = 1.5   # International matches have even higher importance
                
                # Update ratings
                new_home_rating, new_away_rating = elo_system.update_ratings(
                    home_id=home_id,
                    away_id=away_id,
                    home_goals=home_goals,
                    away_goals=away_goals,
                    match_importance=match_importance,
                    league_id=league_id
                )
                
                # Save point-in-time snapshot if we have a date
                if save_history and tracker and match_date:
                    # Format date string if it's not already
                    if not isinstance(match_date, str):
                        match_date = pd.to_datetime(match_date).strftime("%Y-%m-%d")
                    
                    # Update history
                    tracker.update_history(elo_system.ratings.copy(), match_date)
            
            except Exception as e:
                logger.warning(f"Error processing match: {e}")
                continue
        
        # Save final ratings and history
        elo_system.save_ratings()
        
        if save_history and tracker:
            # Save one final snapshot with current date
            tracker.update_history(elo_system.ratings.copy())
            tracker.save_history()
        
        return elo_system
    
    except Exception as e:
        logger.error(f"Error updating Elo ratings: {e}")
        # Return existing or new system
        return elo_system or TeamEloRating()

def create_team_rating_report(
    team_ids: List[int], 
    team_names: Dict[int, str],
    output_dir: str = "reports",
    top_n: int = 20
) -> str:
    """
    Create a report of current Elo ratings
    
    Args:
        team_ids: List of team IDs to include in the report
        team_names: Dictionary mapping team IDs to names
        output_dir: Directory to save the report
        top_n: Number of top teams to include
        
    Returns:
        Path to generated report
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load Elo system
        elo_system = TeamEloRating()
        
        # Get ratings for all teams
        all_ratings = [(team_id, elo_system.get_rating(team_id)) for team_id in team_ids]
        
        # Sort by rating (descending)
        all_ratings.sort(key=lambda x: x[1], reverse=True)
        
        # Create tracker for history
        tracker = EloRatingTracker()
        
        # Generate report name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"elo_ratings_report_{timestamp}.txt")
        
        # Create visualizations for top teams
        top_team_ids = [team_id for team_id, _ in all_ratings[:top_n]]
        vis_file = os.path.join(output_dir, f"top_teams_elo_{timestamp}.png")
        tracker.plot_team_comparisons(team_names, top_team_ids, vis_file)
        
        # Write report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"ELO RATING REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write top teams
            f.write(f"TOP {top_n} TEAMS BY ELO RATING\n")
            f.write("-" * 40 + "\n")
            for rank, (team_id, rating) in enumerate(all_ratings[:top_n], 1):
                team_name = team_names.get(team_id, f"Team {team_id}")
                f.write(f"{rank:2d}. {team_name:<25} {rating:.1f}\n")
            
            f.write("\n")
            
            # Write statistics
            mean_rating = sum(r for _, r in all_ratings) / len(all_ratings) if all_ratings else 0
            f.write(f"Average Rating: {mean_rating:.1f}\n")
            f.write(f"Total Teams: {len(all_ratings)}\n")
            
            # Add reference to visualization
            if os.path.exists(vis_file):
                f.write(f"\nVisualization saved to: {vis_file}\n")
        
        return report_file
    
    except Exception as e:
        logger.error(f"Error creating Elo rating report: {e}")
        return ""

if __name__ == "__main__":
    # Set up logger for standalone execution
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Example usage
    logger.info("This module is designed to be imported, but can be run for testing.")
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'home_team_id': [1, 2, 3, 1],
        'away_team_id': [2, 3, 1, 3],
        'home_goals': [2, 1, 0, 3],
        'away_goals': [1, 1, 2, 0],
        'match_date': pd.date_range(start='2025-01-01', periods=4)
    })
    
    # Update ratings
    elo = update_elo_ratings_from_results(sample_data)
    
    # Print sample ratings
    for team_id in [1, 2, 3]:
        logger.info(f"Team {team_id} rating: {elo.get_rating(team_id):.1f}")
