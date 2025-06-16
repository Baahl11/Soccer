# Documented Prediction Data Fields

## Match Information:
- match_id (int)
- fixture_id (int)
- home_team (str)
- away_team (str)
- home_team_id (int)
- away_team_id (int)
- league (str)
- league_id (int)
- match_date / match_time (datetime)
- status (str)

## Prediction Probabilities (1X2):
- home_win_probability (float)
- draw_probability (float)
- away_win_probability (float)
- probabilities_1x2: {home_win, draw, away_win} (float rounded)
- confidence_score / confidence / overall_confidence (float)
- confidence_level (str: e.g., "medium")
- confidence_factors (dict: increasing, decreasing lists)
- confidence_distribution (dict: counts by confidence level)

## Goals Predictions:
- predicted_home_goals (float)
- predicted_away_goals (float)
- predicted_total_goals (float)
- total_goals (float)
- over_2_5_probability (float)
- under_2_5_probability (float)
- prob_over_2_5 (float)
- prob_btts (float) (both teams to score probability)

## Corners Predictions:
- predicted_home_corners (float)
- predicted_away_corners (float)
- predicted_total_corners (float)
- over_9_5_corners_probability (float)
- under_9_5_corners_probability (float)
- corners: {home, away, total, over_8.5, over_9.5, over_10.5} (float)
- corners confidence (str, e.g., "Alta")

## Cards Predictions:
- cards: {home, away, total, over_2.5, over_3.5, over_4.5} (float)

## Fouls Predictions:
- fouls: {home, away, total, over_19.5, over_21.5, over_23.5} (float)

## ELO Ratings and Probabilities:
- home_elo (float)
- away_elo (float)
- elo_diff (float)
- strength_comparison (str)
- elo_probabilities: {win, draw, loss} (float)
- win_probabilities: {home, draw, away} (float)
- expected_goal_diff (float)

## Tactical Analysis:
- tactical_style: {home: {possession_style, defensive_line}, away: {possession_style, defensive_line}}
- key_battles (list of str)
- strengths: {home: list of str, away: list of str}
- weaknesses: {home: list of str, away: list of str}
- tactical_recommendation (str)
- expected_formations: {home: str, away: str}
- tactical_analysis error messages (str)

## Odds Analysis:
- market_analysis: {efficiency (float), margin (float)}
- value_opportunities: list of {market (str), selection (str), fair_odds (float), market_odds (float), value (float), recommendation (str)}
- market_sentiment: {description (str), implied_probabilities: {home_win, draw, away_win} (float)}

## Value Bets:
- is_value_bet (bool)
- value_bet_type (str, e.g., "1x2", "goals", "corners")
- value_bet_selection (str, e.g., "home_win", "over_2.5")
- expected_value (float)
- value_bet_percentage (float)
- recommended_stake (float)
- value_bets_count (int)
- best_value_percentage (float)

## Model and Metadata:
- model_version (str)
- prediction_method (str)
- model_details (dict)
- additional_predictions (JSON string/dict)
- model_features_used (JSON string/dict)
- created_at (datetime)
- updated_at (datetime)
- id (int)

## Performance and Statistics:
- total_predictions (int)
- correct_predictions (int)
- accuracy_percentage (float)
- average_odds_beaten (float)
- profit_loss (float)
- best_bet_type (str)
- period (str)

## Fixture Statistics Analysis:
- fixture_statistics: {available (bool), confidence_boost (float), goal_modifiers: {home, away} (float), comparative_analysis: {}, note (str)}
- home_team_stats: {shots_per_game, shot_accuracy, possession_avg, disciplinary_risk} (float)
- away_team_stats: {shots_per_game, shot_accuracy, possession_avg, disciplinary_risk} (float)
- statistical_advantages: {shooting_advantage, possession_advantage, discipline_advantage} (str: 'home' or 'away')

## Enhanced Team Statistics:
- shots_per_game (float)
- shots_on_target_per_game (float)
- possession_percentage (float)
- fouls_per_game (float)
- goals_per_game (float)
- goals_conceded_per_game (float)
- passes_completed_per_game (float)
- passes_attempted_per_game (float)

## Dashboard and Summary:
- total_matches_today (int)
- predictions_generated (int)
- value_bets_found (int)
- average_confidence (float)
- hot_matches (list of hot match objects with match_id, confidence_score, value_bets_count, best_value_percentage, reasoning, urgency_score)
- recent_matches (list of match summaries with match_id, home_team, away_team, match_time, league, top_prediction, confidence, value_bets, status)
