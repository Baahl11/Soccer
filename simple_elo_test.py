from auto_updating_elo import get_elo_data_with_auto_rating

# Test with Bosnian teams
elo_data = get_elo_data_with_auto_rating(3361, 3382, 392)
print(f"Home ELO: {elo_data[\"home_elo\"]}")
print(f"Away ELO: {elo_data[\"away_elo\"]}")
print(f"ELO difference: {elo_data[\"elo_diff\"]}")
print(f"Win probability: {elo_data[\"elo_win_probability\"]}")
print(f"Expected goal difference: {elo_data[\"elo_expected_goal_diff\"]}")

