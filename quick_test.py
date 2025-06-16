from odds_based_fixture_discovery import get_matches_with_odds_24h

matches = get_matches_with_odds_24h(3)
print(f"Found {len(matches)} matches:")
for i, match in enumerate(matches, 1):
    print(f"{i}. {match.get('home_team', 'N/A')} vs {match.get('away_team', 'N/A')}")
    print(f"   Method: {match.get('discovery_method', 'Unknown')}")
    print(f"   Team IDs: {match.get('home_team_id', 'N/A')} vs {match.get('away_team_id', 'N/A')}")
