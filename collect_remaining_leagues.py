"""
Script to collect corner and formation data from remaining leagues.
"""
import os
from corner_data_collector_v2 import FootballDataCollector, SERIE_A_ID, LIGUE_1_ID

# API credentials
api_key = "f554e37ff422c7e741bb1acfb35b898f"
api_base_url = "https://v3.football.api-sports.io"
api_host = "v3.football.api-sports.io"

def main():
    collector = FootballDataCollector(api_key, api_base_url, api_host)
    
    # Collect data for current season
    season = "2023"
    
    # Remaining leagues
    remaining_leagues = [
        (SERIE_A_ID, "Serie A"),
        (LIGUE_1_ID, "Ligue 1")
    ]
    
    try:
        all_data = collector.collect_multiple_leagues(season, remaining_leagues)
        print("Data collection completed successfully!")
        
        # Print summary
        for league_name, league_data in all_data.items():
            print(f"{league_name}: {len(league_data)} matches processed")
            
    except Exception as e:
        print(f"Error during data collection: {str(e)}")

if __name__ == "__main__":
    main()
