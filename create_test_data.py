import numpy as np
import pandas as pd
import os

# Create synthetic data
np.random.seed(42)
n_samples = 10000

# Generate features
data = {
    'home_team_id': np.random.randint(1, 100, n_samples),
    'away_team_id': np.random.randint(1, 100, n_samples),
    'home_formation': np.random.choice(['4-3-3', '4-4-2', '3-5-2', '4-2-3-1'], n_samples),
    'away_formation': np.random.choice(['4-3-3', '4-4-2', '3-5-2', '4-2-3-1'], n_samples),
    'total_corners': np.random.normal(10, 3, n_samples).astype(int),
    'date': pd.date_range(start='2023-01-01', periods=n_samples).strftime('%Y-%m-%d'),
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure data directory exists
os.makedirs('data', exist_ok=True)

# Save to CSV
df.to_csv('data/corner_training_data.csv', index=False)
print(f"Created synthetic dataset with {n_samples} samples")
