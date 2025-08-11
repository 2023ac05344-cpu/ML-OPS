import pandas as pd
import os
from sklearn.datasets import fetch_california_housing

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Fetch the California Housing dataset
data = fetch_california_housing(as_frame=True)

# Convert to DataFrame
housing_df = data.frame

# Save to CSV in data directory
output_path = "data/california_housing.csv"
housing_df.to_csv(output_path, index=False)

print(f"California Housing dataset downloaded and saved as {output_path}")
print(f"Dataset shape: {housing_df.shape}")
print(f"Features: {list(housing_df.columns)}")
print(f"Target: median_house_value")
