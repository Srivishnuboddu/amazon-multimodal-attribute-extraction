import json
import pandas as pd

# Replace with your actual path to metadata
metadata_path = "data/raw/abo-listings/abo-listings.json"  

# Load JSON data
with open(metadata_path, "r") as f:
    data = json.load(f)  # or: [json.loads(line) for line in f] if JSON Lines

# Convert to DataFrame
df = pd.DataFrame(data)

# Keep only necessary columns
df_small = df[["title", "description", "category", "brand"]]

# Optional: pick one category for your first run
category_filter = "Clothing"  # adjust as needed
df_small = df_small[df_small["category"].str.contains(category_filter, na=False)]

# Save small sample for baseline
df_small.to_csv("data/processed/sample.csv", index=False)
print(f"Saved {len(df_small)} rows to data/processed/sample.csv")
