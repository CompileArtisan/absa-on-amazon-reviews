import json
import gzip
import pandas as pd
from datetime import datetime

def load_amazon_reviews(file_path, max_rows=None):
    """
    Load Amazon reviews from .jsonl.gz file
    """
    reviews = []
    
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_rows and i >= max_rows:
                break
            
            try:
                review = json.loads(line.strip())
                reviews.append(review)
            except json.JSONDecodeError:
                print(f"Error parsing line {i}")
                continue
            
            if (i + 1) % 100000 == 0:
                print(f"Loaded {i + 1} reviews...")
    
    return pd.DataFrame(reviews)

# Load the data
print("Loading All_Beauty reviews...")
df = load_amazon_reviews('All_Beauty.jsonl.gz')

# Save as CSV for easier access later
df.to_csv('all_beauty_reviews.csv', index=False)

print(f"\n✓ Loaded {len(df)} reviews")
print(f"✓ Saved to all_beauty_reviews.csv")
print(f"\nDataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
