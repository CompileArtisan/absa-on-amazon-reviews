import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('all_beauty_predictions_balanced.csv')

# Analyze products with most reviews
product_analysis = df.groupby('parent_asin').agg({
    'rating': ['count', 'mean'],
    'predicted_sentiment': lambda x: (x == 'positive').sum() / len(x) * 100
}).round(2)

product_analysis.columns = ['Review_Count', 'Avg_Rating', 'Positive_%']
product_analysis = product_analysis[product_analysis['Review_Count'] >= 50]  # At least 50 reviews
product_analysis = product_analysis.sort_values('Review_Count', ascending=False)

print("Top 20 Most Reviewed Products:")
print(product_analysis.head(20))

product_analysis.head(20).to_csv('top_products_analysis.csv')
print("\nSaved: top_products_analysis.csv")
