import pandas as pd

df = pd.read_csv('all_beauty_predictions.csv')

# Analyze users with most reviews
user_analysis = df.groupby('user_id').agg({
    'rating': ['count', 'mean'],
    'predicted_sentiment': lambda x: x.mode()[0] if len(x) > 0 else None
})

user_analysis.columns = ['Review_Count', 'Avg_Rating', 'Dominant_Sentiment']
user_analysis = user_analysis[user_analysis['Review_Count'] >= 10]  # Power users

print("Power Users Analysis:")
print(user_analysis.describe())

# Sentiment distribution of power users
print("\nPower Users Sentiment Distribution:")
print(user_analysis['Dominant_Sentiment'].value_counts())
