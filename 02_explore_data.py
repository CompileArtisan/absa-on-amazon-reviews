import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load data
df = pd.read_csv('all_beauty_reviews.csv')

print("="*70)
print("DATASET OVERVIEW")
print("="*70)

# Basic stats
print(f"\nTotal reviews: {len(df):,}")
print(f"Date range: {pd.to_datetime(df['timestamp'], unit='ms').min()} to {pd.to_datetime(df['timestamp'], unit='ms').max()}")
print(f"Unique users: {df['user_id'].nunique():,}")
print(f"Unique products: {df['parent_asin'].nunique():,}")
print(f"Verified purchases: {df['verified_purchase'].sum():,} ({df['verified_purchase'].mean()*100:.1f}%)")

# Rating distribution
print("\n" + "="*70)
print("RATING DISTRIBUTION")
print("="*70)
print(df['rating'].value_counts().sort_index())
print("\nPercentages:")
print(df['rating'].value_counts(normalize=True).sort_index() * 100)

# Text length analysis
df['text_length'] = df['text'].fillna('').str.len()
df['word_count'] = df['text'].fillna('').str.split().str.len()

print("\n" + "="*70)
print("TEXT STATISTICS")
print("="*70)
print(f"Average text length: {df['text_length'].mean():.0f} characters")
print(f"Average word count: {df['word_count'].mean():.0f} words")
print(f"Median word count: {df['word_count'].median():.0f} words")

# Missing values
print("\n" + "="*70)
print("MISSING VALUES")
print("="*70)
print(df.isnull().sum())

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Rating distribution
df['rating'].value_counts().sort_index().plot(kind='bar', ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title('Rating Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Rating')
axes[0, 0].set_ylabel('Count')
axes[0, 0].grid(axis='y', alpha=0.3)

# Word count distribution
axes[0, 1].hist(df['word_count'], bins=50, color='coral', edgecolor='black')
axes[0, 1].set_title('Word Count Distribution', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Number of Words')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_xlim(0, 500)
axes[0, 1].grid(axis='y', alpha=0.3)

# Reviews over time
df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df['year_month'] = df['date'].dt.to_period('M')
reviews_over_time = df['year_month'].value_counts().sort_index()
reviews_over_time.plot(ax=axes[1, 0], color='green')
axes[1, 0].set_title('Reviews Over Time', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Date')
axes[1, 0].set_ylabel('Number of Reviews')
axes[1, 0].grid(alpha=0.3)

# Verified vs Unverified
verified_counts = df['verified_purchase'].value_counts()
axes[1, 1].pie(verified_counts, labels=['Verified', 'Unverified'], 
               autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
axes[1, 1].set_title('Verified vs Unverified Purchases', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('data_exploration.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved to data_exploration.png")

# Sample reviews
print("\n" + "="*70)
print("SAMPLE REVIEWS")
print("="*70)
for i in range(3):
    print(f"\nReview {i+1}:")
    print(f"Rating: {df.iloc[i]['rating']}/5")
    print(f"Title: {df.iloc[i]['title']}")
    print(f"Text: {df.iloc[i]['text'][:200]}...")
