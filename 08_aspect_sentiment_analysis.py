import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud

# Load predictions
print("Loading predictions...")
df = pd.read_csv('all_beauty_predictions_balanced.csv')

print(f"Total reviews: {len(df)}")

# Remove unknown topics
df = df[df['topic_label'] != 'Unknown'].copy()

print("\n" + "="*70)
print("ASPECT-BASED SENTIMENT ANALYSIS")
print("="*70)

# 1. Overall sentiment by aspect
aspect_sentiment = pd.crosstab(
    df['topic_label'],
    df['predicted_sentiment'],
    normalize='index'
) * 100

print("\nSentiment Distribution by Aspect (%):")
print(aspect_sentiment.round(2))

# Visualization 1: Stacked bar chart
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Absolute counts
aspect_sentiment_counts = pd.crosstab(
    df['topic_label'],
    df['predicted_sentiment']
)

aspect_sentiment_counts.plot(kind='bar', stacked=False, ax=axes[0], 
                             color=['#d62728', '#ff7f0e', '#2ca02c'])
axes[0].set_title('Sentiment Distribution by Aspect (Counts)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Product Aspect', fontsize=12)
axes[0].set_ylabel('Number of Reviews', fontsize=12)
axes[0].legend(title='Sentiment', labels=['Negative', 'Neutral', 'Positive'])
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(axis='y', alpha=0.3)

# Percentages
aspect_sentiment.plot(kind='bar', stacked=False, ax=axes[1],
                      color=['#d62728', '#ff7f0e', '#2ca02c'])
axes[1].set_title('Sentiment Distribution by Aspect (%)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Product Aspect', fontsize=12)
axes[1].set_ylabel('Percentage', fontsize=12)
axes[1].legend(title='Sentiment', labels=['Negative', 'Neutral', 'Positive'])
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('aspect_sentiment_distribution.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: aspect_sentiment_distribution.png")

# Visualization 2: Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(aspect_sentiment_counts, annot=True, fmt='d', cmap='RdYlGn',
            cbar_kws={'label': 'Number of Reviews'})
plt.title('Aspect-Based Sentiment Analysis Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Sentiment', fontsize=12)
plt.ylabel('Product Aspect', fontsize=12)
plt.tight_layout()
plt.savefig('absa_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ Saved: absa_heatmap.png")

# 2. Identify problematic aspects
print("\n" + "="*70)
print("ASPECT ANALYSIS - STRENGTHS & WEAKNESSES")
print("="*70)

aspect_summary = pd.DataFrame({
    'Total_Reviews': df.groupby('topic_label').size(),
    'Positive_%': aspect_sentiment['positive'],
    'Neutral_%': aspect_sentiment['neutral'],
    'Negative_%': aspect_sentiment['negative'],
    'Net_Sentiment': aspect_sentiment['positive'] - aspect_sentiment['negative']
})

aspect_summary = aspect_summary.sort_values('Net_Sentiment', ascending=False)

print("\nAspect Summary (sorted by Net Sentiment):")
print(aspect_summary.round(2))

print("\n🟢 STRONGEST ASPECTS:")
for aspect in aspect_summary.head(3).index:
    pos_pct = aspect_summary.loc[aspect, 'Positive_%']
    neg_pct = aspect_summary.loc[aspect, 'Negative_%']
    print(f"  • {aspect}: {pos_pct:.1f}% positive, {neg_pct:.1f}% negative")

print("\n🔴 WEAKEST ASPECTS:")
for aspect in aspect_summary.tail(3).index:
    pos_pct = aspect_summary.loc[aspect, 'Positive_%']
    neg_pct = aspect_summary.loc[aspect, 'Negative_%']
    print(f"  • {aspect}: {pos_pct:.1f}% positive, {neg_pct:.1f}% negative")

# 3. Deep dive into each aspect
print("\n" + "="*70)
print("DETAILED ASPECT ANALYSIS")
print("="*70)

def analyze_aspect(df, aspect_name, n_examples=3):
    """Detailed analysis of a specific aspect"""
    print(f"\n{'='*70}")
    print(f"ASPECT: {aspect_name}")
    print(f"{'='*70}")
    
    aspect_df = df[df['topic_label'] == aspect_name].copy()
    
    print(f"\nTotal reviews: {len(aspect_df):,}")
    
    # Sentiment breakdown
    sentiment_counts = aspect_df['predicted_sentiment'].value_counts()
    print("\nSentiment Distribution:")
    for sentiment in ['positive', 'neutral', 'negative']:
        if sentiment in sentiment_counts.index:
            count = sentiment_counts[sentiment]
            pct = (count / len(aspect_df)) * 100
            print(f"  {sentiment.capitalize():<10}: {count:>6,} ({pct:>5.1f}%)")
    
    # Average rating
    avg_rating = aspect_df['rating'].mean()
    print(f"\nAverage Rating: {avg_rating:.2f}/5.0")
    
    # Show examples for each sentiment
    for sentiment in ['positive', 'negative', 'neutral']:
        examples = aspect_df[aspect_df['predicted_sentiment'] == sentiment].head(n_examples)
        
        if len(examples) > 0:
            print(f"\n{sentiment.upper()} EXAMPLES:")
            for idx, (_, row) in enumerate(examples.iterrows(), 1):
                print(f"\n  [{idx}] Rating: {row['rating']:.0f}/5 | Confidence: {row['dominant_prob']:.2f}")
                text = row['text'] if len(row['text']) <= 200 else row['text'][:200] + "..."
                print(f"      \"{text}\"")
    
    return aspect_df

# Analyze all aspects
aspect_analyses = {}
for aspect in aspect_summary.index:
    aspect_analyses[aspect] = analyze_aspect(df, aspect, n_examples=2)

# 4. Word clouds by aspect and sentiment
print("\n" + "="*70)
print("GENERATING WORD CLOUDS")
print("="*70)

def generate_aspect_wordclouds(df, aspect_name):
    """Generate word clouds for each sentiment within an aspect"""
    aspect_df = df[df['topic_label'] == aspect_name]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sentiments = ['positive', 'neutral', 'negative']
    colors = ['Greens', 'Blues', 'Reds']
    
    for ax, sentiment, colormap in zip(axes, sentiments, colors):
        text_data = aspect_df[aspect_df['predicted_sentiment'] == sentiment]['processed_text']
        
        if len(text_data) > 0:
            text = ' '.join(text_data.values)
            
            wordcloud = WordCloud(
                width=600, height=400,
                background_color='white',
                colormap=colormap,
                max_words=50,
                relative_scaling=0.5
            ).generate(text)
            
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.set_title(f'{sentiment.capitalize()} ({len(text_data)} reviews)', 
                        fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No reviews', ha='center', va='center')
            ax.set_title(f'{sentiment.capitalize()} (0 reviews)', 
                        fontsize=12, fontweight='bold')
        
        ax.axis('off')
    
    fig.suptitle(f'Word Clouds: {aspect_name}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    filename = f'wordcloud_{aspect_name.replace(" ", "_").replace("&", "and").lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

# Generate word clouds for top aspects
print("\nGenerating word clouds for key aspects...")
for aspect in aspect_summary.head(3).index:
    generate_aspect_wordclouds(df, aspect)

for aspect in aspect_summary.tail(2).index:
    generate_aspect_wordclouds(df, aspect)

# 5. Sentiment trends over time
print("\n" + "="*70)
print("TEMPORAL ANALYSIS")
print("="*70)

df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
df['year_month'] = df['date'].dt.to_period('M')

# Overall sentiment trend
print("\nAnalyzing sentiment trends over time...")
sentiment_over_time = df.groupby(['year_month', 'predicted_sentiment']).size().unstack(fill_value=0)

plt.figure(figsize=(14, 6))
sentiment_over_time.plot(kind='line', marker='o', ax=plt.gca(),
                         color=['#d62728', '#ff7f0e', '#2ca02c'])
plt.title('Overall Sentiment Trends Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Number of Reviews', fontsize=12)
plt.legend(title='Sentiment', labels=['Negative', 'Neutral', 'Positive'])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('sentiment_trend_overall.png', dpi=300, bbox_inches='tight')
print("✓ Saved: sentiment_trend_overall.png")

# Trend for specific aspects
for aspect in aspect_summary.head(2).index:
    aspect_df = df[df['topic_label'] == aspect]
    trend = aspect_df.groupby(['year_month', 'predicted_sentiment']).size().unstack(fill_value=0)
    
    plt.figure(figsize=(14, 6))
    trend.plot(kind='line', marker='o', ax=plt.gca(),
               color=['#d62728', '#ff7f0e', '#2ca02c'])
    plt.title(f'Sentiment Trend: {aspect}', fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Number of Reviews', fontsize=12)
    plt.legend(title='Sentiment', labels=['Negative', 'Neutral', 'Positive'])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f'sentiment_trend_{aspect.replace(" ", "_").replace("&", "and").lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {filename}")
    plt.close()

# Save detailed results
aspect_summary.to_csv('aspect_summary.csv')
print("\n✓ Saved: aspect_summary.csv")

print("\n" + "="*70)
print("ASPECT-BASED SENTIMENT ANALYSIS COMPLETE!")
print("="*70)
