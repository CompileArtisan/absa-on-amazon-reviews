import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def clean_text(text):
    """Clean review text"""
    if pd.isna(text) or text == '':
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_text(text):
    """Tokenize, remove stopwords, lemmatize"""
    # Clean first
    text = clean_text(text)
    
    if not text:
        return []
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords (but keep negations for sentiment)
    stop_words = set(stopwords.words('english'))
    keep_words = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', 
                  'nowhere', 'hardly', 'barely', 'scarcely', "don't", "doesn't",
                  "didn't", "won't", "wouldn't", "shouldn't", "couldn't", "can't"}
    stop_words = stop_words - keep_words
    
    # Remove short words and stopwords
    tokens = [w for w in tokens if len(w) > 2 and w not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    
    return tokens

# Load data
print("Loading data...")
df = pd.read_csv('all_beauty_reviews.csv')

print(f"Original dataset: {len(df)} reviews")

# Filter out reviews with no text
df = df[df['text'].notna()].copy()
print(f"After removing null text: {len(df)} reviews")

# Apply preprocessing
print("\nCleaning text...")
df['cleaned_text'] = df['text'].apply(clean_text)

print("Tokenizing and lemmatizing...")
df['tokens'] = df['cleaned_text'].apply(preprocess_text)
df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))

# Filter out very short reviews (less than 5 words)
df['token_count'] = df['tokens'].apply(len)
df = df[df['token_count'] >= 5].copy()

print(f"After filtering short reviews: {len(df)} reviews")

# Create sentiment labels
def create_sentiment_label(rating):
    if rating <= 2:
        return 'negative'
    elif rating == 3:
        return 'neutral'
    else:
        return 'positive'

df['sentiment'] = df['rating'].apply(create_sentiment_label)

# Save preprocessed data
df.to_csv('all_beauty_preprocessed.csv', index=False)

print("\n" + "="*70)
print("PREPROCESSING COMPLETE")
print("="*70)
print(f"Final dataset: {len(df)} reviews")
print("\nSentiment distribution:")
print(df['sentiment'].value_counts())
print("\nPercentages:")
print(df['sentiment'].value_counts(normalize=True) * 100)

# Show examples
print("\n" + "="*70)
print("PREPROCESSING EXAMPLES")
print("="*70)
for i in range(2):
    print(f"\nExample {i+1}:")
    print(f"Original: {df.iloc[i]['text'][:150]}...")
    print(f"Cleaned: {df.iloc[i]['cleaned_text'][:150]}...")
    print(f"Tokens: {df.iloc[i]['tokens'][:20]}")

print("\n✓ Preprocessed data saved to all_beauty_preprocessed.csv")
