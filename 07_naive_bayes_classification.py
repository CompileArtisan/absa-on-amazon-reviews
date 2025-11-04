import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.sparse import hstack

# Load data with topics
print("Loading data with topics...")
df = pd.read_csv('all_beauty_with_topics.csv')

print(f"Dataset size: {len(df)}")
print(f"\nSentiment distribution:")
print(df['sentiment'].value_counts())

# Prepare features
print("\nCreating TF-IDF features...")
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    min_df=5,
    max_df=0.7,
    ngram_range=(1, 2),  # Unigrams and bigrams
    sublinear_tf=True
)

X_tfidf = tfidf_vectorizer.fit_transform(df['processed_text'])

# Get topic probability features
topic_prob_cols = [col for col in df.columns if col.startswith('topic_') and col.endswith('_prob')]
X_topics = df[topic_prob_cols].fillna(0).values

# Combine features
print("Combining TF-IDF and topic features...")
X_combined = hstack([X_tfidf, X_topics])

print(f"Feature shape: {X_combined.shape}")

# Target variable
y = df['sentiment']

# Train-test split
print("\nSplitting data...")
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X_combined, y, df.index,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train models
print("\n" + "="*70)
print("TRAINING NAIVE BAYES MODELS")
print("="*70)

# 1. Multinomial Naive Bayes
print("\n1. Training Multinomial Naive Bayes...")
mnb = MultinomialNB(alpha=0.1)
mnb.fit(X_train, y_train)
y_pred_mnb = mnb.predict(X_test)
acc_mnb = accuracy_score(y_test, y_pred_mnb)
print(f"   Accuracy: {acc_mnb:.4f}")

# 2. Complement Naive Bayes
print("\n2. Training Complement Naive Bayes...")
cnb = ComplementNB(alpha=0.1)
cnb.fit(X_train, y_train)
y_pred_cnb = cnb.predict(X_test)
acc_cnb = accuracy_score(y_test, y_pred_cnb)
print(f"   Accuracy: {acc_cnb:.4f}")

# Choose best model
if acc_cnb > acc_mnb:
    best_model = cnb
    y_pred = y_pred_cnb
    model_name = "Complement Naive Bayes"
    print(f"\n>>> Using Complement Naive Bayes (better performance)")
else:
    best_model = mnb
    y_pred = y_pred_mnb
    model_name = "Multinomial Naive Bayes"
    print(f"\n>>> Using Multinomial Naive Bayes (better performance)")

# Hyperparameter tuning
print("\n" + "="*70)
print("HYPERPARAMETER TUNING")
print("="*70)

param_grid = {'alpha': [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]}

print("\nPerforming grid search...")
grid_search = GridSearchCV(
    ComplementNB(),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV score: {grid_search.best_score_:.4f}")

# Use tuned model
tuned_model = grid_search.best_estimator_
y_pred_tuned = tuned_model.predict(X_test)

# Evaluation
print("\n" + "="*70)
print(f"EVALUATION: {model_name} (Tuned)")
print("="*70)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_tuned, digits=4))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_tuned, labels=['negative', 'neutral', 'positive'])

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'],
            cbar_kws={'label': 'Count'})
plt.title(f'{model_name} - Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')

print("\n✓ Confusion matrix saved to confusion_matrix.png")

# Save models
joblib.dump(tuned_model, 'naive_bayes_model.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("\n✓ Models saved:")
print("  - naive_bayes_model.pkl")
print("  - tfidf_vectorizer.pkl")

# Save predictions
df_test = df.loc[idx_test].copy()
df_test['predicted_sentiment'] = y_pred_tuned
df_test.to_csv('all_beauty_predictions.csv', index=False)

print("✓ Predictions saved to all_beauty_predictions.csv")
