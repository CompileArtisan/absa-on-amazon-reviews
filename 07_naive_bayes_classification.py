import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.sparse import hstack
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# Load data with topics
print("Loading data with topics...")
df = pd.read_csv('all_beauty_with_topics.csv')

print(f"Dataset size: {len(df)}")
print(f"\nOriginal sentiment distribution:")
print(df['sentiment'].value_counts())
print("\nPercentages:")
print(df['sentiment'].value_counts(normalize=True) * 100)

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

print("\nTraining set distribution:")
print(pd.Series(y_train).value_counts())

# ============================================================================
# APPLY SMOTE + UNDERSAMPLING TO HANDLE CLASS IMBALANCE
# ============================================================================
print("\n" + "="*70)
print("APPLYING SMOTE + UNDERSAMPLING FOR CLASS BALANCE")
print("="*70)

# Strategy:
# 1. Over-sample neutral and negative to reasonable sizes
# 2. Under-sample positive to prevent overwhelming the dataset

# Target distribution (adjust these numbers based on your preference)
# We'll aim for more balanced classes while keeping dataset manageable
target_positive = 150000  # Reduce from ~294k
target_negative = 100000  # Increase from ~93k  
target_neutral = 80000     # Increase from ~37k

print(f"\nTarget distribution after resampling:")
print(f"  Positive: {target_positive:,}")
print(f"  Negative: {target_negative:,}")
print(f"  Neutral: {target_neutral:,}")
print(f"  Total: {target_positive + target_negative + target_neutral:,}")

# SMOTE for over-sampling minority classes
print("\nApplying SMOTE to minority classes...")
smote = SMOTE(
    sampling_strategy={
        'neutral': target_neutral,
        'negative': target_negative
    },
    random_state=42,
    k_neighbors=5
)

X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

print(f"After SMOTE:")
print(pd.Series(y_train_over).value_counts())

# Under-sample majority class
print("\nApplying undersampling to majority class...")
undersample = RandomUnderSampler(
    sampling_strategy={'positive': target_positive},
    random_state=42
)

X_train_balanced, y_train_balanced = undersample.fit_resample(X_train_over, y_train_over)

print(f"\nFinal balanced training set:")
print(pd.Series(y_train_balanced).value_counts())
print(f"Total training samples: {len(y_train_balanced):,}")

# ============================================================================
# TRAIN MODELS ON BALANCED DATA
# ============================================================================
print("\n" + "="*70)
print("TRAINING NAIVE BAYES MODELS (ON BALANCED DATA)")
print("="*70)

# Test both models
models = {
    'MultinomialNB': MultinomialNB(alpha=0.1),
    'ComplementNB': ComplementNB(alpha=0.1)
}

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        'model': model,
        'accuracy': acc,
        'f1': f1,
        'predictions': y_pred
    }
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Weighted F1: {f1:.4f}")

# Select best model based on F1 score
best_name = max(results, key=lambda k: results[k]['f1'])
best_model = results[best_name]['model']
BestModelClass = type(best_model)

print(f"\n>>> Best performing model: {best_name}")
print(f"    F1 Score: {results[best_name]['f1']:.4f}")

# ============================================================================
# HYPERPARAMETER TUNING ON BEST MODEL
# ============================================================================
print("\n" + "="*70)
print(f"HYPERPARAMETER TUNING: {best_name}")
print("="*70)

param_grid = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 2.0],
    'fit_prior': [True, False]
}

print("\nPerforming grid search with 5-fold CV...")
grid_search = GridSearchCV(
    BestModelClass(),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_balanced, y_train_balanced)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV F1 score: {grid_search.best_score_:.4f}")

# Use tuned model
tuned_model = grid_search.best_estimator_
y_pred_tuned = tuned_model.predict(X_test)

# ============================================================================
# DETAILED EVALUATION
# ============================================================================
print("\n" + "="*70)
print(f"EVALUATION: {best_name} (Tuned + Balanced Data)")
print("="*70)

# Overall metrics
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned, average='weighted')

print(f"\nOverall Performance:")
print(f"  Accuracy: {accuracy_tuned:.4f}")
print(f"  Weighted F1: {f1_tuned:.4f}")

# Classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred_tuned, digits=4))

# Comparison with baseline (before SMOTE)
print("\n" + "="*70)
print("COMPARISON: Before vs After SMOTE")
print("="*70)

# Train a baseline model without SMOTE for comparison
baseline_model = BestModelClass(alpha=0.1)
baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

print("\nBEFORE SMOTE (Imbalanced Data):")
print(classification_report(y_test, y_pred_baseline, digits=4))

print("\nAFTER SMOTE (Balanced Data):")
print(classification_report(y_test, y_pred_tuned, digits=4))

# Per-class improvement
print("\n" + "="*70)
print("PER-CLASS IMPROVEMENT")
print("="*70)

from sklearn.metrics import precision_recall_fscore_support

before_metrics = precision_recall_fscore_support(y_test, y_pred_baseline, 
                                                   labels=['negative', 'neutral', 'positive'])
after_metrics = precision_recall_fscore_support(y_test, y_pred_tuned,
                                                  labels=['negative', 'neutral', 'positive'])

improvement_df = pd.DataFrame({
    'Class': ['Negative', 'Neutral', 'Positive'],
    'F1_Before': before_metrics[2],
    'F1_After': after_metrics[2],
    'F1_Improvement': after_metrics[2] - before_metrics[2],
    'Recall_Before': before_metrics[1],
    'Recall_After': after_metrics[1],
    'Recall_Improvement': after_metrics[1] - before_metrics[1]
})

print("\n", improvement_df.to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# 1. Confusion Matrix Comparison
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Before SMOTE
cm_before = confusion_matrix(y_test, y_pred_baseline, 
                             labels=['negative', 'neutral', 'positive'])
sns.heatmap(cm_before, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'],
            cbar_kws={'label': 'Count'})
axes[0].set_title('Before SMOTE (Imbalanced)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# After SMOTE
cm_after = confusion_matrix(y_test, y_pred_tuned,
                           labels=['negative', 'neutral', 'positive'])
sns.heatmap(cm_after, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['Negative', 'Neutral', 'Positive'],
            yticklabels=['Negative', 'Neutral', 'Positive'],
            cbar_kws={'label': 'Count'})
axes[1].set_title('After SMOTE (Balanced)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

plt.tight_layout()
plt.savefig('confusion_matrix_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: confusion_matrix_comparison.png")

# 2. F1 Score Comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(['Negative', 'Neutral', 'Positive']))
width = 0.35

bars1 = ax.bar(x - width/2, before_metrics[2], width, label='Before SMOTE', color='lightcoral')
bars2 = ax.bar(x + width/2, after_metrics[2], width, label='After SMOTE', color='lightgreen')

ax.set_xlabel('Sentiment Class', fontsize=12)
ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('F1 Score Comparison: Before vs After SMOTE', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['Negative', 'Neutral', 'Positive'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('f1_score_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: f1_score_comparison.png")

# ============================================================================
# SAVE MODELS AND PREDICTIONS
# ============================================================================
print("\n" + "="*70)
print("SAVING MODELS AND RESULTS")
print("="*70)

# Save tuned model
joblib.dump(tuned_model, 'naive_bayes_model_balanced.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

print("\n✓ Models saved:")
print("  - naive_bayes_model_balanced.pkl")
print("  - tfidf_vectorizer.pkl")

# Save predictions
df_test = df.loc[idx_test].copy()
df_test['predicted_sentiment'] = y_pred_tuned
df_test.to_csv('all_beauty_predictions_balanced.csv', index=False)

print("✓ Predictions saved: all_beauty_predictions_balanced.csv")

# Save comparison report
with open('smote_improvement_report.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("SMOTE IMPROVEMENT REPORT\n")
    f.write("="*70 + "\n\n")
    
    f.write("ORIGINAL DATA DISTRIBUTION:\n")
    f.write(f"  Positive: {(y_train == 'positive').sum():,} ({(y_train == 'positive').mean()*100:.1f}%)\n")
    f.write(f"  Negative: {(y_train == 'negative').sum():,} ({(y_train == 'negative').mean()*100:.1f}%)\n")
    f.write(f"  Neutral: {(y_train == 'neutral').sum():,} ({(y_train == 'neutral').mean()*100:.1f}%)\n\n")
    
    f.write("BALANCED DATA DISTRIBUTION:\n")
    f.write(f"  Positive: {(y_train_balanced == 'positive').sum():,}\n")
    f.write(f"  Negative: {(y_train_balanced == 'negative').sum():,}\n")
    f.write(f"  Neutral: {(y_train_balanced == 'neutral').sum():,}\n\n")
    
    f.write("PERFORMANCE COMPARISON:\n")
    f.write(f"  Before SMOTE - Weighted F1: {f1_score(y_test, y_pred_baseline, average='weighted'):.4f}\n")
    f.write(f"  After SMOTE  - Weighted F1: {f1_tuned:.4f}\n\n")
    
    f.write("PER-CLASS F1 IMPROVEMENTS:\n")
    f.write(improvement_df.to_string(index=False))
    f.write("\n\n")
    
    f.write("BEFORE SMOTE:\n")
    f.write(classification_report(y_test, y_pred_baseline))
    f.write("\n\nAFTER SMOTE:\n")
    f.write(classification_report(y_test, y_pred_tuned))

print("✓ Report saved: smote_improvement_report.txt")

print("\n" + "="*70)
print("SMOTE BALANCING COMPLETE!")
print("="*70)
print(f"\n🎯 Key Improvements:")
print(f"   Neutral F1: {before_metrics[2][1]:.4f} → {after_metrics[2][1]:.4f} (+{after_metrics[2][1]-before_metrics[2][1]:.4f})")
print(f"   Neutral Recall: {before_metrics[1][1]:.4f} → {after_metrics[1][1]:.4f} (+{after_metrics[1][1]-before_metrics[1][1]:.4f})")
