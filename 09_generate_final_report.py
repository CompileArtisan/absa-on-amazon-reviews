import pandas as pd
import json
import pickle
from datetime import datetime

print("="*70)
print("GENERATING FINAL COMPREHENSIVE REPORT")
print("="*70)

# Load all necessary data
df = pd.read_csv('all_beauty_predictions.csv')
aspect_summary = pd.read_csv('aspect_summary.csv', index_col=0)

with open('topic_labels.pkl', 'rb') as f:
    topic_labels = pickle.load(f)

with open('coherence_results.pkl', 'rb') as f:
    coherence_results = pickle.load(f)

# Load classification metrics from previous run
from sklearn.metrics import classification_report, accuracy_score
import joblib

model = joblib.load('naive_bayes_model.pkl')

# Read test data
df_full = pd.read_csv('all_beauty_with_topics.csv')
X_test_indices = df.index

y_test = df_full.loc[X_test_indices, 'sentiment']
y_pred = df['predicted_sentiment']

accuracy = accuracy_score(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

# Create comprehensive JSON report
report = {
    'metadata': {
        'report_generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset': 'Amazon Reviews 2023 - All Beauty',
        'analysis_type': 'Aspect-Based Sentiment Analysis (LDA + Naive Bayes)'
    },
    'dataset_statistics': {
        'total_reviews': int(len(df_full)),
        'processed_reviews': int(len(df)),
        'date_range': {
            'start': pd.to_datetime(df_full['timestamp'], unit='ms').min().strftime('%Y-%m-%d'),
            'end': pd.to_datetime(df_full['timestamp'], unit='ms').max().strftime('%Y-%m-%d')
        },
        'unique_users': int(df_full['user_id'].nunique()),
        'unique_products': int(df_full['parent_asin'].nunique()),
        'rating_distribution': df_full['rating'].value_counts().sort_index().to_dict(),
        'sentiment_distribution': df['predicted_sentiment'].value_counts().to_dict()
    },
    'topic_modeling': {
        'method': 'Latent Dirichlet Allocation (LDA)',
        'num_topics': int(coherence_results['optimal_topics']),
        'coherence_score': float(coherence_results['coherence_scores'][
            coherence_results['num_topics'].index(coherence_results['optimal_topics'])
        ]),
        'topics': {str(k): v for k, v in topic_labels.items()}
    },
    'classification': {
        'model': 'Complement Naive Bayes',
        'features': 'TF-IDF (5000 features) + Topic Probabilities',
        'accuracy': float(accuracy),
        'precision': {
            'negative': float(class_report['negative']['precision']),
            'neutral': float(class_report['neutral']['precision']),
            'positive': float(class_report['positive']['precision']),
            'weighted_avg': float(class_report['weighted avg']['precision'])
        },
        'recall': {
            'negative': float(class_report['negative']['recall']),
            'neutral': float(class_report['neutral']['recall']),
            'positive': float(class_report['positive']['recall']),
            'weighted_avg': float(class_report['weighted avg']['recall'])
        },
        'f1_score': {
            'negative': float(class_report['negative']['f1-score']),
            'neutral': float(class_report['neutral']['f1-score']),
            'positive': float(class_report['positive']['f1-score']),
            'weighted_avg': float(class_report['weighted avg']['f1-score'])
        }
    },
    'aspect_based_insights': {
        'overall_summary': aspect_summary.to_dict('index'),
        'top_3_aspects': aspect_summary.head(3).index.tolist(),
        'bottom_3_aspects': aspect_summary.tail(3).index.tolist()
    }
}

# Save JSON report
with open('absa_final_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n✓ JSON report saved: absa_final_report.json")

# Create Excel report with multiple sheets
print("\nCreating Excel report...")
with pd.ExcelWriter('absa_final_report.xlsx', engine='openpyxl') as writer:
    
    # Sheet 1: Executive Summary
    exec_summary = pd.DataFrame([
        ['Dataset', 'Amazon Reviews 2023 - All Beauty'],
        ['Total Reviews', len(df_full)],
        ['Processed Reviews', len(df)],
        ['Date Range', f"{report['dataset_statistics']['date_range']['start']} to {report['dataset_statistics']['date_range']['end']}"],
        ['', ''],
        ['Topic Modeling Method', 'LDA'],
        ['Number of Topics', coherence_results['optimal_topics']],
        ['Coherence Score', f"{report['topic_modeling']['coherence_score']:.4f}"],
        ['', ''],
        ['Classification Model', 'Complement Naive Bayes'],
        ['Overall Accuracy', f"{accuracy:.4f}"],
        ['Weighted F1-Score', f"{class_report['weighted avg']['f1-score']:.4f}"],
    ], columns=['Metric', 'Value'])
    exec_summary.to_excel(writer, sheet_name='Executive Summary', index=False)
    
    # Sheet 2: Topics
    topics_df = pd.DataFrame([
        {'Topic_ID': k, 'Topic_Label': v} 
        for k, v in topic_labels.items()
    ])
    topics_df.to_excel(writer, sheet_name='Topics', index=False)
    
    # Sheet 3: Aspect Summary
    aspect_summary.to_excel(writer, sheet_name='Aspect Summary')
    
    # Sheet 4: Classification Metrics
    metrics_df = pd.DataFrame(class_report).T
    metrics_df.to_excel(writer, sheet_name='Classification Metrics')
    
    # Sheet 5: Sentiment by Aspect
    sentiment_by_aspect = pd.crosstab(
        df['topic_label'],
        df['predicted_sentiment']
    )
    sentiment_by_aspect.to_excel(writer, sheet_name='Sentiment by Aspect')
    
    # Sheet 6: Sample Predictions
    sample_predictions = df[[
        'text', 'rating', 'topic_label', 'sentiment', 
        'predicted_sentiment', 'dominant_prob'
    ]].head(200)
    sample_predictions.to_excel(writer, sheet_name='Sample Predictions', index=False)
    
    # Sheet 7: Misclassifications
    misclassified = df[df['sentiment'] != df['predicted_sentiment']][[
        'text', 'rating', 'sentiment', 'predicted_sentiment', 'topic_label'
    ]].head(100)
    misclassified.to_excel(writer, sheet_name='Misclassifications', index=False)

print("✓ Excel report saved: absa_final_report.xlsx")

# Create a text summary report
print("\nCreating text summary...")
with open('absa_summary.txt', 'w', encoding='utf-8') as f:
    f.write("="*70 + "\n")
    f.write("ASPECT-BASED SENTIMENT ANALYSIS - FINAL REPORT\n")
    f.write("Amazon Reviews 2023 - All Beauty Category\n")
    f.write("="*70 + "\n\n")
    
    f.write("1. DATASET OVERVIEW\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total Reviews: {len(df_full):,}\n")
    f.write(f"Date Range: {report['dataset_statistics']['date_range']['start']} to {report['dataset_statistics']['date_range']['end']}\n")
    f.write(f"Unique Users: {report['dataset_statistics']['unique_users']:,}\n")
    f.write(f"Unique Products: {report['dataset_statistics']['unique_products']:,}\n\n")
    
    f.write("2. TOPIC MODELING RESULTS\n")
    f.write("-" * 70 + "\n")
    f.write(f"Method: Latent Dirichlet Allocation (LDA)\n")
    f.write(f"Number of Topics: {coherence_results['optimal_topics']}\n")
    f.write(f"Coherence Score: {report['topic_modeling']['coherence_score']:.4f}\n\n")
    f.write("Discovered Topics:\n")
    for topic_id, label in topic_labels.items():
        f.write(f"  {topic_id}. {label}\n")
    f.write("\n")
    
    f.write("3. CLASSIFICATION PERFORMANCE\n")
    f.write("-" * 70 + "\n")
    f.write(f"Model: Complement Naive Bayes\n")
    f.write(f"Overall Accuracy: {accuracy:.4f}\n")
    f.write(f"Weighted F1-Score: {class_report['weighted avg']['f1-score']:.4f}\n\n")
    f.write("Per-Class Performance:\n")
    for sentiment in ['negative', 'neutral', 'positive']:
        f.write(f"\n  {sentiment.capitalize()}:\n")
        f.write(f"    Precision: {class_report[sentiment]['precision']:.4f}\n")
        f.write(f"    Recall:    {class_report[sentiment]['recall']:.4f}\n")
        f.write(f"    F1-Score:  {class_report[sentiment]['f1-score']:.4f}\n")
    f.write("\n")
    
    f.write("4. ASPECT-BASED INSIGHTS\n")
    f.write("-" * 70 + "\n")
    f.write("\nTop 3 Aspects (Highest Positive Sentiment):\n")
    for i, aspect in enumerate(aspect_summary.head(3).index, 1):
        pos_pct = aspect_summary.loc[aspect, 'Positive_%']
        neg_pct = aspect_summary.loc[aspect, 'Negative_%']
        net = aspect_summary.loc[aspect, 'Net_Sentiment']
        f.write(f"  {i}. {aspect}\n")
        f.write(f"     Positive: {pos_pct:.1f}% | Negative: {neg_pct:.1f}% | Net: +{net:.1f}%\n")
    
    f.write("\nBottom 3 Aspects (Areas for Improvement):\n")
    for i, aspect in enumerate(aspect_summary.tail(3).index, 1):
        pos_pct = aspect_summary.loc[aspect, 'Positive_%']
        neg_pct = aspect_summary.loc[aspect, 'Negative_%']
        net = aspect_summary.loc[aspect, 'Net_Sentiment']
        f.write(f"  {i}. {aspect}\n")
        f.write(f"     Positive: {pos_pct:.1f}% | Negative: {neg_pct:.1f}% | Net: {net:.1f}%\n")
    
    f.write("\n" + "="*70 + "\n")
    f.write("Report generated: " + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "\n")
    f.write("="*70 + "\n")

print("✓ Text summary saved: absa_summary.txt")

print("\n" + "="*70)
print("ALL REPORTS GENERATED SUCCESSFULLY!")
print("="*70)
print("\nGenerated files:")
print("  📄 absa_final_report.json")
print("  📊 absa_final_report.xlsx")
print("  📝 absa_summary.txt")
print("\n" + "="*70)
