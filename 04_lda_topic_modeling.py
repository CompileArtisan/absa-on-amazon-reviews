import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
import matplotlib.pyplot as plt
import pickle

# Load preprocessed data
print("Loading preprocessed data...")
df = pd.read_csv('all_beauty_preprocessed.csv')
df['tokens'] = df['tokens'].apply(eval)  # Convert string back to list

print(f"Dataset size: {len(df)} reviews")

SAMPLE_SIZE = 100000  
if len(df) > SAMPLE_SIZE:
    print(f"\nSampling {SAMPLE_SIZE} reviews for coherence testing...")
    df_sample = df.sample(n=SAMPLE_SIZE, random_state=42)
else:
    df_sample = df.copy()

# Create dictionary and corpus
print("\nCreating dictionary and corpus...")
dictionary = corpora.Dictionary(df_sample['tokens'])

print(f"Dictionary size before filtering: {len(dictionary)}")

# Filter extremes
dictionary.filter_extremes(
    no_below=10,     
    no_above=0.5,   
    keep_n=5000    
)

print(f"Dictionary size after filtering: {len(dictionary)}")

# Create corpus
corpus = [dictionary.doc2bow(tokens) for tokens in df_sample['tokens']]

# Save dictionary and corpus
dictionary.save('beauty_dictionary.dict')
with open('beauty_corpus.pkl', 'wb') as f:
    pickle.dump(corpus, f)

print("\n" + "="*70)
print("FINDING OPTIMAL NUMBER OF TOPICS")
print("="*70)

def compute_coherence_values(dictionary, corpus, texts, limit=20, start=5, step=1):
    """
    Compute coherence scores for different numbers of topics
    """
    coherence_values = []
    model_list = []
    
    for num_topics in range(start, limit, step):
        print(f"\nTraining LDA with {num_topics} topics...")
        
        model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            chunksize=2000,
            passes=10,
            alpha='auto',
            eta='auto',
            per_word_topics=True,
            eval_every=None
        )
        
        model_list.append(model)
        
        # Calculate coherence
        coherencemodel = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_v'
        )
        coherence = coherencemodel.get_coherence()
        coherence_values.append(coherence)
        
        print(f"  Coherence score: {coherence:.4f}")
    
    return model_list, coherence_values

# Compute coherence scores
model_list, coherence_values = compute_coherence_values(
    dictionary=dictionary,
    corpus=corpus,
    texts=df_sample['tokens'].tolist(),
    start=5,
    limit=16,
    step=1
)

# Plot results
x = range(5, 16, 1)
plt.figure(figsize=(12, 6))
plt.plot(x, coherence_values, marker='o', linewidth=2, markersize=8)
plt.xlabel("Number of Topics", fontsize=12)
plt.ylabel("Coherence Score", fontsize=12)
plt.title("Topic Coherence Scores", fontsize=14, fontweight='bold')
plt.xticks(x)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('topic_coherence_scores.png', dpi=300, bbox_inches='tight')

# Find optimal
optimal_idx = coherence_values.index(max(coherence_values))
optimal_topics = list(x)[optimal_idx]

print("\n" + "="*70)
print(f"OPTIMAL NUMBER OF TOPICS: {optimal_topics}")
print(f"COHERENCE SCORE: {coherence_values[optimal_idx]:.4f}")
print("="*70)

# Save results
results = {
    'num_topics': list(x),
    'coherence_scores': coherence_values,
    'optimal_topics': optimal_topics
}

with open('coherence_results.pkl', 'wb') as f:
    pickle.dump(results, f)

print("\n✓ Results saved to coherence_results.pkl")
print("✓ Plot saved to topic_coherence_scores.png")
