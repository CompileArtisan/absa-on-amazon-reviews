import pandas as pd
import pickle
from gensim import corpora
from gensim.models import LdaModel
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis

# Load data
print("Loading data...")
df = pd.read_csv('all_beauty_preprocessed.csv')
df['tokens'] = df['tokens'].apply(eval)

# Load dictionary
dictionary = corpora.Dictionary.load('beauty_dictionary.dict')

# Create corpus for full dataset
print("\nCreating corpus for full dataset...")
corpus = [dictionary.doc2bow(tokens) for tokens in df['tokens']]

# Load optimal number of topics
with open('coherence_results.pkl', 'rb') as f:
    results = pickle.load(f)
    num_topics = results['optimal_topics']

print(f"\nTraining final LDA model with {num_topics} topics...")

# Train final model
lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=num_topics,
    random_state=42,
    chunksize=2000,
    passes=15,          # More passes for final model
    iterations=400,
    alpha='auto',
    eta='auto',
    per_word_topics=True,
    eval_every=10
)

# Save model
lda_model.save('beauty_lda_final.model')

print("\n" + "="*70)
print("DISCOVERED TOPICS")
print("="*70)

# Print topics
for idx, topic in lda_model.print_topics(num_topics, num_words=15):
    print(f"\nTopic {idx}:")
    print(f"  {topic}")

# Manually label topics based on keywords (you'll need to do this)
# These are EXAMPLE labels - you must create your own based on the actual topics
topic_labels = {
    0: "Hair Care & Styling",
    1: "Skin Care & Moisturizers",
    2: "Makeup & Cosmetics",
    3: "Nail Care & Polish",
    4: "Scent & Fragrance",
    5: "Product Quality & Packaging",
    6: "Price & Value",
    7: "Shipping & Delivery",
    # Add more based on your num_topics
}

# Save topic labels
with open('topic_labels.pkl', 'wb') as f:
    pickle.dump(topic_labels, f)

# Create visualization
print("\nCreating interactive visualization...")
vis = gensimvis.prepare(lda_model, corpus, dictionary, mds='mmds')
pyLDAvis.save_html(vis, 'lda_visualization.html')

print("\n✓ Model saved to beauty_lda_final.model")
print("✓ Topic labels saved to topic_labels.pkl")
print("✓ Visualization saved to lda_visualization.html")
print("\n>>> Open lda_visualization.html in your browser to explore topics!")
