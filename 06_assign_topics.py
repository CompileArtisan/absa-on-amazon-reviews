import pandas as pd
import pickle
from gensim import corpora
from gensim.models import LdaModel

# Load data and models
print("Loading data and models...")
df = pd.read_csv('all_beauty_preprocessed.csv')
df['tokens'] = df['tokens'].apply(eval)

dictionary = corpora.Dictionary.load('beauty_dictionary.dict')
lda_model = LdaModel.load('beauty_lda_final.model')

with open('topic_labels.pkl', 'rb') as f:
    topic_labels = pickle.load(f)

# Create corpus
corpus = [dictionary.doc2bow(tokens) for tokens in df['tokens']]

print("\nAssigning topics to reviews...")

def get_document_topics(lda_model, corpus):
    """Extract topic probabilities for each document"""
    all_topics = []
    
    for i, doc_topics in enumerate(lda_model[corpus]):
        if (i + 1) % 50000 == 0:
            print(f"  Processed {i + 1} reviews...")
        
        # Get topic probabilities
        topic_probs = dict(doc_topics[0])
        
        # Find dominant topic
        if topic_probs:
            dominant_topic = max(topic_probs.items(), key=lambda x: x[1])
            
            # Create dictionary with all topic probabilities
            topic_dict = {
                'dominant_topic': dominant_topic[0],
                'dominant_prob': dominant_topic[1]
            }
            
            # Add individual topic probabilities
            for topic_id in range(lda_model.num_topics):
                topic_dict[f'topic_{topic_id}_prob'] = topic_probs.get(topic_id, 0.0)
            
            all_topics.append(topic_dict)
        else:
            # Handle edge case
            all_topics.append({
                'dominant_topic': -1,
                'dominant_prob': 0.0
            })
    
    return pd.DataFrame(all_topics)

# Get topic assignments
topics_df = get_document_topics(lda_model, corpus)

# Merge with original data
df = pd.concat([df.reset_index(drop=True), topics_df], axis=1)

# Map topic labels
df['topic_label'] = df['dominant_topic'].map(topic_labels)
df.loc[df['dominant_topic'] == -1, 'topic_label'] = 'Unknown'

# Save
df.to_csv('all_beauty_with_topics.csv', index=False)

print("\n" + "="*70)
print("TOPIC ASSIGNMENT COMPLETE")
print("="*70)
print(f"\nTopic distribution:")
print(df['topic_label'].value_counts())

print("\n✓ Data with topics saved to all_beauty_with_topics.csv")
