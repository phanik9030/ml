from collections import Counter

def compute_common_words(sentences, threshold=2):
    words = []
    for sentence in sentences:
        words.extend(sentence.split())
    
    word_counts = Counter(words)
    common_words = {word for word, count in word_counts.items() if count >= threshold}
    
    return common_words

# Example sentences
sentences = [
    "Attorney full name",
    "Banker name",
    "Doctor name"
]

common_words = compute_common_words(sentences)
print("Common words:", common_words)

import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embed_word(word):
    tokens = tokenizer(word, return_tensors='pt')
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

# Load embeddings from pickle file
with open('sentence_embeddings.pkl', 'rb') as f:
    sentence_embeddings = pickle.load(f)

def compute_common_words(sentences, threshold=2):
    words = []
    for sentence in sentences:
        words.extend(sentence.split())
    
    word_counts = Counter(words)
    common_words = {word for word, count in word_counts.items() if count >= threshold}
    
    return common_words

# Example sentences
sentences = [
    "name",
    "name",
    "name"
]

common_words = compute_common_words(sentences)

def sentence_similarity(input_words, sentence_embeddings, common_words):
    # Embed each input word
    input_embeddings = [embed_word(word) for word in input_words]

    sentence_scores = []
    for sentence, embeddings in sentence_embeddings.items():
        sentence_words = list(embeddings.keys())
        
        scores = []
        for input_emb in input_embeddings:
            word_scores = []
            for sent_word, sent_emb in embeddings.items():
                sim_score = cosine_similarity(input_emb.reshape(1, -1), sent_emb.reshape(1, -1))[0][0]
                if sent_word in common_words:
                    sim_score *= 0.5  # Reduce priority of common words
                word_scores.append(sim_score)
            scores.append(max(word_scores))
        
        # Assign higher weight to scores from uncommon words
        weighted_scores = [score if word not in common_words else score * 0.5 for word, score in zip(sentence_words, scores)]
        
        average_score = np.mean(weighted_scores)
        
        sentence_scores.append((sentence, average_score))
    
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    
    return sentence_scores

# Example input words
input_words = ["Name", "Name"]

# Calculate similarity
results = sentence_similarity(input_words, sentence_embeddings, common_words)
for sentence, score in results:
    print(f"Sentence: {sentence}, Score: {score:.4f}")
