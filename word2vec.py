import gensim.downloader as api

# Load pre-trained Word2Vec model
word2vec_model = api.load('word2vec-google-news-300')

import numpy as np

def get_sentence_embedding(model, sentence):
    words = sentence.split()
    valid_words = [word for word in words if word in model]
    if not valid_words:  # If no valid words, return a zero vector
        return np.zeros(model.vector_size)
    word_vectors = [model[word] for word in valid_words]
    return np.mean(word_vectors, axis=0)

# Example input string and definitions
input_string = "example input text"
definitions = ["definition text " + str(i) for i in range(30000)]

# Generate embeddings for the input string and definitions
input_vector = get_sentence_embedding(word2vec_model, input_string)
definition_vectors = [get_sentence_embedding(word2vec_model, def_text) for def_text in definitions]

from sklearn.metrics.pairwise import cosine_similarity

# Calculate cosine similarity between input vector and each definition vector
similarities = cosine_similarity([input_vector], definition_vectors)[0]

# Combine definitions with their similarities
definition_similarity_pairs = list(zip(definitions, similarities))

# Sort the definitions by similarity scores in descending order
sorted_definitions = sorted(definition_similarity_pairs, key=lambda x: x[1], reverse=True)

# Get the top three results with their scores
top_three = sorted_definitions[:3]

# Output the top three results with their scores
for idx, (definition, score) in enumerate(top_three):
    print(f"Rank {idx + 1}:")
    print(f"Definition: {definition}")
    print(f"Similarity Score: {score}\n")
