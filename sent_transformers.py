from sentence_transformers import SentenceTransformer
import pickle

# Example data
strings_to_train = ["user name"]
word_array = ["user"]

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings for the training strings
train_embeddings = model.encode(strings_to_train, convert_to_tensor=True)

# Save the embeddings and model
with open('sentence_embeddings.pkl', 'wb') as f:
    pickle.dump((train_embeddings, model, strings_to_train), f)

# Load the embeddings and model
with open('sentence_embeddings.pkl', 'rb') as f:
    train_embeddings, model, strings_to_train = pickle.load(f)

import torch

# Function to compute cosine similarity
def compute_cosine_similarity(vec1, vec2):
    cos_sim = torch.nn.functional.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0))
    return cos_sim.item()

# Compute embeddings for the word array
word_embeddings = model.encode(word_array, convert_to_tensor=True)

# Compute similarities between each word and the training strings
similarities = []
for word_embedding in word_embeddings:
    word_similarity = []
    for train_embedding in train_embeddings:
        similarity = compute_cosine_similarity(word_embedding, train_embedding)
        word_similarity.append(similarity)
    similarities.append(word_similarity)

# Print the similarities
for word, similarity in zip(word_array, similarities):
    print(f"Similarities for '{word}':")
    for string, sim in zip(strings_to_train, similarity):
        print(f"  {string}: {sim:.4f}")
