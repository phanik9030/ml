import torch
from transformers import BertTokenizer, BertModel
import pickle

# Example data
strings_to_train = ["your list of more than 100 strings here"]
word_array = ["approved date", "loaded date"]

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Set the model to evaluation mode
model.eval()

# Function to get BERT embeddings
def get_bert_embedding(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Generate BERT embeddings for the training strings
train_embeddings = [get_bert_embedding(text, model, tokenizer) for text in strings_to_train]


# Save the embeddings, model, and tokenizer
with open('bert_embeddings.pkl', 'wb') as f:
    pickle.dump((train_embeddings, model, tokenizer, strings_to_train), f)


# Load the embeddings, model, and tokenizer
with open('bert_embeddings.pkl', 'rb') as f:
    train_embeddings, model, tokenizer, strings_to_train = pickle.load(f)


# Function to compute cosine similarity
def compute_cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Compute BERT embeddings for the word array
word_embeddings = [get_bert_embedding(word, model, tokenizer) for word in word_array]

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
