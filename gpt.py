import pickle
from transformers import GPT2Tokenizer, GPT2Model
import torch

def encode_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2Model.from_pretrained('gpt2-medium')

# Example input string and definitions
input_string = "example input text"
definitions = [
    "definition one text",
    "definition two text",
    "definition three text"
]

# Encode the input string
input_vector = encode_text(model, tokenizer, input_string)

# Encode the definitions
definition_vectors = [encode_text(model, tokenizer, def_text) for def_text in definitions]

# Save embeddings to a pickle file
with open('embeddings.pkl', 'wb') as f:
    pickle.dump({
        'input_vector': input_vector,
        'definition_vectors': definition_vectors,
        'definitions': definitions
    }, f)


import pickle
from transformers import GPT2Tokenizer, GPT2Model
import torch
from sklearn.metrics.pairwise import cosine_similarity

def encode_text(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
model = GPT2Model.from_pretrained('gpt2-medium')

# Load embeddings from the pickle file
with open('embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
    input_vector = data['input_vector']
    definition_vectors = data['definition_vectors']
    definitions = data['definitions']

# Calculate cosine similarity between input vector and each definition vector
similarities = [cosine_similarity(input_vector.detach().numpy(), def_vector.detach().numpy())[0][0] for def_vector in definition_vectors]

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
