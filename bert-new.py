import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

data = [
    {
        'name': 'test data',
        'description': 'test data desc'
    }
]

# Convert to DataFrame
data_df = pd.DataFrame(data)

# Combine name and description for text representation
data_df['combined'] = data_df['name'] + " " + data_df['description']

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings
def get_bert_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

embeddings_list = []
for text in glossary_df['combined']:
    embedding = get_bert_embedding(text, tokenizer, model)
    embeddings_list.append(embedding)

# Add the embeddings to the DataFrame
data_df['embedding'] = embeddings_list

def find_most_similar_data(user_input, data_df, tokenizer, model):
    input_embedding = get_bert_embedding(user_input, tokenizer, model)
    embeddings = np.vstack(data_df['embedding'].values)
    similarities = cosine_similarity(input_embedding, embeddings)
    most_similar_idx = similarities.argmax()
    most_similar_entry = data_df.iloc[most_similar_idx]
    return most_similar_entry['name'], most_similar_entry['description']

# Example user input
user_input = "test input"
name, description = find_most_similar_data(user_input, data_df, tokenizer, model)
print(f"Most similar data entry: {name}\nDescription: {description}")
