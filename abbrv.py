import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Load MiniLM model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Custom vocabulary of common database column names
db_vocab = [
    'test'
]

# Create embeddings for the database vocabulary
word_embeddings = model.encode(db_vocab)

# Build a FAISS index
index = faiss.IndexFlatL2(word_embeddings.shape[1])  # L2 distance
index.add(word_embeddings.astype(np.float32))  # Add embeddings to the index

# Example abbreviations
abbreviations = ['tst']

# Function to predict the full form of an abbreviation
def predict_full_form(abbreviation, model, index, db_vocab):
    # Encode the abbreviation
    abbr_embedding = model.encode([abbreviation]).astype(np.float32)

    # Search for the closest matches in the FAISS index
    distances, indices = index.search(abbr_embedding, k=10)  # Lower k since we're using a smaller, relevant vocab

    # Get the predicted full forms
    predicted_full_forms = [db_vocab[idx] for idx in indices[0]]

    # Filter results
    matching_full_forms = []
    for word in predicted_full_forms:
        if (word.lower().startswith(abbreviation.lower()[0])  # Start with the same letter
                and len(word) > len(abbreviation)  # Ensure the word is longer than the abbreviation
                and all(char in word.lower() for char in abbreviation.lower())):  # All abbreviation chars in word
            matching_full_forms.append(word)

    # Return the matching full forms or a message if none are found
    return matching_full_forms if matching_full_forms else [f"No matching full form found for abbreviation '{abbreviation}'."]

# Predict full forms for the example abbreviations
predictions = {abbr: predict_full_form(abbr, model, index, db_vocab) for abbr in abbreviations}

# Output the predictions
for abbr, prediction in predictions.items():
    print(f"Abbreviation: {abbr}, Predicted full forms: {prediction}")
