pip install gensim nltk

query_string = "seed tracking progress"
candidate_strings = [
    "Monitoring the growth of seeds in the field",
    "Tracking the development of seed traits",
    "Ensuring seeds are disease-free",
    "Progress updates on seed testing"
]

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return tokens

query_tokens = preprocess(query_string)
candidate_tokens = [preprocess(sentence) for sentence in candidate_strings]

from gensim.models import Word2Vec

# Training a Word2Vec model (optional if using pre-trained)
all_tokens = [query_tokens] + candidate_tokens
model = Word2Vec(sentences=all_tokens, vector_size=100, window=5, min_count=1, workers=4)

# Load a pre-trained model
# model = Word2Vec.load('path/to/pretrained/model')


import numpy as np

def vectorize_sentence(tokens, model):
    vector = np.mean([model.wv[word] for word in tokens if word in model.wv], axis=0)
    return vector

query_vector = vectorize_sentence(query_tokens, model)
candidate_vectors = [vectorize_sentence(tokens, model) for tokens in candidate_tokens]

from scipy.spatial.distance import cosine

similarities = [1 - cosine(query_vector, candidate_vector) for candidate_vector in candidate_vectors]


ranked_candidates = sorted(zip(candidate_strings, similarities), key=lambda x: x[1], reverse=True)

for candidate, similarity in ranked_candidates:
    print(f"Candidate: {candidate}, Similarity: {similarity:.4f}")

