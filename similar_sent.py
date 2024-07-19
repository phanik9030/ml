from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def embed_word(word):
    tokens = tokenizer(word, return_tensors='pt')
    outputs = model(**tokens)
    # Get the embeddings for the token (use the mean of the embeddings)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return embeddings

def sentence_similarity(input_words, sentences):
    # Embed each input word
    input_embeddings = [embed_word(word) for word in input_words]

    sentence_scores = []
    for sentence in sentences:
        # Tokenize and embed each word in the sentence
        sentence_words = sentence.split()
        sentence_embeddings = [embed_word(word) for word in sentence_words]

        # Calculate the similarity between each input word and each word in the sentence
        scores = []
        for input_emb in input_embeddings:
            for sent_emb in sentence_embeddings:
                score = cosine_similarity(input_emb, sent_emb)[0][0]
                scores.append(score)
        
        # Aggregate similarity scores (e.g., average)
        average_score = np.mean(scores)
        sentence_scores.append((sentence, average_score))
    
    # Sort sentences by similarity score
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    
    return sentence_scores

# Example usage
input_words = ["find", "similarity"]
sentences = [
    "This is a test sentence to find the similarity.",
    "Another example to calculate the similarity between words.",
    "This sentence does not have much in common.",
    "Similarity can be measured in various ways."
]

results = sentence_similarity(input_words, sentences)
for sentence, score in results:
    print(f"Sentence: {sentence}, Score: {score:.4f}")
