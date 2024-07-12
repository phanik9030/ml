import torch
from transformers import BertTokenizer, BertModel
import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download necessary NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Set the model to evaluation mode
model.eval()

# Function to get the BERT embeddings for a word
def get_bert_embedding(word):
    inputs = tokenizer(word, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    # Take the mean of the token embeddings (excluding special tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

# Example words
words = ["approve", "approved", "payment", "payments", "amount", "money"]

# Lemmatize words
lemmatized_words = [lemmatizer.lemmatize(word) for word in words]

# Get BERT embeddings for the lemmatized words
word_embeddings = [get_bert_embedding(word) for word in lemmatized_words]

# Compute cosine similarity
similarity_matrix = cosine_similarity(word_embeddings)

# Print the similarity matrix
print("Similarity matrix:")
print(similarity_matrix)

# Function to find synonyms using WordNet
def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

# Example usage: Find synonyms for a word
word = 'approve'
synonyms = get_synonyms(word)
print(f"Synonyms for {word}: {synonyms}")

# Compute similarity between specific words
similarity_approve_approved = cosine_similarity([get_bert_embedding('approve')], [get_bert_embedding('approved')])
print(f"Similarity between 'approve' and 'approved': {similarity_approve_approved[0][0]}")

similarity_payment_payments = cosine_similarity([get_bert_embedding('payment')], [get_bert_embedding('payments')])
print(f"Similarity between 'payment' and 'payments': {similarity_payment_payments[0][0]}")

similarity_amount_money = cosine_similarity([get_bert_embedding('amount')], [get_bert_embedding('money')])
print(f"Similarity between 'amount' and 'money': {similarity_amount_money[0][0]}")
