import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example phrases and sentences
phrases = ["eating apple", "favorite fruit", "vitamin C"]
sentences = [
    "I love eating an apple every day.",
    "Bananas are my favorite fruit.",
    "Oranges are rich in vitamin C.",
    "Oranges I have an apple and a banana.",
    "The orange tree is in my backyard.",
]

# Combine phrases and sentences for vectorization
all_text = phrases + sentences

# Create TF-IDF vectorizer and transform the text
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_text)

# Separate the vectors of phrases and sentences
phrase_vectors = tfidf_matrix[:len(phrases)]
sentence_vectors = tfidf_matrix[len(phrases):]

# Calculate cosine similarity between phrases and sentences
similarity_matrix = cosine_similarity(phrase_vectors, sentence_vectors)

# Display the similarity matrix
print("Similarity matrix:")
print(similarity_matrix)

# Match phrases to sentences based on highest similarity
for i, phrase in enumerate(phrases):
    most_similar_sentence_index = np.argmax(similarity_matrix[i])
    most_similar_sentence = sentences[most_similar_sentence_index]
    similarity_score = similarity_matrix[i][most_similar_sentence_index]
    print(f"The phrase '{phrase}' is most similar to the sentence: '{most_similar_sentence}' with a similarity score of {similarity_score:.2f}")
