import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# Initialize the all-MiniLM model for sentence embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Sample data
data = [
    {"title": "test", "sentence": "testing."}
]

# Encode the sentences
sentences = [entry["sentence"] for entry in data]
embeddings = model.encode(sentences)

# Create a FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save the FAISS index and data
faiss.write_index(index, 'faiss_index')
with open('data.pkl', 'wb') as f:
    pickle.dump(data, f)


import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load the saved FAISS index and data
index = faiss.read_index('faiss_index')
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

# Initialize the model for generating embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Initialize the question generation pipeline
question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qa-qg-hl")


# Function to find closest matches using FAISS
def find_closest_match_faiss(user_input, data, top_n=3):
    input_vec = model.encode([user_input])
    distances, indices = index.search(input_vec, top_n)

    # Convert distances to similarity percentages
    max_distance = np.max(distances)
    matches = []
    for i in range(top_n):
        similarity = (1 - (distances[0][i] / max_distance)) * 100
        matches.append((similarity, data[indices[0][i]]))

    return matches

# Function to generate questions using text2text-generation
# def generate_questions(sentences):
#     questions = []
#     for entry in sentences:
#         sentence = entry['sentence']
#         # generated_text = question_generator(f"highlight: {sentence}")[0]['generated_text']
#         generated_text =  question_generator(f"generate question: {sentence}")[0]['generated_text']
#         formatted_question = f"Is it related to {generated_text.lower()}".replace(" what is the","")
#         questions.append((formatted_question, entry))
#     return questions

# Function to generate questions using text2text-generation
def generate_questions(matches):
    questions = []
    for distance, entry in matches:
        sentence = entry['sentence']
        generated_text = question_generator(f"generate question: {sentence}")[0]['generated_text']
        formatted_question = f"Is it related to {generated_text.lower()}".replace(" what is", "")
        questions.append((formatted_question, entry, distance))
    return questions

# User input
user_input = "test data"
matches = find_closest_match_faiss(user_input, data)

if matches:
    print("Multiple matches found. Please answer the following questions to help us narrow down:")
    questions = generate_questions(matches)
    for idx, (question, entry, similarity) in enumerate(questions):
        print(f"{idx + 1}. {question} (Score: {similarity:.2f}%)")

    user_response = int(input("Select the number corresponding to the best match: "))
    best_match = matches[user_response - 1][1]
    print(f"Best match found: Title: {best_match['title']}, Sentence: {best_match['sentence']} (Score: {matches[user_response - 1][0]:.2f}%)")
else:
    print("No close matches found.")
