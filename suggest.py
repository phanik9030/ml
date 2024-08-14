from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Initialize models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load GPT-2 model and tokenizer
gpt2_model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
model = GPT2LMHeadModel.from_pretrained(gpt2_model_name)

# Example sentences
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast, agile fox leaps over a sleepy canine.",
    # Add more sentences here
]

# Generate sentence embeddings
sentence_embeddings = sentence_model.encode(sentences)

def find_most_similar_sentence(user_query):
    query_embedding = sentence_model.encode([user_query])
    similarities = cosine_similarity(query_embedding, sentence_embeddings)
    most_similar_idx = similarities.argmax()
    return sentences[most_similar_idx]

def generate_dynamic_prompt(user_query, relevant_sentence):
    prompt = f"Based on the user's input: '{user_query}', and the relevant sentence: '{relevant_sentence}', suggest a follow-up question or prompt to refine the results."
    
    # Encode the prompt
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate response
    output = model.generate(inputs, max_length=100, num_return_sequences=1)
    
    # Decode the response
    dynamic_prompt = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return dynamic_prompt

# Example user input
user_query = "Tell me more about the fox and the dog."

# Find the most relevant sentence
relevant_sentence = find_most_similar_sentence(user_query)

# Generate a dynamic prompt
dynamic_prompt = generate_dynamic_prompt(user_query, relevant_sentence)

print(f"Relevant Sentence: {relevant_sentence}")
print(f"Dynamic Prompt: {dynamic_prompt}")
