from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize models
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
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

def generate_prompt_suggestions(user_query, relevant_sentence, num_prompts=5):
    prompts = []
    for _ in range(num_prompts):
        prompt = f"Based on the user's input: '{user_query}', and the relevant sentence: '{relevant_sentence}', suggest a follow-up question or prompt to refine the results."
        
        # Encode the prompt
        inputs = tokenizer.encode(prompt, return_tensors='pt')
        
        # Generate response
        output = model.generate(inputs, max_length=100, num_return_sequences=1, num_beams=5, early_stopping=True)
        
        # Decode the response
        dynamic_prompt = tokenizer.decode(output[0], skip_special_tokens=True)
        
        prompts.append(dynamic_prompt)
    
    return prompts

# Example user input
user_query = "Tell me more about the fox and the dog."

# Find the most relevant sentence
relevant_sentence = find_most_similar_sentence(user_query)

# Generate multiple prompt suggestions
prompt_suggestions = generate_prompt_suggestions(user_query, relevant_sentence, num_prompts=5)

print(f"Relevant Sentence: {relevant_sentence}")
print("Prompt Suggestions:")
for i, prompt in enumerate(prompt_suggestions, 1):
    print(f"{i}. {prompt}")
