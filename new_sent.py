from sentence_transformers import SentenceTransformer, util

# Load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')


def find_best_match_sentence_transformer(input_string, phrases, model):
    # Encode texts
    embeddings = model.encode([input_string] + phrases, convert_to_tensor=True)

    # Compute cosine similarity
    input_embedding = embeddings[0]
    phrase_embeddings = embeddings[1:]
    similarities = util.pytorch_cos_sim(input_embedding, phrase_embeddings)

    # Find the index of the most similar phrase
    best_match_index = similarities.argmax()
    return phrases[best_match_index]

phrases = ["test"]

input_string = "testing"

best_match_phrase = find_best_match_sentence_transformer(input_string, phrases, model)

print(f"The best match for '{input_string}' is '{best_match_phrase}'")
