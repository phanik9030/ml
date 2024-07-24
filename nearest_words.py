from gensim.models import KeyedVectors

# Load Google's pre-trained Word2Vec model.
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

def find_top_synonyms(word, top_n=3):
    try:
        synonyms = model.most_similar(positive=[word], topn=top_n)
        return synonyms
    except KeyError:
        return None

word = 'example'  # Replace with your target word
top_synonyms = find_top_synonyms(word)

if top_synonyms:
    print(f"Top {len(top_synonyms)} synonyms for '{word}':")
    for synonym, similarity in top_synonyms:
        print(f"{synonym}: {similarity}")
else:
    print(f"The word '{word}' is not in the vocabulary.")
