import nltk
from pattern.en import conjugate, lemma, PRESENT, PAST, PARTICIPLE, GERUND

nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_tenses(word):
    lemmatized = lemma(word)
    past = conjugate(lemmatized, tense=PAST)
    present = conjugate(lemmatized, tense=PRESENT)
    gerund = conjugate(lemmatized, tense=GERUND)
    
    return {
        "base": lemmatized,
        "past": past,
        "present": present,
        "gerund": gerund
    }

word = "load"
tenses = get_tenses(word)
print(tenses)

def get_all_forms(word):
    base_form = lemma(word)
    past = conjugate(base_form, tense=PAST)
    present = conjugate(base_form, tense=PRESENT)
    gerund = conjugate(base_form, tense=GERUND)
    
    return {
        "base": base_form,
        "past": past,
        "present": present,
        "gerund": gerund
    }

word = "loading"
forms = get_all_forms(word)
print(forms)
