# Ansatz 1: Wordnet Synonyms with NLTK
from nltk.tokenize import word_tokenize
# https://www.nltk.org/api/nltk.corpus.reader.html#module-nltk.corpus.reader.wordnet
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
import random

stoplist = stopwords.words('english')


def find_synonyms_for_term(term):
    synsets = wn.synsets(term)
    if (synsets):
        first_result = synsets[0]
        lemmas = first_result.lemmas()
        possible_synonyms = list(map(lambda l: l.name(), lemmas))
        #print(possible_synonyms)
        if len(possible_synonyms) > 1:
            index = random.randrange(len(possible_synonyms))
            #print(index)
            return possible_synonyms[index]
        else:
            return None
    else:
        return None


def replace_all_words_within(sentence):
    # Tokenize
    result = sentence
    terms = word_tokenize(sentence)
    #print(terms)
    for term in terms:
        if term and (term not in stoplist):
            synonym = find_synonyms_for_term(term)
            #print(synonym)
            if(synonym):
                result = result.replace(term, synonym)
    return result

if __name__ == '__main__':
    sentences = [
        "The technology can be used to investigate how many active wild animals there are in the field."]
    for sentence in sentences:
        print(sentence)
        print(replace_all_words_within(sentence))
        print('\n')
