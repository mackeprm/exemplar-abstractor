# Generate a variation of an input sentence
# Example:
#
# Input:
#
# Expected Output:
#

# Ansatz 1: Wordnet Synonyms:

# Ansatz 2: Use a downloaded paraphrase model
# see: https://github.com/Opla/SmallData-Augmentation-MachineLearning/blob/master/data_augmentation/word_replacment.py

from nltk import word_tokenize
from nltk.corpus import stopwords

stoplist = stopwords.words('english')


def get_synonyms_lexicon(path):
    synonyms_lexicon = {}
    text_entries = [l.strip() for l in open(path, encoding="utf8").readlines()]
    for e in text_entries:
        e = e.split(' ')
        k = e[0]
        v = e[1:len(e)]
        synonyms_lexicon[k] = v
    return synonyms_lexicon


def synonym_replacement(sentence, synonyms_lexicon):
    keys = synonyms_lexicon.keys()
    words = word_tokenize(sentence)
    n_sentence = sentence
    for w in words:
        if w not in stoplist:
            if w in keys:
                n_sentence = n_sentence.replace(w, synonyms_lexicon[w][0])  # we replace with the first synonym
    return n_sentence


if __name__ == '__main__':
    text = 'The technology can be used to investigate how many active wild animals there are in the field.' \
           'This technology could be used to map a patients recovery from severe disfigurement or burns and help show consistent patterns in rehabilitation.' \
           'to find lost livestock'
    sentences = text.split('.')
    #sentences.remove('')
    print(sentences)
    synonyms_lexicon = get_synonyms_lexicon('./models/ppdb-xl.txt')
    for sentence in sentences:
        new_sentence = synonym_replacement(sentence, synonyms_lexicon)
        print(sentence)
        print(new_sentence)
        print('\n')
