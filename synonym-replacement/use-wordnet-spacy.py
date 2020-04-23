import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
import en_core_web_sm

nlp = en_core_web_sm.load()
nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')

doc = nlp(
    "Can be used in amusement parks or anywhere else to find children who are lost by looking at specific characteristics with how they walk and their size")

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
# for noun_phrase in doc.noun_chunks:
#    print(noun_phrase._.wordnet.synsets())

# Find named entities, phrases and concepts
for token in doc:
    if (token.pos_ == "VERB"):
        print(token._.wordnet.synsets())
