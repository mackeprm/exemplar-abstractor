import spacy
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
import en_core_web_sm

nlp = en_core_web_sm.load()
nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')

doc = nlp("The technology can be used to investigate how many active wild animals there are in the field.")

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Verbs:", [token.lemma_ for token in doc if token.pos_ == "VERB"])

# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)
