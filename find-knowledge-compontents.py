# Find key components within a sentence
# Example 1:
# Input:
# "The technology can be used to investigate how many active wild animals there are in the field."
# Expected Output:
# Technology, Investigate, Wild Animals, Field
# Example 2:
# Input:
# "This technology could be used to map a patient's recovery from severe disfigurement or burns and help show consistent patterns in rehabilitation."
# Expected Output:
# Technology, map, recovery, show patterns?

import en_core_web_lg

nlp = en_core_web_lg.load()
doc = nlp("The technology can be used to investigate how many active wild animals there are in the field.")
for word in doc:
    if word.dep_ in ('xcomp', 'ccomp'):
        print(''.join(w.text_with_ws for w in word.subtree))