from SPARQLWrapper import SPARQLWrapper, JSON
from gensim import corpora
from gensim.models import LsiModel
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt


def toLocalId(input_id):
    return input_id.split("/")[len(input_id.split("/")) - 1]


def truncate(input_string, length):
    return (input_string[:length] + '..') if len(input_string) > length else input_string


def load_data():
    sparql = SPARQLWrapper("https://innovonto-core.imp.fu-berlin.de/management/core/query")
    sparql.setQuery("""
        PREFIX gi2mo: <http://purl.org/gi2mo/ns#>  
        PREFIX dcterms: <http://purl.org/dc/terms/>
        PREFIX inov:<http://purl.org/innovonto/types/#>

        SELECT ?idea ?content WHERE {
          ?idea a gi2mo:Idea;
                gi2mo:content ?content;
                gi2mo:hasIdeaContest <https://innovonto-core.imp.fu-berlin.de/entities/ideaContests/i2m-bionic-radar>.

        }
        ORDER BY ASC(?idea)
    """)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    labels = [(str(i) + " : " + str(truncate(binding["content"]["value"], 60))) for i, binding in
              enumerate(results["results"]["bindings"])]
    contents = list(map(lambda binding: binding["content"]["value"], results["results"]["bindings"]))

    return contents, labels


def preprocess_data(doc_set):
    # initialize regex tokenizer
    tokenizer = RegexpTokenizer(r'\w+')
    # create English stop words list
    en_stop = set(stopwords.words('english'))
    # Create p_stemmer of class PorterStemmer
    # TODO this produces bad results!
    p_stemmer = PorterStemmer()
    # list for tokenized documents in loop
    texts = []
    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)
        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]
        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]
        # add tokens to list
        texts.append(stemmed_tokens)
    return texts


def prepare_corpus(doc_clean):
    # Creating the term dictionary of our courpus, where every unique term is assigned an index. dictionary = corpora.Dictionary(doc_clean)
    dictionary = corpora.Dictionary(doc_clean)
    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    # generate LDA model
    return dictionary, doc_term_matrix


def create_gensim_lsa_model(doc_clean, number_of_topics, words):
    dictionary, doc_term_matrix = prepare_corpus(doc_clean)
    # generate LSA model
    lsamodel = LsiModel(doc_term_matrix, num_topics=number_of_topics, id2word=dictionary)  # train model
    print(lsamodel.print_topics(num_topics=number_of_topics, num_words=words))
    return lsamodel


def compute_coherence_values(dictionary, doc_term_matrix, doc_clean, stop, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, stop, step):
        # generate LSA model
        model = LsiModel(doc_term_matrix, num_topics=num_topics, id2word=dictionary)  # train model
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=doc_clean, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values


def plot_graph(doc_clean, start, stop, step):
    dictionary, doc_term_matrix = prepare_corpus(doc_clean)
    model_list, coherence_values = compute_coherence_values(dictionary, doc_term_matrix, doc_clean,
                                                            stop, start, step)
    # Show graph
    x = range(start, stop, step)
    plt.plot(x, coherence_values)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()


if __name__ == '__main__':
    contents, labels = load_data()
    clean_text = preprocess_data(contents)
    #start, stop, step = 2, 12, 1
    #plot_graph(clean_text, start, stop, step)
    number_of_topics = 6
    words = 10
    model = create_gensim_lsa_model(clean_text, number_of_topics, words)
    print(model)