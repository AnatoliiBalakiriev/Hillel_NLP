import logging
import os
from sklearn.datasets import fetch_20newsgroups
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from app.services.text_processing import clean_text
import joblib

# Logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_lda(model_path, num_topics=20):
    logging.info("Fetching 20 newsgroups data...")
    newsgroups_data = fetch_20newsgroups(subset='all')
    texts = [clean_text(doc) for doc in newsgroups_data.data]

    logging.info("Creating dictionary and corpus for LDA...")
    dictionary = Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    logging.info("Training LDA model...")
    lda_model = LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    logging.info("Saving LDA model and dictionary...")
    lda_model.save(model_path)
    dictionary.save(os.path.join(os.path.dirname(model_path), 'lda_dictionary.dict'))

    return lda_model, dictionary


def load_lda(model_path):
    lda_model = LdaModel.load(model_path)
    dictionary = Dictionary.load(os.path.join(os.path.dirname(model_path), 'lda_dictionary.dict'))
    return lda_model, dictionary
