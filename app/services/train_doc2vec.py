import logging
import os
from sklearn.datasets import fetch_20newsgroups
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from app.services.text_processing import clean_text
import joblib

# Logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_doc2vec(model_path, epochs=40):
    logging.info("Fetching 20 newsgroups data...")
    newsgroups_data = fetch_20newsgroups(subset='all')
    unique_groups = set(newsgroups_data.target)
    print(f"Number of unique groups in 20 Newsgroups dataset: {len(unique_groups)}")
    documents = [TaggedDocument(words=clean_text(doc), tags=[str(i)]) for i, doc in enumerate(newsgroups_data.data)]

    logging.info("Training Doc2Vec model...")
    model = Doc2Vec(documents, vector_size=100, window=5, min_count=2, workers=4, epochs=epochs)

    logging.info("Saving Doc2Vec model...")
    model.save(model_path)

    return model


def load_doc2vec(model_path):
    return Doc2Vec.load(model_path)
