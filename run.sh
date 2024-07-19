#!/bin/bash

# Removing old virtual environment
echo "Removing old virtual environment..."
poetry env remove python

# Creating new virtual environment and installing dependencies
echo "Creating new virtual environment..."
poetry install --no-root
if [ $? -ne 0 ]; then
  echo "Error during dependencies installation"
  exit 1
fi

# Loading NLTK resources
echo "Loading NLTK resources..."
poetry run python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
if [ $? -ne 0 ]; then
  echo "Error during NLTK resources download"
  exit 1
fi

# Checking and training models if not present
echo "Checking and training models if not present..."
poetry run python -c "
import os
from app.services.train_doc2vec import train_doc2vec, load_doc2vec
from app.services.train_lda import train_lda, load_lda

doc2vec_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'app/api/doc2vec_model.pkl'))
lda_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'app/api/lda_model.model'))

if not os.path.exists(doc2vec_model_path):
    print('Doc2Vec model not found, training model...')
    train_doc2vec(doc2vec_model_path, epochs=40)
else:
    print('Doc2Vec model already exists.')

if not os.path.exists(lda_model_path):
    print('LDA model not found, training model...')
    train_lda(lda_model_path, num_topics=10)
else:
    print('LDA model already exists.')
"
if [ $? -ne 0 ]; then
  echo "Error during model check or training"
  exit 1
fi

# Starting the FastAPI server
echo "Starting FastAPI server..."
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 7010
if [ $? -ne 0 ]; then
  echo "Error during FastAPI server start"
  exit 1
fi
