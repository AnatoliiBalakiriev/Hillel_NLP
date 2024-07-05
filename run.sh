#!/bin/bash

# Installing dependencies
echo "Installing dependencies..."
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

# Starting the FastAPI server
echo "Starting FastAPI server..."
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 9000
if [ $? -ne 0 ]; then
  echo "Error during FastAPI server start"
  exit 1
fi