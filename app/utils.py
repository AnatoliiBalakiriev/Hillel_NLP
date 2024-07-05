# app/utils.py
import logging
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import joblib

# Loading required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def preprocess_text(text, lower=True, remove_stopwords=True, stemming=False, lemmatization=True):
    # Lower case
    if lower:
        text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Removal of stop words
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    if stemming:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

    # Lemmatization
    if lemmatization:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens


def train_model(data_path, model_path):
    logging.info("Loading data...")
    df = pd.read_csv(data_path)
    X = df['review']
    y = df['sentiment']

    logging.info("Separation of data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logging.info("Creating pipeline for vectorization and classification...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=preprocess_text)),
        ('clf', MultinomialNB())
    ])

    logging.info("Training the model...")
    pipeline.fit(X_train, y_train)

    logging.info("Evaluating the model...")
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    logging.info("Saving model...")
    joblib.dump(pipeline, model_path)

    logging.info("Training completed with accuracy: %.2f", accuracy)
    return accuracy
