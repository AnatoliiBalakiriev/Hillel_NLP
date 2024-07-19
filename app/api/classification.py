# app/api/classification.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import os
import logging
import nltk
from nltk.corpus import stopwords
from app.utils import preprocess_text, train_model
from app.services.train_doc2vec import train_doc2vec, load_doc2vec
from app.services.train_lda import train_lda, load_lda
from app.services.text_processing import clean_text

# Logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Ensure stopwords are downloaded
nltk.download('stopwords')

router = APIRouter()

# Defining paths to the data and model
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/IMDB Dataset.csv'))
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model.pkl'))
doc2vec_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'doc2vec_model.pkl'))
lda_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'lda_model.model'))

# Checking the existence of the model, if it does not exist - training the model
if not os.path.exists(model_path):
    logging.info("Model not found, training model...")
    accuracy = train_model(data_path, model_path)
    logging.info('Training complete, accuracy: %.2f', accuracy)

if not os.path.exists(doc2vec_model_path):
    logging.info("Doc2Vec model not found, training model...")
    train_doc2vec(doc2vec_model_path)

if not os.path.exists(lda_model_path):
    logging.info("LDA model not found, training model...")
    train_lda(lda_model_path, 20)

# Loading the models
model = joblib.load(model_path)
doc2vec_model = load_doc2vec(doc2vec_model_path)
lda_model, lda_dictionary = load_lda(lda_model_path)


class TextRequest(BaseModel):
    text: str


class TextResponse(BaseModel):
    text: str
    label: str


@router.post("/classify", response_model=TextResponse)
async def classify_text(request: TextRequest):
    try:
        prediction = model.predict([request.text])[0]
        return TextResponse(text=request.text, label=prediction)
    except Exception as e:
        logging.error("Error while classifying text: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
async def train_model_endpoint():
    try:
        training_accuracy = train_model(data_path, model_path)
        return {"accuracy": training_accuracy}
    except Exception as e:
        logging.error("Error while training model: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


class GroupSentencesRequest(BaseModel):
    sentences: list[str]


class GroupSentencesResponse(BaseModel):
    groups: dict[str, list[str]]


def get_topic_name(topic_keywords):
    stop_words = set(stopwords.words('english'))
    words = topic_keywords.split(' + ')
    for word in words:
        keyword = word.split('*')[1].replace('"', '')
        if keyword not in stop_words:
            return keyword.capitalize()  # Capitalize the keyword to make it look more like a title
    return words[0].split('*')[1].replace('"', '').capitalize()  # Return the first word if all others are stop words


@router.post("/group_sentences", response_model=GroupSentencesResponse)
async def group_sentences(request: GroupSentencesRequest):
    try:
        sentences = request.sentences
        vectors = [doc2vec_model.infer_vector(clean_text(sentence)) for sentence in sentences]

        # Use DBSCAN clustering algorithm
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=0.5, min_samples=20)
        labels = dbscan.fit_predict(vectors)

        # Assign topics using LDA model
        topics = {}
        for idx, sentence in enumerate(sentences):
            bow = lda_dictionary.doc2bow(clean_text(sentence))
            topic_distribution = lda_model[bow]
            main_topic = max(topic_distribution, key=lambda item: item[1])[0]
            topic_keywords = lda_model.print_topic(main_topic)
            topic_name = get_topic_name(topic_keywords)
            if topic_name in topics:
                topics[topic_name].append(sentence)
            else:
                topics[topic_name] = [sentence]

        return GroupSentencesResponse(groups=topics)
    except Exception as e:
        logging.error("Error while grouping sentences: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))
