# app/api/classification.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import joblib
import os
import logging
from app.utils import preprocess_text, train_model

# Logging settings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

router = APIRouter()

# Defining paths to the data and model
data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/IMDB Dataset.csv'))
model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'model.pkl'))

# Checking the existence of the model, if it does not exist - training the model
if not os.path.exists(model_path):
    logging.info("Model not found, training model...")
    accuracy = train_model(data_path, model_path)
    logging.info('Training complete, accuracy: %.2f', accuracy)

# Loading a saved model
model = joblib.load(model_path)


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
