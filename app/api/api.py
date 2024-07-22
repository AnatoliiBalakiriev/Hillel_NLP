from typing import Any
from fastapi import APIRouter, Request
from app.models.predict import PredictRequest, PredictResponse, PreprocessResponse
from app.services.text_processing import clean_text, spacy_clean_text

api_router = APIRouter()


@api_router.post("/predict", response_model=PredictResponse)
async def predict(request: Request, payload: PredictRequest) -> Any:
    """
    ML Prediction API
    """
    input_text = payload.input_text
    model = request.app.state.model

    predict_value = model.predict(input_text)
    return PredictResponse(result=predict_value)


@api_router.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(request: Request, payload: PredictRequest, method: str = "nltk") -> Any:
    """
    Text Preprocessing API
    """
    input_text = payload.input_text

    if method == "spacy":
        processed_text = spacy_clean_text(input_text)
    else:
        processed_text = clean_text(input_text)

    return PreprocessResponse(processed_text=processed_text)
