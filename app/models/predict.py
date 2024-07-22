from pydantic import BaseModel, Field, StrictStr


class PredictRequest(BaseModel):
    input_text: StrictStr = Field(..., title="input_text", description="Input text", example="Input text for ML")
    method: StrictStr = Field("nltk", title="method", description="Preprocessing method (nltk or spacy)",
                              example="nltk")


class PredictResponse(BaseModel):
    result: float = Field(..., title="result", description="Predict value", example=0.9)


class PreprocessResponse(BaseModel):
    processed_text: list[str] = Field(..., title="processed_text", description="Processed text as a list of tokens")
