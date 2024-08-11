from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os

load_dotenv()


class Settings(BaseSettings):
    PROJECT_NAME: str = "Hillel NLP project"
    API_V1_STR: str = "/api/v1"

    MODEL_PATH: str = "app/ml_model/model.pth"
    HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN")

    class Config:
        case_sensitive = True


settings = Settings()
