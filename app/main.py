import os
import torch
from fastapi import FastAPI
from mistral_inference.transformer import Transformer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

from app.api.api import api_router
from app.api.heartbeat import heartbeat_router
from app.core.config import settings
from app.core.event_handler import start_app_handler, stop_app_handler
from app.api.similarity import router as similarity_router
from app.api.classification import router as classification_router
from app.api.mistral_endpoint import mistral_router


# Функція для завантаження моделі та токенізатора, якщо їх ще немає
def download_mistral_model_and_tokenizer():
    model_dir = os.path.expanduser("~") + "/mistral_7b_instruct_v3"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print("Завантаження моделі Mistral...")
        os.system("wget https://models.mistralcdn.com/mistral-7b-v0-3/mistral-7B-Instruct-v0.3.tar")
        print("Розпаковка моделі...")
        os.system(f"tar -xf mistral-7B-Instruct-v0.3.tar -C {model_dir}")
        print("Модель Mistral успішно завантажена і розпакована.")


# Завантаження моделі та токенізатора, якщо їх ще немає
download_mistral_model_and_tokenizer()

# Завантаження токенізатора для Mistral
mistral_tokenizer = MistralTokenizer.from_file(os.path.expanduser("~") + "/mistral_7b_instruct_v3/tokenizer.model.v3")

# Завантаження моделі Mistral на CPU
device = torch.device("cpu")
model = Transformer.from_folder(os.path.expanduser("~") + "/mistral_7b_instruct_v3").to(device)

app = FastAPI(title=settings.PROJECT_NAME)

app.include_router(heartbeat_router)
app.include_router(api_router, prefix=settings.API_V1_STR, tags=["ML API"])
app.include_router(mistral_router, prefix="/mistral", tags=["Mistral API"])
app.include_router(similarity_router)
app.include_router(classification_router, prefix="/classification", tags=["Classification"])

app.add_event_handler("startup", start_app_handler(app, settings.MODEL_PATH))
app.add_event_handler("shutdown", stop_app_handler(app))

if __name__ == "__main__":
    # Use this for debugging purposes only
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
