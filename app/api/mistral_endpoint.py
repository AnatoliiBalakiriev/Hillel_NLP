from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from mistral_inference.generate import generate
from mistral_common.protocol.instruct.messages import UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest

mistral_router = APIRouter()


class QueryRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 64


@mistral_router.post("/query/")
async def query_mistral(request: QueryRequest):
    try:
        # Імпортуємо токенізатор та модель всередині функції, щоб уникнути циклічної залежності
        from app.main import mistral_tokenizer, model

        # Створення запиту на завершення чату
        completion_request = ChatCompletionRequest(messages=[UserMessage(content=request.prompt)])

        # Кодування запиту
        tokens = mistral_tokenizer.encode_chat_completion(completion_request).tokens

        # Генерація відповіді
        out_tokens, _ = generate([tokens], model, max_tokens=request.max_new_tokens, temperature=0.0,
                                 eos_id=mistral_tokenizer.instruct_tokenizer.tokenizer.eos_id)

        # Декодування відповіді
        result = mistral_tokenizer.instruct_tokenizer.tokenizer.decode(out_tokens[0])

        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
