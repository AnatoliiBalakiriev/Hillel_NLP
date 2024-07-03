from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import textdistance

router = APIRouter()


class SimilarityRequest(BaseModel):
    method: str
    line1: str
    line2: str


class SimilarityResponse(BaseModel):
    method: str
    line1: str
    line2: str
    similarity: float


@router.post("/similarity", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    try:
        method = getattr(textdistance, request.method)
        similarity_value = method.normalized_similarity(request.line1, request.line2)
    except AttributeError:
        raise HTTPException(status_code=400, detail="Invalid method")

    return SimilarityResponse(
        method=request.method,
        line1=request.line1,
        line2=request.line2,
        similarity=similarity_value
    )
