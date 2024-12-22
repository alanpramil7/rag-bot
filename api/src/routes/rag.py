from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from src.services.rag import RAGService

router = APIRouter(prefix="/chat", tags=["chat"])
rag_service = RAGService()

class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[list] = []

class ChatResponse(BaseModel):
    answer: str
    sources: list
    processing_time: float


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process chat request and generate response using RAG.

    Args:
        request: The chat request containing question and optional parameters

    Returns:
        ChatResponse: Contains generated response and metadata

    Raises:
        HTTPException: If processing fails
    """
    try:
        response = await rag_service.generate_response(
            question=request.question,
            chat_history=request.chat_history,
        )

        if not response or not isinstance(response, dict):
            raise HTTPException(
                status_code=500,
                detail="Invalid response format from RAG service"
            )

        return ChatResponse(
            answer=response.get("answer", "No answer generated"),
            sources=response.get("sources", []),
            processing_time=response.get("processing_time", 0.0)
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
        )
