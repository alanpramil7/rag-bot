from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: list
    processing_time: float
    reformulated_question: str
    session_id: str
