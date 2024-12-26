from pydantic import BaseModel
from typing import Optional

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    processing_time: float
    session_id: str

class DocumentUploadRersponse(BaseModel):
    status: str
    message: str
    file_id: str
    chunks_created: int
    session_id: str
