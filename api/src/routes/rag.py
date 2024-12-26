from fastapi import APIRouter, HTTPException
from src.services.rag import RAGService
from src.models.chat import ChatRequest, ChatResponse
from src.services.session import SessionService
from src.utils.logger import logger
from src.services.chat import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])
rag_service = RAGService()
session_service = SessionService()
chat_service = ChatService()


@router.post("/history")
async def get_chat_history(session_id: str):
    """
    Get chat history based on session_id

    Args:
        session_id: Seesion id for the chat
    """
    try:
        if not session_id:
            logger.error("No session id proivided for retiving chat hsitory.")
            raise HTTPException(
                status_code=400,
                detail="No session id provoded."
            )

        return chat_service.get_chat_history(session_id)
    except Exception as e:
        logger.error(f"Error while retiving the chat history: {str(e)}")


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process chat request and generate response using RAG.

    Args:
        request: The chat request containing question and optional parameters
    """
    try:
        # Get or create session
        session_id = request.session_id

        if not session_id:
            logger.debug("No session id creating a new seesion.")
            session_id = session_service.create_session()
        elif not session_service.get_session(session_id):
            raise HTTPException(
                status_code=404,
                detail="Session not found"
            )
        logger.debug(f"Session Id: {session_id}")

        # Get chat history
        chat_history = chat_service.get_chat_history(session_id)

        # Get file_id
        file_id = session_service.get_file_id(session_id)[0] if session_service.get_file_id(session_id) else None
        logger.debug(f"Retrieved file_id from session: {file_id}")

        # Save user message
        chat_service.save_message(
            session_id=session_id,
            role="user",
            content=request.question
        )

        response = await rag_service.generate_response(
            question=request.question,
            chat_history=chat_history,
            file_id=file_id
        )

        if not response or not isinstance(response, dict):
            raise HTTPException(
                status_code=500,
                detail="Invalid response format from RAG service"
            )

        # Save assistant response
        chat_service.save_message(
            session_id=session_id,
            role="assistant",
            content=response.get("answer", "No answer generated"),
            metadata=str({
                "processing_time": response.get("processing_time", 0.0),
            })
        )

        return ChatResponse(
            answer=response.get("answer", "No answer generated"),
            processing_time=response.get("processing_time", 0.0),
            session_id=session_id,
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process chat request: {str(e)}"
        )
