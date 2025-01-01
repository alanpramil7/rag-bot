from fastapi import APIRouter
from typing import Optional
from pydantic import BaseModel
from src.services.session import SessionService
from src.utils.logger import logger
from src.models.chat import WebsiteUploadResponse
from src.utils.scrape_website import load_website

router = APIRouter(prefix="/website", tags=["website"])
session_service = SessionService()


class WebsiteProcessRequest(BaseModel):
    url: str
    session_id: Optional[str] = None


@router.post("/", response_model=WebsiteUploadResponse)
async def upload_website(request: WebsiteProcessRequest):
    """
    Upload and process website

    Args:
        url: URL to be processed
    """
    try:
        if not request.url:
            logger.error("Please provide URL to process.")

        logger.debug(f"Start processing URL: {request.url}")
        docs = load_website(request.url)
        print(docs)

        return {
            "status": "success",
            "message": f"Url {request.url} has processed succesfully.",
            "file_id": "",
            "chunks_created": 1,
            "session_id": ""
        }

    except Exception as e:
        logger.error(f"Error while processing website: {str(e)}")
