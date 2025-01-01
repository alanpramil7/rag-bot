from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pathlib import Path
from typing import Optional
from src.config import settings
from src.services.indexer import Indexer
from src.utils.dependency import get_indexer
from src.utils.logger import logger
from src.utils.logger import log_time
from src.models.chat import DocumentUploadRersponse
from src.services.session import SessionService
from src.utils.process_file import process_file
import tempfile
import uuid
import time

router = APIRouter(prefix="/documents", tags=["documents"])
session_service = SessionService()


@router.post("/upload", response_model=DocumentUploadRersponse)
@log_time
async def upload_document(
        file: UploadFile = File(...),
        # Depends(get_indexer) tells FastAPI to inject the Indexer instance
        indexer: Indexer = Depends(get_indexer),
        session_id: Optional[str] = None
):
    """
    Upload and process the file

    Args:
        file (UploadFile): the file to be uploaded and processed
    """
    try:

        logger.debug(f"Processing file: {file.filename}")
        # Verify the uploaded file
        if file.filename is None:
            raise HTTPException(
                status_code=400,
                detail="Filename cannot be empty"
            )

        file_extension = Path(str(file.filename)).suffix.lower()
        if file_extension not in settings.SUPPORTED_FILE_TYPE:
            logger.error(f"Unsupported file format: {file_extension}")
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file types. Supported filetypes are: {', '.join(settings.SUPPORTED_FILE_TYPE.keys())}"
            )

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            # Write uploaded contents to tmp file
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name

        try:
            # Process the file content
            logger.debug(f"Extarcting document: {file.filename}")
            documents = process_file(tmp_file_path, file_extension)

            if not indexer.is_initialized:
                indexer.initialize()

            if indexer.text_splitter is None:
                raise RuntimeError("Text spliter is not initialized properly.")

            if indexer.vector_store is None:
                raise RuntimeError("Vector Store is not initialized properly.")

            # split document into chunks
            logger.debug("Splititng documents into chunks.")
            chunks = indexer.text_splitter.split_documents(documents)
            logger.debug(f"Documents splitted into {len(chunks)} chunks.")

            # Generate unique file_id for each files
            file_id = str(uuid.uuid4())
            current_time = time.time()

            # Add metadata to chunks
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "file_id": file_id,  # Same file_id for all chunks from same file
                    "file_name": file.filename,
                    "file_type": file_extension,
                    "upload_timestamp": current_time,
                    "chunk_size": len(chunk.page_content),
                    "chunk_index": i,  # Add index to track chunk order
                    "total_chunks": len(chunks),
                    "source_type": "upload"
                })

            # Add chunks to vector store
            indexer.vector_store.add_documents(chunks)

            if not session_id:
                session_id = session_service.create_session(file_id)
            elif session_service.get_session(session_id):
                logger.debug("Session is already initiated")
                session_service.insert_file_id(session_id, file_id)
                logger.debug(f"New file id {file_id} is added to the session.")

            return {
                "status": "success",
                "message": f"File {file.filename} processed and indexed sucessfully.",
                "file_id": file_id,
                "chunks_created": len(chunks),
                "session_id": session_id
            }

        finally:
            if tmp_file_path and Path(tmp_file_path).exists():
                Path(tmp_file_path).unlink()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file: {str(e)}"
        )
