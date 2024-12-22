from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from pathlib import Path
from src.config import settings
from src.services.indexer import Indexer
from src.utils.dependency import get_indexer
from src.utils.process_file import process_file
import tempfile
import uuid
import time

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload")
async def upload_document(
        file: UploadFile = File(...),
        # Depends(get_indexer) tells FastAPI to inject the Indexer instance
        indexer: Indexer = Depends(get_indexer)
):
    """
    Upload and process the file

    Args:
        file (UploadFile): the file to be uploaded and processed

    Returns:
        dict: A dictionary containig processed details
    """
    try:
        # Verify the uploaded file
        if file.filename is None:
            raise HTTPException(
                status_code=400,
                detail="Filename cannot be empty"
            )

        file_extension = Path(str(file.filename)).suffix.lower()
        if file_extension not in settings.SUPPORTED_FILE_TYPE:
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
            documents = process_file(tmp_file_path, file_extension)

            if not indexer.is_initialized:
                indexer.initialize()

            if indexer.text_splitter is None:
                raise RuntimeError("Text spliter is not initialized properly.")

            if indexer.vector_store is None:
                raise RuntimeError("Vector Store is not initialized properly.")

            # split document into chunks
            chunks = indexer.text_splitter.split_documents(documents)

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

            return {
                "status": "success",
                "message": f"File {file.filename} processed and indexed sucessfully.",
                "file_id": file_id,
                "chunks_created": len(chunks),
                "chunks": chunks
            }

        finally:
            if tmp_file_path and Path(tmp_file_path).exists():
                Path(tmp_file_path).unlink()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process file: {str(e)}"
        )
