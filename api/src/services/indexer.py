from pathlib import Path
import time
from typing import Optional
from chromadb.config import Settings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.utils.logger import logger
from src.config import settings


class Indexer:
    """
    Handles document indexing operations using Langchain components
    Manage text splitting, embedding and vector store operations
    """

    def __init__(self):
        self.vector_store: Optional[Chroma] = None
        self.embedding_model: Optional[HuggingFaceEmbeddings] = None
        self.text_splitter: Optional[RecursiveCharacterTextSplitter] = None
        self.is_initialized: bool = False
        self.initialize()

    def initialize(self) -> None:
        """
        Initialize text spliter, embedding model, and vector store
        """
        if self.is_initialized:
            logger.debug("Indexer already initilaized")
            return

        try:
            self._initialize_text_splitter()
            self._initialize_embedding_model()
            self._initialze_vector_store()
            self.is_initialized = True

        except Exception as e:
            logger.error("Error initilaizing ")
            print(e)

    def _initialize_text_splitter(self):
        """Initialize text splitter component"""

        start_time = time.time()
       # https://dev.to/eteimz/understanding-langchains-recursivecharactertextsplitter-2846
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info(f"Text spliter initialized in {time.time() - start_time:.2f} seconds.")

    def _initialze_vector_store(self):
        """Initialize vector store component"""
        # Initialze chroma vector store
        # https://python.langchain.com/docs/integrations/vectorstores/chroma/

        # Create persist dir if not present
        Path(settings.PERSIST_DIR).mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        self.vector_store = Chroma(
            persist_directory=str(settings.PERSIST_DIR),
            embedding_function=self.embedding_model,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True)
        )
        logger.info(f"Vector store initialized in {time.time() - start_time:.2f} seconds.")

    def _initialize_embedding_model(self):
        """Initialize vector store component"""
        # Initialze embedding model
        # https://python.langchain.com/docs/integrations/vectorstores/chroma/
        start_time = time.time()
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            cache_folder=str(settings.MODEL_CACHE)
        )
        logger.info(f"Embedding Model initialized in {time.time() - start_time:.2f} seconds.")
