from pathlib import Path
from typing import Optional, List, Dict
from chromadb.config import Settings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from src.utils.logger import logger
from src.utils.logger import log_time
from src.config import settings


class Indexer:
    """
    Handles document indexing operations using Langchain components
    Manage text splitting, embedding and vector store operations
    """

    def __init__(self):
        self.vector_store: Optional[Chroma] = None
        self.embedding_model: Optional[OllamaEmbeddings] = None
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

    @log_time
    def _initialize_text_splitter(self):
        """Initialize text splitter component"""

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    @log_time
    def _initialze_vector_store(self):
        """Initialize vector store component"""
        # Initialze chroma vector store
        # https://python.langchain.com/docs/integrations/vectorstores/chroma/

        # Create persist dir if not present
        Path(settings.PERSIST_DIR).mkdir(parents=True, exist_ok=True)

        self.vector_store = Chroma(
            persist_directory=str(settings.PERSIST_DIR),
            embedding_function=self.embedding_model,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True)
        )

    @log_time
    def _initialize_embedding_model(self):
        """Initialize vector store component"""
        # Initialze embedding model
        # https://python.langchain.com/docs/integrations/vectorstores/chroma/
        self.embedding_model = OllamaEmbeddings(
            model=settings.EMBEDDING_MODEL,
        )

    @log_time
    async def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[Dict] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store

        Args:
            query: The search query
            k: Number of results to return
            filter_metadata: Optional metadata filters

        Returns:
            List[Document]: List of similar documents
        """
        try:
            if not self.is_initialized:
                self.initialize()

            if not self.vector_store:
                raise

            # If filter_metadata is provided, use filtered search
            if filter_metadata:
                similar_docs = self.vector_store.similarity_search(
                    query,
                    k=k,
                    filter=filter_metadata
                )
            else:
                similar_docs = self.vector_store.similarity_search(
                    query,
                    k=k
                )

            return similar_docs

        except Exception as e:
            logger.error(f"Error while searching documents in vector store: {str(e)}")
            raise
