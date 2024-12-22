from pathlib import Path
import time
from typing import Optional, List, Dict
from chromadb.config import Settings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
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
            print(e)
            raise

    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        score_threshold: float = 0.5,
        filter_metadata: Optional[Dict] = None
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search and return documents with relevance scores

        Args:
            query: The search query
            k: Number of results to return
            score_threshold: Minimum similarity score threshold
            filter_metadata: Optional metadata filters

        Returns:
            List[tuple[Document, float]]: List of documents and their similarity scores
        """
        try:
            if not self.is_initialized:
                self.initialize()

            if not self.vector_store:
                raise

            # Get documents with scores
            if filter_metadata:
                docs_and_scores = self.vector_store.similarity_search_with_score(
                    query,
                    k=k,
                    filter=filter_metadata
                )
            else:
                docs_and_scores = self.vector_store.similarity_search_with_score(
                    query,
                    k=k
                )

            # Filter by score threshold and return
            filtered_results = [
                (doc, score)
                for doc, score in docs_and_scores
                if score >= score_threshold
            ]

            return filtered_results

        except Exception as e:
            print(e)
            raise

    async def get_relevant_documents(
        self,
        query: str,
        k: int = 4,
        filter_metadata: Optional[Dict] = None,
        use_scoring: bool = False,
        score_threshold: float = 0.5
    ) -> List[Document]:
        """
        High-level method to get relevant documents based on query

        Args:
            query: The search query
            k: Number of results to return
            filter_metadata: Optional metadata filters
            use_scoring: Whether to use similarity scoring
            score_threshold: Minimum similarity score threshold

        Returns:
            List[Document]: List of relevant documents
        """
        try:
            if use_scoring:
                docs_and_scores = await self.similarity_search_with_score(
                    query,
                    k=k,
                    score_threshold=score_threshold,
                    filter_metadata=filter_metadata
                )
                return [doc for doc, _ in docs_and_scores]
            else:
                return await self.similarity_search(
                    query,
                    k=k,
                    filter_metadata=filter_metadata
                )

        except Exception as e:
            print(e)
            raise

