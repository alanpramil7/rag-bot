from typing import Dict, List
from typing_extensions import Optional
from src.utils.dependency import get_indexer
from src.config import settings
from src.utils.logger import logger
from langchain_ollama import ChatOllama
from langchain.chains import history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.utils.logger import log_time
from datetime import datetime


class RAGService():
    """Service class for Retrival-Augmented Generation"""

    def __init__(self):
        self.indexer = get_indexer()
        self.is_initialized = False
        self._initilaize()

    def _initilaize(self):
        """Initialize RAG components"""

        # Initialize LLM
        self.llm = ChatOllama(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE
        )

        # Setup prompts
        self._setup_prompts()
        self.is_initialized = True

    def _setup_prompts(self):
        """Setup prompt templates"""
        # Prompt for context retrieval
        self.context_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given the chat history and the latest question, "
             "create a standalone question that captures the full context."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Prompt for answer generation
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the following context from document"
             "to answer the user's question accurately and concisely."),
            ("system", "Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

    async def generate_stream_response(
        self,
        question: str,
        file_ids: Optional[List[str]] = None,
        chat_history: List[Dict] = []
    ):
        """
        Generate a streaming response using RAG

        Args:
            question: User's question
            file_id: Optional file id
            chat_history: Previous chat interactions
        """
        start_time = datetime.now()
        try:
            logger.debug(f"Starting RAG pipeline for question: {question}")
            logger.debug(f"File ID: {file_ids}")
            logger.debug(f"Chat history length: {len(chat_history)}")

            if not self.indexer.is_initialized:
                self.indexer.initialize()

            if not hasattr(self.indexer, 'vector_store') or self.indexer.vector_store is None:
                raise ValueError("Vector store not properly initialized")

            retriver = await self._create_retiver(file_ids)

            st = datetime.now()
            retriever_chain = history_aware_retriever.create_history_aware_retriever(
                self.llm,
                retriver,
                self.context_prompt
            )
            logger.debug(f"History aware chain creation time: {(datetime.now() - st).total_seconds()}s")

            st = datetime.now()
            qa_chain = create_stuff_documents_chain(
                self.llm,
                self.qa_prompt,
            )
            rag_chain = create_retrieval_chain(retriever_chain, qa_chain)
            logger.debug(f"Retrival chain creation time: {(datetime.now() - st).total_seconds()}s")

            processing_time = (datetime.now() - start_time).total_seconds()

            # Stream the response
            logger.debug("Stated streaming")
            st = datetime.now()
            chunks_count = 0
            async for chunk in rag_chain.astream({
                "input": question,
                "chat_history": chat_history,
            }):
                # logger.debug("Starting streaming.")
                chunks_count += 1
                if "answer" in chunk:
                    # logger.debug(f"Streaming chunk {chunks_count}, Length: {len(chunk['answer'])}")
                    yield {
                        "answer": chunk["answer"],
                        "processing_time": processing_time,
                        "is_complete": False
                    }

            logger.debug(f"Stream time: {(datetime.now() - st).total_seconds()}s")

            # Send final chunk indicating completion
            yield {
                "answer": "",
                "processing_time": processing_time,
                "is_complete": True
            }

        except Exception as e:
            logger.error(f"Error generating streaming response: {str(e)}")
            yield {
                "answer": "Error while processing your question.",
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "is_complete": True
            }

    @log_time
    async def generate_response(
        self,
        question: str,
        file_ids: Optional[str] = None,
        chat_history: List[Dict] = []
    ):
        """
        Generate a response using RAG

        Args:
            question: User's question
            file_id: Uploaded file id
            chat_history: Previous chat interactions
        """
        start_time = datetime.now()
        try:
            chat_history = chat_history or []

            if not self.indexer.is_initialized:
                self.indexer.initialize()

            # Verify vector store is initialized
            if not hasattr(self.indexer, 'vector_store') or self.indexer.vector_store is None:
                raise ValueError("Vector store not properly initialized")

            retriver = await self._create_retiver(file_ids)

            # Create Retrieval chain
            retriever_chain = history_aware_retriever.create_history_aware_retriever(
                self.llm,
                retriver,
                self.context_prompt
            )

            qa_chain = create_stuff_documents_chain(
                self.llm,
                self.qa_prompt,
            )
            rag_chain = create_retrieval_chain(retriever_chain, qa_chain)

            # Generate final response
            response = rag_chain.invoke({
                "input": question,
                "chat_history": chat_history,
            })

            processing_time = (datetime.now() - start_time).total_seconds()

            return {
                "answer": response.get("answer", "Failed to generate an answer."),
                "processing_time": processing_time,
            }

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "Error while processing your question.",
                "sources": [],
                "processing_time": (datetime.now() - start_time).total_seconds()
            }

    @log_time
    async def _create_retiver(self, file_ids: Optional[List[str]] = None):
        search_kwargs = {
            "k": 3
        }
        if file_ids:
            search_kwargs["filter"] = {"file_id": {"$in": file_ids}}

        retriever = self.indexer.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )
        return retriever
