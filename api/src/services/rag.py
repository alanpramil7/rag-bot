from typing import Dict, List
from src.utils.dependency import get_indexer
from src.config import settings
from langchain_ollama import ChatOllama
from langchain.chains import history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from datetime import datetime

class RAGService():
    """Service class for Retrival-Augmented Generation"""

    def __init__(self):
        self.indexer = get_indexer()
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
            ("system", "You are a helpful AI assistant. Use the following context "
             "to answer the user's question accurately and concisely."),
            ("system", "Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

    async def generate_response(
        self,
        question: str,
        chat_history: List[Dict] = None
    ):
        """
        Generate a response using RAG

        Args:
            question: User's question
            chat_history: Previous chat interactions
            model: Optional model override

        Returns:
            Dict: Generated response with metadata
        """
        try:
            start_time = datetime.now()

            chat_history = chat_history or []

            if not self.indexer.is_initialized:
                self.indexer.initialize()

            similar_docs = await self.indexer.similarity_search(question, 5)

            # Create Retrival chain
            retriever_chain = history_aware_retriever.create_history_aware_retriever(
                self.llm,
                self.indexer.vector_store.as_retriever(),
                self.context_prompt
            )

            # Create qa chain
            qa_chain = create_stuff_documents_chain(
                self.llm,
                self.qa_prompt
            )

            # create rag chain
            rag_Chain = create_retrieval_chain(retriever_chain, qa_chain)

            # Generate response
            response = rag_Chain.invoke({
                "input": question,
                "chat_history": chat_history,
            })
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()

            # Format and return response
            return {
                "answer": response.get("answer", "Failed to generate an answer."),
                "sources": [doc.metadata for doc in similar_docs],
                "processing_time": processing_time
            }

        except Exception as chain_error:
            print(chain_error)
            return {
                "answer": "I encountered an error while processing your question.",
                "sources": [],
                "processing_time": (datetime.now() - start_time).total_seconds()
            }

        except Exception as e:
            print(e)
            return {
                "answer": "An error occurred while processing your request.",
                "sources": [],
                "processing_time": 0.0
            }
