import logging
import os
import uuid
import sqlite3

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from chromadb.config import Settings
from langchain_ollama import ChatOllama

# Basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()


# User request model
class Input(BaseModel):
    question: str
    session_id: Optional[str] = Field(default=None)
    model: str = Field(default="llama3.2")


# Document response model
class DocumentInfo(BaseModel):
    id: int
    file_name: str
    uploaded_at: datetime


class Database:
    def __init__(self, db_name="rag.db"):
        self.DB_NAME = db_name
        self._create_tables()

    # Initiate the db connection
    def _get_connection(self):
        conn = sqlite3.connect(self.DB_NAME)
        conn.row_factory = sqlite3.Row
        return conn

    # create tables if not present
    def _create_tables(self):
        with self._get_connection() as conn:
            # Create chat history table
            conn.execute('''
                         CREATE TABLE IF NOT EXISTS history
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         session_id TEXT,
                         user_query TEXT,
                         llm_res TEXT,
                         model TEXT,
                         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
                         ''')

            # create document store
            conn.execute('''
                         CREATE TABLE IF NOT EXISTS document
                         (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         file_name TEXT,
                         uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
                         ''')

    # Function to get the chat history based on session
    def get_chat_history(self, session_id: str):
        with self._get_connection() as conn:
            cursor = conn.cursor()  # object for manipulating db records
            cursor.execute('''SELECT user_query, llm_res FROM history
                           WHERE session_id= ? ORDER BY created_at''',
                           (session_id,))

            messages = []
            # Structure the message
            for row in cursor.fetchall():
                messages.extend([
                    {"role": "human", "content": row['user_query']},
                    {"role": "ai", "content": row['llm_res']}
                ])

            return messages

    # To insert document data in db
    def insert_document(self, file_name):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''INSERT INTO document (file_name) VALUES (?)''',
                           (file_name,))
            file_id = cursor.lastrowid
            return file_id

    # Get all the uploaded document
    def get_document(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''SELECT id, file_name, uploaded_at FROM document
                ORDER BY uploaded_at''')

            documents = cursor.fetchall()
            return [dict(doc) for doc in documents]


class Indexer:
    def __init__(self,
                 embedding_model='sentence-transformers/all-MiniLM-L6-V2',
                 chunk_size=1000,
                 chunk_overlap=200,
                 persist_dir='./chroma-db'):
        # Split the given doc into chunks
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

        # initialize embedding
        self.embedding_fucntion = HuggingFaceEmbeddings(
            model_name=embedding_model,
            # model_kwargs={"device": "cuda"}  # Add this to specify GPU
        )

        # initialize chromadb
        self.vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=self.embedding_fucntion,
            client_settings=Settings(anonymized_telemetry=False, is_persistent=True)
        )

    # function to load and split document
    def split_document(self, file_path) -> List[Document]:
        # Load the documents from tmp path
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        return self.text_splitter.split_documents(documents)

    # Index documments to chromadb
    def index_chroma(self, file_path, file_id):
        try:
            splits = self.split_document(file_path)

            # Add metatdata to splits
            for split in splits:
                split.metadata['file_id'] = file_id

            self.vector_store.add_documents(splits)
            return True

        except Exception as e:
            logger.error(f"Error indexing documents: {e}")
            return False

class RAG:
    def __init__(self):
        self.llm = ChatOllama(model="llama3.2", temperature=0)

        # Prompt to rephrase the question based on history
        self.contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        self.contextual_q_prompt = ChatPromptTemplate.from_messages([
            ("system", self.contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Actual qa prompt
        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Use the following context to answer the "
             "user's question."),
            ("system", "Context: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        self.idx = Indexer()
        self.vector_store = self.idx.vector_store
        self.retriver = self.vector_store.as_retriever(search_kwargs={"k": 3})

    # Create a history aware chain with new question and qa prompt
    def get_rag_chain(self):
        history_aware_retiver = create_history_aware_retriever(
            self.llm, self.retriver, self.contextual_q_prompt)
        question_answer_chain = create_stuff_documents_chain(self.llm, self.qa_prompt)

        rag_chain = create_retrieval_chain(history_aware_retiver, question_answer_chain)
        return rag_chain

# Route to upload document
@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No file provided"
        )

    # Check for allowed extensions
    allowed_extensions = [".pdf"]
    file_extension = os.path.splitext(str(file.filename))[1]

    if file_extension.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {file_extension}"
        )

    tmp_file = f"tmp_{file.filename}"

    try:
        # Create a temp file
        with open(tmp_file, "wb") as buffer:
            buffer.write(await file.read())

        db = Database()
        idx = Indexer()

        file_id = db.insert_document(file.filename)
        success = idx.index_chroma(tmp_file, file_id)
        if success:
            return {
                "message": f"File uploaded succesfully: {file.filename}",
                "file_id": file_id
            }
        else:
            return f"Failed to index file: {file.filename}"

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error processing pdf: {e}"
        )

    finally:
        if os.path.exists(tmp_file):
            os.remove(tmp_file)


# Route to generate response from llm
@app.post("/chat")
def chat(request: Input):
    session_id = request.session_id
    logger.info(f"Seesion ID: {session_id}, User Input: {request.question}")
    db = Database()
    rag = RAG()

    # Generate new session_id if new chat
    if not session_id:
        session_id = str(uuid.uuid4())

    chat_history = db.get_chat_history(str(session_id)) if session_id else []
    rag_chain = rag.get_rag_chain()
    resposne = rag_chain.invoke({
        "input": request.question,
        "chat_history": chat_history
    })['answer']

    # Store the interaction in database
    with db._get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO history (session_id, user_query, llm_res, model)
            VALUES (?, ?, ?, ?)
       ''', (session_id, request.question, resposne, "llama3.2"))

    return {
        "session_id": session_id,
        "response": resposne
    }

# Route to get all document info
@app.get("/list-doc")
def list_document():
    db = Database()
    return db.get_document()
