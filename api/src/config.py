from typing import Dict, Type
from langchain_core.document_loaders.base import BaseLoader
from pydantic_settings import BaseSettings
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


class Settings(BaseSettings):
    """
    Application settings class using Pydantic BaseSettings.
    This class handles configuration for the entire application,
    automatically reading from environment variables if available.
    """

    # Basic Application Settings
    APP_NAME: str = "RAG BOT"
    APP_VERSION: str = "0.0.1"
    PROJECT_ROOT: Path = Path(__file__).parent.parent  # Root directory of the project

    # API Server Settings
    API_HOST: str = "0.0.0.0"  # Host address for the API server (0.0.0.0 allows external access)
    API_PORT: int = 8000  # Port number for the API server

    # Embedding model Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-V2"

    # Database Settings
    DB_NAME: str = "rag.db"

    # Chroma Settings
    PERSIST_DIR: Path = PROJECT_ROOT / "data/chroma-db"
    MODEL_CACHE: Path = PROJECT_ROOT / "cache"

    # Supported file type
    SUPPORTED_FILE_TYPE: Dict[str, Type[BaseLoader]] = {
        ".pdf": PyPDFLoader
    }

    class Config:
        """
        Pydantic configuration class
        - case_sensitive: Determines if environment variable names are case-sensitive
        - env_file: Optional .env file to load environment variables from
        """
        case_sensitive = True
        env_file = ".env"


# Create a single instance of Settings to be imported by other modules
settings = Settings()
