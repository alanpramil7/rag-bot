from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """
    Application settings class using Pydantic BaseSettings.
    This class handles configuration for the entire application,
    automatically reading from environment variables if available.
    """

    # Basic Application Settings
    APP_NAME: str = "RAG BOT"  # Name of the application
    APP_VERSION: str = "0.0.1"  # Current version of the application
    PROJECT_ROOT: Path = Path(__file__).parent.parent  # Root directory of the project

    # API Server Settings
    API_HOST: str = "0.0.0.0"  # Host address for the API server (0.0.0.0 allows external access)
    API_PORT: int = 8000  # Port number for the API server

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
