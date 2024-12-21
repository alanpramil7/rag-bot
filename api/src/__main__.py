from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config import settings
from src.services.indexer import Indexer

indexer = Indexer()


def create_app() -> FastAPI:
    """
    Function to create FastAPI instance

    Returns:
        FastAPI: Configured FastAPI Application
    """
    application = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="RAG chatbot API"
    )

    # Add CORS middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_headers=["*"],
        allow_methods=["*"],
        allow_credentials=True
    )

    # Endpoint to check application health
    @application.get("/check-health")
    def health_check():
        """
        Endpont to check health

        Returns:
            dict: A dictornary containing the application status and version
                {
                    "Status": str,
                    "version": str
                }
        """

        return {
            "Status": "healthy",
            "Version": settings.APP_VERSION
        }

    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )
