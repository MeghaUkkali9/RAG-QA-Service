from dotenv import load_dotenv
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app import __version__
from app.api.routes import documents, health, query
from app.config import get_settings
from app.utils.logger import get_logger, setup_logging

load_dotenv()
settings = get_settings()

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(settings.log_level)

    logger = get_logger(__name__)
    logger.info(f"Starting {settings.app_name} v{__version__}")
    logger.info(f"Log level: {settings.log_level}")

    yield
    
    logger.info("Shutting down application")

app = FastAPI(
    title=settings.app_name,
    description="""
                    ## RAG Q&A Service

                    A Retrieval-Augmented Generation (RAG) question-answering system built with:
                    - **FastAPI** for the API layer
                    - **LangChain** for RAG orchestration
                    - **Qdrant Cloud** for vector storage
                    - **OpenAI** for embeddings and LLM

                    ### Features
                    - Upload PDF, TXT, and CSV documents
                    - Ask questions and get AI-powered answers
                    - View source documents for transparency
                    - Streaming responses for real-time feedback
                        """,
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(documents.router)
app.include_router(query.router)

@app.get("/", tags=["Root"])
async def root():
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": __version__,
        "docs": "/docs",
    }

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger = get_logger(__name__)
    logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
        },
    )

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )