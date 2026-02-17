from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

# ============== Health Schemas ==============

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp",
    )
    version: str = Field(..., description="Application version")


class ReadinessResponse(BaseModel):
    status: str = Field(..., description="Service status")
    qdrant_connected: bool = Field(..., description="Qdrant connection status")
    collection_info: dict = Field(..., description="Collection information")


# ============== Document Schemas ==============

class DocumentUploadResponse(BaseModel):
    message: str = Field(..., description="Status message")
    filename: str = Field(..., description="Uploaded filename")
    chunks_created: int = Field(..., description="Number of chunks created")
    document_ids: list[str] = Field(..., description="List of document IDs")


class DocumentInfo(BaseModel):
    source: str = Field(..., description="Document source/filename")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Document metadata",
    )


class DocumentListResponse(BaseModel):

    collection_name: str = Field(..., description="Collection name")
    total_documents: int = Field(..., description="Total document count")
    status: str = Field(..., description="Collection status")


# ============== Query Schemas ==============

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="Question to ask",
        min_length=1,
        max_length=1000,
    )
    include_sources: bool = Field(
        default=True,
        description="Include source documents in response",
    )
    enable_evaluation: bool = Field(
        default=False,
        description="Enable RAGAS evaluation (faithfulness, answer relevancy)",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "question": "What is RAG?",
                    "include_sources": True,
                    "enable_evaluation": False,
                }
            ]
        }
    }


class SourceDocument(BaseModel):
    content: str = Field(..., description="Document content excerpt")
    metadata: dict[str, Any] = Field(..., description="Document metadata")


class EvaluationScores(BaseModel):
    faithfulness: float | None = Field(
        None,
        description="Faithfulness score (0-1): measures factual consistency with sources",
        ge=0.0,
        le=1.0,
    )
    answer_relevancy: float | None = Field(
        None,
        description="Answer relevancy score (0-1): measures relevance to question",
        ge=0.0,
        le=1.0,
    )
    evaluation_time_ms: float | None = Field(
        None,
        description="Time taken for evaluation in milliseconds",
    )
    error: str | None = Field(
        None,
        description="Error message if evaluation failed",
    )


class QueryResponse(BaseModel):
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    sources: list[SourceDocument] | None = Field(
        None,
        description="Source documents used",
    )
    processing_time_ms: float = Field(
        ...,
        description="Query processing time in milliseconds",
    )
    evaluation: EvaluationScores | None = Field(
        None,
        description="RAGAS evaluation scores (if requested)",
    )


# ============== Error Schemas ==============

class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    detail: str | None = Field(None, description="Detailed error information")


class ValidationErrorResponse(BaseModel):
    error: str = Field(default="Validation Error", description="Error type")
    message: str = Field(..., description="Error message")
    errors: list[dict] = Field(..., description="Validation errors")