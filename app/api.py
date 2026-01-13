from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from contextlib import asynccontextmanager
import os
import logging
from dotenv import load_dotenv

from app.rag import initialize_rag_components, process_chat_query

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize components
vector_store = None
retriever = None
generator = None
guardrails = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global vector_store, retriever, generator, guardrails
    
    # Startup: Initialize RAG components
    logger.info("Initializing RAG components...")
    try:
        vector_store, retriever, generator, guardrails = initialize_rag_components()
        logger.info("RAG components initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing components: {e}", exc_info=True)
        raise
    
    yield  # Application runs here
    
    # Shutdown: Cleanup (if needed)
    logger.info("Shutting down API server...")
    # Currently no cleanup needed, but can be added here


app = FastAPI(
    title="RAG Lab Tests API",
    description="Conversational AI assistant for laboratory test information",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = None
    include_sources: bool = True


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]
    metadata: dict
    guardrails: dict


class HealthResponse(BaseModel):
    status: str
    vector_store_loaded: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "vector_store_loaded": vector_store is not None
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for asking questions.
    
    Args:
        request: Chat request with query
        
    Returns:
        Chat response with answer, sources, and metadata
    """
    if not generator:
        logger.error("RAG system not initialized")
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    
    query = request.query.strip()
    
    if not query:
        logger.warning("Empty query received")
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    
    logger.info(f"Processing chat request: query='{query[:100]}...', top_k={request.top_k}")
    
    # Process query using shared function
    try:
        result = process_chat_query(
            query=query,
            generator=generator,
            guardrails=guardrails,
            top_k=request.top_k,
            include_sources=request.include_sources
        )
        
        logger.info(
            f"Chat request completed: retrieved_docs={result['metadata'].get('retrieved_docs', 0)}, "
            f"is_relevant={result.get('is_relevant', False)}"
        )
        
        return ChatResponse(
            answer=result["answer"],
            sources=result["sources"],
            metadata=result["metadata"],
            guardrails=result["guardrails"]
        )
    except Exception as e:
        logger.error(f"Error processing chat request: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "RAG Lab Tests API",
        "endpoints": {
            "health": "/health",
            "chat": "/chat (POST)",
            "docs": "/docs"
        }
    }
