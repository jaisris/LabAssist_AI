"""
RAG system initialization utilities and shared query processing.
"""
from typing import Optional
import time
import logging

from app.ingestion.indexer import load_vector_store
from app.rag.retriever import RAGRetriever
from app.rag.generator import RAGGenerator
from app.rag.guardrails import Guardrails
from app.config import (
    DEFAULT_TOP_K,
    SIMILARITY_THRESHOLD,
    ENABLE_RERANKING,
    LLM_MODEL,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
    MIN_RELEVANCE_SCORE
)

logger = logging.getLogger(__name__)


def initialize_rag_components():
    """
    Initialize all RAG components (vector store, retriever, generator, guardrails).
    
    Returns:
        tuple: (vector_store, retriever, generator, guardrails)
        
    Raises:
        Exception: If vector store cannot be loaded or components cannot be initialized
    """
    try:
        vector_store = load_vector_store()
    except Exception as e:
        raise Exception(f"Failed to load vector store: {e}. Run ingestion first: python -m app.main --ingest") from e
    
    try:
        retriever = RAGRetriever(
            vector_store=vector_store,
            top_k=DEFAULT_TOP_K,
            similarity_threshold=SIMILARITY_THRESHOLD,
            enable_reranking=ENABLE_RERANKING
        )
        
        generator = RAGGenerator(
            retriever=retriever,
            model_name=LLM_MODEL,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS
        )
        
        guardrails = Guardrails(min_relevance_score=MIN_RELEVANCE_SCORE)
    except Exception as e:
        raise Exception(f"Failed to initialize RAG components: {e}") from e
    
    return vector_store, retriever, generator, guardrails


def process_chat_query(
    query: str,
    generator: RAGGenerator,
    guardrails: Guardrails,
    top_k: Optional[int] = None,
    include_sources: bool = True
) -> dict:
    """
    Process a single chat query and return the response.
    Shared logic for both CLI and API.
    
    Args:
        query: User query string
        generator: RAGGenerator instance
        guardrails: Guardrails instance
        top_k: Number of documents to retrieve (optional)
        include_sources: Whether to include source attribution
        
    Returns:
        Dictionary with answer, sources, metadata, and guardrails info
    """
    # Start timing
    start_time = time.perf_counter()
    
    # Guardrails checks
    ambiguity_check = guardrails.check_ambiguous_query(query)
    emergency_check = guardrails.check_medical_emergency(query)
    
    # If query is ambiguous, return early
    if ambiguity_check.get("is_ambiguous"):
        total_time = time.perf_counter() - start_time
        return {
            "answer": guardrails.get_fallback_response(query, {}, ambiguity_check),
            "sources": [],
            "metadata": {
                "query": query,
                "reason": "ambiguous_query",
                "total_time_seconds": total_time
            },
            "guardrails": {
                "ambiguity_check": ambiguity_check,
                "emergency_check": emergency_check
            },
            "is_ambiguous": True,
            "is_relevant": False
        }
    
    # Generate response
    result = generator.generate(
        query=query,
        top_k=top_k,
        include_sources=include_sources
    )
    
    # Get context docs from result (already retrieved in generator)
    context_docs = result.get("context_docs", [])
    
    # Check relevance
    relevance_check = guardrails.check_relevance(query, context_docs)
    
    # Apply guardrails
    if not relevance_check.get("is_relevant"):
        result["answer"] = guardrails.get_fallback_response(
            query, relevance_check, ambiguity_check
        )
    else:
        # Add source attribution if relevant
        if include_sources and context_docs:
            result["answer"] = guardrails.add_source_attribution(
                result["answer"],
                context_docs
            )
    
    # Validate response
    validation = guardrails.validate_response(result["answer"])
    
    # Calculate total time
    total_time = time.perf_counter() - start_time
    
    # Add guardrails info to metadata
    result["metadata"]["guardrails"] = {
        "relevance_check": relevance_check,
        "ambiguity_check": ambiguity_check,
        "emergency_check": emergency_check,
        "validation": validation
    }
    
    # Add total time to metadata
    result["metadata"]["total_time_seconds"] = total_time
    
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "metadata": result["metadata"],
        "guardrails": result["metadata"]["guardrails"],
        "is_ambiguous": False,
        "is_relevant": relevance_check.get("is_relevant", False)
    }
