"""
Integration tests for RAG system.
"""
import pytest
import os
from pathlib import Path

# Skip integration tests if vector store doesn't exist
VECTOR_STORE_EXISTS = Path("chroma_db").exists() and any(Path("chroma_db").iterdir())


@pytest.mark.skipif(not VECTOR_STORE_EXISTS, reason="Vector store not initialized")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set")
def test_end_to_end_retrieval():
    """Test end-to-end retrieval flow."""
    from app.ingestion.indexer import load_vector_store
    from app.rag.retriever import RAGRetriever
    
    vector_store = load_vector_store()
    retriever = RAGRetriever(
        vector_store, 
        top_k=3, 
        similarity_threshold=0.5,
        enable_reranking=False
    )
    
    query = "What is normal cholesterol?"
    results = retriever.retrieve(query)
    
    assert len(results) > 0
    assert all(hasattr(doc, 'page_content') for doc in results)


@pytest.mark.skipif(not VECTOR_STORE_EXISTS, reason="Vector store not initialized")
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OpenAI API key not set")
def test_end_to_end_generation():
    """Test end-to-end generation flow."""
    from app.ingestion.indexer import load_vector_store
    from app.rag.retriever import RAGRetriever
    from app.rag.generator import RAGGenerator
    
    vector_store = load_vector_store()
    retriever = RAGRetriever(
        vector_store, 
        top_k=3,
        enable_reranking=False
    )
    generator = RAGGenerator(retriever, model_name="gpt-4o-mini")
    
    query = "What is cholesterol?"
    result = generator.generate(query)
    
    assert "answer" in result
    assert len(result["answer"]) > 0
    assert "sources" in result
    assert "metadata" in result
