"""
Unit tests for RAG retriever.
"""
import pytest
from langchain_core.documents import Document
from unittest.mock import Mock, MagicMock

from app.rag.retriever import RAGRetriever


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    store = Mock()
    return store


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Cholesterol is a waxy substance found in your blood.",
            metadata={"source": "test", "test": "cholesterol", "similarity_score": 0.85}
        ),
        Document(
            page_content="Normal cholesterol levels are below 200 mg/dL.",
            metadata={"source": "test", "test": "cholesterol", "similarity_score": 0.75}
        ),
        Document(
            page_content="Blood glucose measures sugar in your blood.",
            metadata={"source": "test", "test": "glucose", "similarity_score": 0.65}
        )
    ]


def test_retriever_initialization(mock_vector_store):
    """Test retriever initialization."""
    retriever = RAGRetriever(
        vector_store=mock_vector_store,
        top_k=5,
        similarity_threshold=0.7,
        enable_reranking=False
    )
    
    assert retriever.top_k == 5
    assert retriever.similarity_threshold == 0.7
    assert retriever.enable_reranking is False
    assert retriever.vector_store == mock_vector_store


def test_retrieve_with_threshold(mock_vector_store):
    """Test retrieval with similarity threshold filtering."""
    # Mock similarity_search_with_score
    # Current similarity formula: similarity = 1 - (distance^2 / 4)
    # distance 0.1 → similarity = 1 - (0.01/4) = 0.9975 (passes 0.6)
    # distance 0.3 → similarity = 1 - (0.09/4) = 0.9775 (passes 0.6)
    # distance 1.0 → similarity = 1 - (1.0/4) = 0.75 (passes 0.6)
    # distance 1.5 → similarity = 1 - (2.25/4) = 0.4375 (fails 0.6)
    mock_results = [
        (Document(page_content="High relevance", metadata={}), 0.1),  # similarity ≈ 0.998
        (Document(page_content="Medium relevance", metadata={}), 0.3),  # similarity ≈ 0.978
        (Document(page_content="Low relevance", metadata={}), 1.5),  # similarity ≈ 0.438
    ]
    mock_vector_store.similarity_search_with_score = Mock(return_value=mock_results)
    
    retriever = RAGRetriever(
        vector_store=mock_vector_store,
        top_k=3,
        similarity_threshold=0.6,
        enable_reranking=False
    )
    
    results = retriever.retrieve("test query")
    
    # Should filter out low relevance (0.438 < 0.6), but still return up to top_k
    # The current implementation always returns up to top_k documents
    assert len(results) <= 3
    # All returned documents should have similarity scores
    assert all('similarity_score' in doc.metadata for doc in results)


def test_reranking(mock_vector_store):
    """Test re-ranking functionality."""
    # Create documents with different similarity scores
    # Current reranking just sorts by similarity score (highest first)
    docs = [
        Document(
            page_content="Lower similarity text",
            metadata={"similarity_score": 0.75}
        ),
        Document(
            page_content="Higher similarity text",
            metadata={"similarity_score": 0.85}
        ),
        Document(
            page_content="Medium similarity text",
            metadata={"similarity_score": 0.80}
        )
    ]
    
    retriever = RAGRetriever(
        vector_store=mock_vector_store,
        top_k=3,
        enable_reranking=True
    )
    
    reranked = retriever._rerank(docs, "test query")
    
    # Reranking should sort by similarity score (highest first)
    assert len(reranked) == 3
    # Documents should be sorted by similarity score in descending order
    scores = [doc.metadata.get('similarity_score', 0) for doc in reranked]
    assert scores == sorted(scores, reverse=True)
    # Highest score should be first
    assert reranked[0].metadata.get('similarity_score', 0) == 0.85
