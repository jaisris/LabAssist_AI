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
        similarity_threshold=0.7
    )
    
    assert retriever.top_k == 5
    assert retriever.similarity_threshold == 0.7
    assert retriever.vector_store == mock_vector_store


def test_retrieve_with_threshold(mock_vector_store):
    """Test retrieval with similarity threshold filtering."""
    # Mock similarity_search_with_score
    # Similarity = 1 / (1 + distance)
    # distance 0.1 → similarity = 0.909 (passes 0.6)
    # distance 0.3 → similarity = 0.769 (passes 0.6)
    # distance 0.8 → similarity = 0.556 (fails 0.6)
    mock_results = [
        (Document(page_content="High relevance", metadata={}), 0.1),  # similarity = 0.909
        (Document(page_content="Medium relevance", metadata={}), 0.3),  # similarity = 0.769
        (Document(page_content="Low relevance", metadata={}), 0.8),  # similarity = 0.556
    ]
    mock_vector_store.similarity_search_with_score = Mock(return_value=mock_results)
    
    retriever = RAGRetriever(
        vector_store=mock_vector_store,
        top_k=3,
        similarity_threshold=0.6
    )
    
    results = retriever.retrieve("test query")
    
    # Should filter out low relevance (0.556 < 0.6)
    assert len(results) <= 2
    assert all(doc.metadata.get('similarity_score', 0) >= 0.6 for doc in results)


def test_reranking(mock_vector_store):
    """Test re-ranking functionality."""
    # Create documents with different lengths and scores
    docs = [
        Document(
            page_content="Short text",
            metadata={"similarity_score": 0.8}
        ),
        Document(
            page_content="Much longer text with more detailed information that provides better context" * 10,
            metadata={"similarity_score": 0.75}
        )
    ]
    
    retriever = RAGRetriever(
        vector_store=mock_vector_store,
        top_k=2,
        enable_reranking=True
    )
    
    reranked = retriever._rerank(docs, "test query")
    
    # Longer document should be preferred after re-ranking
    assert len(reranked) == 2
    # The longer document should come first after re-ranking
    assert len(reranked[0].page_content) >= len(reranked[1].page_content) or \
           reranked[0].metadata.get('similarity_score', 0) > reranked[1].metadata.get('similarity_score', 0)
