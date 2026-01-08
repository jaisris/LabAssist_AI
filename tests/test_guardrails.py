"""
Unit tests for guardrails.
"""
import pytest
from langchain_core.documents import Document

from app.rag.guardrails import Guardrails


@pytest.fixture
def guardrails():
    """Create guardrails instance."""
    return Guardrails(min_relevance_score=0.5)


@pytest.fixture
def sample_documents():
    """Create sample documents."""
    return [
        Document(
            page_content="Test content",
            metadata={"similarity_score": 0.8, "source": "test"}
        ),
        Document(
            page_content="Another test",
            metadata={"similarity_score": 0.6, "source": "test"}
        )
    ]


def test_check_relevance_high_score(guardrails, sample_documents):
    """Test relevance check with high similarity scores."""
    result = guardrails.check_relevance("test query", sample_documents)
    
    assert result["is_relevant"] is True
    assert result["max_similarity"] >= 0.5
    assert result["num_docs"] == 2


def test_check_relevance_low_score(guardrails):
    """Test relevance check with low similarity scores."""
    low_score_docs = [
        Document(
            page_content="Test",
            metadata={"similarity_score": 0.3}
        )
    ]
    
    result = guardrails.check_relevance("test query", low_score_docs)
    
    assert result["is_relevant"] is False
    assert result["max_similarity"] < 0.5


def test_check_ambiguous_query(guardrails):
    """Test ambiguous query detection."""
    # Very short query
    result = guardrails.check_ambiguous_query("what")
    assert result["is_ambiguous"] is True
    
    # Normal query
    result = guardrails.check_ambiguous_query("What is normal cholesterol?")
    assert result["is_ambiguous"] is False


def test_check_medical_emergency(guardrails):
    """Test medical emergency keyword detection."""
    # Emergency keyword
    result = guardrails.check_medical_emergency("I have severe chest pain")
    assert result["potential_emergency"] is True
    
    # Normal query
    result = guardrails.check_medical_emergency("What is cholesterol?")
    assert result["potential_emergency"] is False


def test_add_source_attribution(guardrails, sample_documents):
    """Test source attribution."""
    answer = "This is a test answer."
    result = guardrails.add_source_attribution(answer, sample_documents)
    
    assert "Sources:" in result
    assert "test" in result


def test_validate_response(guardrails):
    """Test response validation."""
    # Good answer
    good_answer = "This is a comprehensive answer with enough detail to be useful."
    result = guardrails.validate_response(good_answer)
    
    assert result["is_valid"] is True
    assert result["length"] > 0
    
    # Short answer
    short_answer = "Yes."
    result = guardrails.validate_response(short_answer)
    
    assert len(result["issues"]) > 0
