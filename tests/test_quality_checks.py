"""
Unit tests for quality checks.
"""
import pytest
from langchain_core.documents import Document

from app.eval.quality_checks import QualityChecker


@pytest.fixture
def quality_checker():
    """Create quality checker instance."""
    return QualityChecker()


@pytest.fixture
def sample_documents():
    """Create sample documents."""
    return [
        Document(
            page_content="Cholesterol is a waxy substance.",
            metadata={"similarity_score": 0.8, "source": "test"}
        ),
        Document(
            page_content="Normal levels are below 200.",
            metadata={"similarity_score": 0.75, "source": "test"}
        )
    ]


def test_evaluate_answer_quality(quality_checker, sample_documents):
    """Test answer quality evaluation."""
    query = "What is cholesterol?"
    answer = "Cholesterol is a waxy substance found in your blood. Normal levels are below 200 mg/dL."
    
    result = quality_checker.evaluate_answer_quality(query, answer, sample_documents)
    
    assert "quality_score" in result
    assert result["has_sources"] is True
    assert result["num_sources"] == 2
    assert result["avg_similarity"] > 0.0


def test_check_answer_completeness(quality_checker):
    """Test answer completeness check."""
    query = "What is cholesterol?"
    
    # Complete answer
    complete_answer = "Cholesterol is a waxy substance found in your blood that your body needs to build healthy cells."
    result = quality_checker.check_answer_completeness(query, complete_answer)
    assert result["is_complete"] is True
    
    # Incomplete answer
    incomplete_answer = "I don't know."
    result = quality_checker.check_answer_completeness(query, incomplete_answer)
    assert result["is_incomplete"] is True


def test_evaluate_retrieval_quality(quality_checker, sample_documents):
    """Test retrieval quality evaluation."""
    query = "What is cholesterol?"
    
    result = quality_checker.evaluate_retrieval_quality(query, sample_documents, top_k=5)
    
    assert result["num_retrieved"] == 2
    assert result["retrieval_rate"] > 0
    assert result["avg_similarity"] > 0
    assert "retrieval_quality_score" in result


def test_run_full_evaluation(quality_checker, sample_documents):
    """Test full evaluation pipeline."""
    query = "What is cholesterol?"
    answer = "Cholesterol is a waxy substance found in your blood."
    
    result = quality_checker.run_full_evaluation(query, answer, sample_documents, top_k=5)
    
    assert "answer_quality" in result
    assert "completeness" in result
    assert "retrieval_quality" in result
    assert "overall_score" in result
    assert 0 <= result["overall_score"] <= 1
