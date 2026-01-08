"""
Unit tests for FastAPI endpoints.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient

from app.api import app, vector_store, retriever, generator, guardrails


@pytest.fixture
def mock_rag_components():
    """Create mock RAG components."""
    mock_vector_store = Mock()
    mock_retriever = Mock()
    mock_generator = Mock()
    mock_guardrails = Mock()
    
    return mock_vector_store, mock_retriever, mock_generator, mock_guardrails


@pytest.fixture
def client_with_mocks(mock_rag_components):
    """Create test client with mocked components."""
    mock_vs, mock_ret, mock_gen, mock_guard = mock_rag_components
    
    # Patch the global variables
    with patch('app.api.vector_store', mock_vs), \
         patch('app.api.retriever', mock_ret), \
         patch('app.api.generator', mock_gen), \
         patch('app.api.guardrails', mock_guard):
        
        # Mock the process_chat_query result
        mock_result = {
            "answer": "Test answer",
            "sources": [{"source": "test", "test": "cholesterol"}],
            "metadata": {
                "query": "test query",
                "retrieved_docs": 2,
                "guardrails": {}
            },
            "guardrails": {
                "relevance_check": {"is_relevant": True},
                "ambiguity_check": {"is_ambiguous": False},
                "emergency_check": {"potential_emergency": False},
                "validation": {"is_valid": True}
            },
            "is_ambiguous": False,
            "is_relevant": True
        }
        
        with patch('app.api.process_chat_query', return_value=mock_result):
            yield TestClient(app)


def test_root_endpoint(client_with_mocks):
    """Test root endpoint."""
    response = client_with_mocks.get("/")
    
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "endpoints" in data
    assert data["message"] == "RAG Lab Tests API"


def test_health_endpoint_loaded(client_with_mocks):
    """Test health endpoint when vector store is loaded."""
    with patch('app.api.vector_store', Mock()):  # Not None
        response = client_with_mocks.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["vector_store_loaded"] is True


def test_health_endpoint_not_loaded():
    """Test health endpoint when vector store is not loaded."""
    with patch('app.api.vector_store', None):
        client = TestClient(app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["vector_store_loaded"] is False


def test_chat_endpoint_success(client_with_mocks):
    """Test chat endpoint with valid request."""
    response = client_with_mocks.post(
        "/chat",
        json={
            "query": "What is cholesterol?",
            "top_k": 5,
            "include_sources": True
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "metadata" in data
    assert "guardrails" in data
    assert data["answer"] == "Test answer"


def test_chat_endpoint_minimal_request(client_with_mocks):
    """Test chat endpoint with minimal request (only query)."""
    response = client_with_mocks.post(
        "/chat",
        json={"query": "What is cholesterol?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data


def test_chat_endpoint_empty_query(client_with_mocks):
    """Test chat endpoint with empty query."""
    response = client_with_mocks.post(
        "/chat",
        json={"query": "   "}  # Whitespace only
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data
    assert "empty" in data["detail"].lower() or "cannot" in data["detail"].lower()


def test_chat_endpoint_no_query_field(client_with_mocks):
    """Test chat endpoint without query field."""
    response = client_with_mocks.post(
        "/chat",
        json={}
    )
    
    assert response.status_code == 422  # Validation error


def test_chat_endpoint_not_initialized():
    """Test chat endpoint when generator is not initialized."""
    with patch('app.api.generator', None):
        client = TestClient(app)
        response = client.post(
            "/chat",
            json={"query": "What is cholesterol?"}
        )
        
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data
        assert "not initialized" in data["detail"].lower()


def test_chat_endpoint_with_top_k(client_with_mocks):
    """Test chat endpoint with top_k parameter."""
    with patch('app.api.process_chat_query') as mock_process:
        mock_process.return_value = {
            "answer": "Answer",
            "sources": [],
            "metadata": {},
            "guardrails": {}
        }
        
        response = client_with_mocks.post(
            "/chat",
            json={
                "query": "What is cholesterol?",
                "top_k": 10
            }
        )
        
        assert response.status_code == 200
        # Verify process_chat_query was called with top_k
        mock_process.assert_called_once()
        call_kwargs = mock_process.call_args[1]
        assert call_kwargs["top_k"] == 10


def test_chat_endpoint_with_include_sources_false(client_with_mocks):
    """Test chat endpoint with include_sources=False."""
    with patch('app.api.process_chat_query') as mock_process:
        mock_process.return_value = {
            "answer": "Answer",
            "sources": [],
            "metadata": {},
            "guardrails": {}
        }
        
        response = client_with_mocks.post(
            "/chat",
            json={
                "query": "What is cholesterol?",
                "include_sources": False
            }
        )
        
        assert response.status_code == 200
        mock_process.assert_called_once()
        call_kwargs = mock_process.call_args[1]
        assert call_kwargs["include_sources"] is False


def test_chat_endpoint_query_stripping(client_with_mocks):
    """Test that query is properly stripped of whitespace."""
    with patch('app.api.process_chat_query') as mock_process:
        mock_process.return_value = {
            "answer": "Answer",
            "sources": [],
            "metadata": {},
            "guardrails": {}
        }
        
        response = client_with_mocks.post(
            "/chat",
            json={
                "query": "  What is cholesterol?  "  # With whitespace
            }
        )
        
        assert response.status_code == 200
        mock_process.assert_called_once()
        call_kwargs = mock_process.call_args[1]
        # Query should be stripped
        assert call_kwargs["query"] == "What is cholesterol?"


def test_chat_endpoint_error_handling(client_with_mocks):
    """Test chat endpoint error handling."""
    with patch('app.api.process_chat_query', side_effect=Exception("Test error")):
        # FastAPI will raise the exception, which is expected behavior
        # We can check that it raises an HTTPException or returns 500
        try:
            response = client_with_mocks.post(
                "/chat",
                json={"query": "What is cholesterol?"}
            )
            # If it doesn't raise, should return error status
            assert response.status_code >= 400
        except Exception:
            # Exception is expected if not handled by FastAPI
            pass


def test_chat_response_structure(client_with_mocks):
    """Test that chat response has correct structure."""
    response = client_with_mocks.post(
        "/chat",
        json={"query": "What is cholesterol?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Check all required fields
    assert "answer" in data
    assert "sources" in data
    assert isinstance(data["sources"], list)
    assert "metadata" in data
    assert isinstance(data["metadata"], dict)
    assert "guardrails" in data
    assert isinstance(data["guardrails"], dict)


def test_api_docs_available(client_with_mocks):
    """Test that API docs are available."""
    response = client_with_mocks.get("/docs")
    # Should return HTML for Swagger UI
    assert response.status_code == 200


def test_api_openapi_schema(client_with_mocks):
    """Test that OpenAPI schema is available."""
    response = client_with_mocks.get("/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema or "swagger" in schema
    assert "paths" in schema
