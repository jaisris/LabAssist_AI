"""
Unit tests for RAGGenerator.
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from langchain_core.documents import Document

from app.rag.generator import RAGGenerator
from app.rag.retriever import RAGRetriever


@pytest.fixture
def mock_retriever():
    """Create a mock retriever."""
    retriever = Mock(spec=RAGRetriever)
    return retriever


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="Cholesterol is a waxy substance found in your blood.",
            metadata={"source": "MedlinePlus", "test": "cholesterol", "similarity_score": 0.85}
        ),
        Document(
            page_content="Normal cholesterol levels are below 200 mg/dL.",
            metadata={"source": "MedlinePlus", "test": "cholesterol", "similarity_score": 0.75}
        )
    ]


@pytest.fixture
def generator_with_mock_llm(mock_retriever):
    """Create generator with mocked LLM."""
    with patch('app.rag.generator.ChatOpenAI') as mock_chat:
        mock_llm_instance = Mock()
        mock_llm_instance.model_name = "gpt-4o-mini"
        mock_chat.return_value = mock_llm_instance
        
        generator = RAGGenerator(
            retriever=mock_retriever,
            model_name="gpt-4o-mini",
            temperature=0.1,
            max_tokens=1000
        )
        
        # Replace the actual LLM with a mock that has invoke and stream methods
        mock_chain = Mock()
        mock_chain.invoke.return_value = "This is a test answer about cholesterol."
        mock_chain.stream.return_value = ["This ", "is ", "a ", "test ", "answer."]
        
        # Mock the chain creation
        generator._mock_chain = mock_chain
        generator.llm = mock_llm_instance
        
        return generator, mock_chain


def test_generator_initialization(mock_retriever):
    """Test RAGGenerator initialization."""
    with patch('app.rag.generator.ChatOpenAI'):
        generator = RAGGenerator(
            retriever=mock_retriever,
            model_name="gpt-4o-mini",
            temperature=0.2,
            max_tokens=500,
            max_context_tokens=2000
        )
        
        assert generator.retriever == mock_retriever
        assert generator.prompt_builder.max_context_tokens == 2000


def test_generate_with_empty_context(mock_retriever, generator_with_mock_llm):
    """Test generate() when no documents are retrieved."""
    generator, _ = generator_with_mock_llm
    mock_retriever.retrieve.return_value = []
    
    result = generator.generate("What is cholesterol?")
    
    assert result["answer"] == "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your query or consult a healthcare professional."
    assert result["sources"] == []
    assert result["metadata"]["retrieved_docs"] == 0
    assert "model" in result["metadata"]


def test_generate_with_documents(mock_retriever, generator_with_mock_llm, sample_documents):
    """Test generate() with retrieved documents."""
    generator, mock_chain = generator_with_mock_llm
    mock_retriever.retrieve.return_value = sample_documents
    
    # Mock the chain invoke
    with patch.object(generator, 'prompt_builder') as mock_prompt_builder:
        mock_prompt = Mock()
        mock_prompt_builder.build_prompt.return_value = mock_prompt
        mock_prompt_builder.truncate_context.return_value = sample_documents
        mock_prompt_builder.max_context_tokens = 3000
        
        # Create a mock chain that can be invoked
        mock_chain_obj = Mock()
        mock_chain_obj.invoke.return_value = "Cholesterol is a waxy substance found in your blood."
        
        # Mock the chain creation
        with patch('app.rag.generator.RunnablePassthrough') as mock_passthrough:
            # Create a chain-like object
            chain_mock = Mock()
            chain_mock.__or__ = Mock(return_value=chain_mock)
            chain_mock.invoke = Mock(return_value="Cholesterol is a waxy substance found in your blood.")
            
            # Make prompt | llm | parser return our mock
            mock_prompt.__or__ = Mock(return_value=chain_mock)
            
            result = generator.generate("What is cholesterol?")
            
            assert "answer" in result
            assert len(result["answer"]) > 0
            assert "sources" in result
            assert "metadata" in result
            assert result["metadata"]["retrieved_docs"] == len(sample_documents)


def test_generate_source_extraction(mock_retriever, generator_with_mock_llm, sample_documents):
    """Test that sources are correctly extracted from documents."""
    generator, _ = generator_with_mock_llm
    mock_retriever.retrieve.return_value = sample_documents
    
    with patch.object(generator, 'prompt_builder') as mock_prompt_builder:
        mock_prompt = Mock()
        mock_prompt_builder.build_prompt.return_value = mock_prompt
        mock_prompt_builder.truncate_context.return_value = sample_documents
        mock_prompt_builder.max_context_tokens = 3000
        
        chain_mock = Mock()
        chain_mock.__or__ = Mock(return_value=chain_mock)
        chain_mock.invoke = Mock(return_value="Test answer")
        mock_prompt.__or__ = Mock(return_value=chain_mock)
        
        result = generator.generate("What is cholesterol?", include_sources=True)
        
        assert len(result["sources"]) > 0
        assert all("source" in source for source in result["sources"])
        assert all("test" in source for source in result["sources"])


def test_generate_without_sources(mock_retriever, generator_with_mock_llm, sample_documents):
    """Test generate() with include_sources=False."""
    generator, _ = generator_with_mock_llm
    mock_retriever.retrieve.return_value = sample_documents
    
    with patch.object(generator, 'prompt_builder') as mock_prompt_builder:
        mock_prompt = Mock()
        mock_prompt_builder.build_prompt.return_value = mock_prompt
        mock_prompt_builder.truncate_context.return_value = sample_documents
        mock_prompt_builder.max_context_tokens = 3000
        
        chain_mock = Mock()
        chain_mock.__or__ = Mock(return_value=chain_mock)
        chain_mock.invoke = Mock(return_value="Test answer")
        mock_prompt.__or__ = Mock(return_value=chain_mock)
        
        result = generator.generate("What is cholesterol?", include_sources=False)
        
        assert result["sources"] == []


def test_generate_with_top_k(mock_retriever, generator_with_mock_llm, sample_documents):
    """Test generate() with custom top_k parameter."""
    generator, _ = generator_with_mock_llm
    mock_retriever.retrieve.return_value = sample_documents
    
    with patch.object(generator, 'prompt_builder') as mock_prompt_builder:
        mock_prompt = Mock()
        mock_prompt_builder.build_prompt.return_value = mock_prompt
        mock_prompt_builder.truncate_context.return_value = sample_documents
        mock_prompt_builder.max_context_tokens = 3000
        
        chain_mock = Mock()
        chain_mock.__or__ = Mock(return_value=chain_mock)
        chain_mock.invoke = Mock(return_value="Test answer")
        mock_prompt.__or__ = Mock(return_value=chain_mock)
        
        generator.generate("What is cholesterol?", top_k=5)
        
        # Verify retriever was called with top_k
        mock_retriever.retrieve.assert_called_once()
        call_args = mock_retriever.retrieve.call_args
        assert call_args[1]["top_k"] == 5


def test_generate_with_examples(mock_retriever, generator_with_mock_llm, sample_documents):
    """Test generate() with include_examples=True."""
    generator, _ = generator_with_mock_llm
    mock_retriever.retrieve.return_value = sample_documents
    
    with patch.object(generator, 'prompt_builder') as mock_prompt_builder:
        mock_prompt = Mock()
        mock_prompt_builder.build_prompt.return_value = mock_prompt
        mock_prompt_builder.truncate_context.return_value = sample_documents
        mock_prompt_builder.max_context_tokens = 3000
        
        chain_mock = Mock()
        chain_mock.__or__ = Mock(return_value=chain_mock)
        chain_mock.invoke = Mock(return_value="Test answer")
        mock_prompt.__or__ = Mock(return_value=chain_mock)
        
        generator.generate("What is cholesterol?", include_examples=True)
        
        # Verify build_prompt was called with include_examples=True
        mock_prompt_builder.build_prompt.assert_called_once()
        call_args = mock_prompt_builder.build_prompt.call_args
        assert call_args[1]["include_examples"] is True


def test_generate_context_truncation(mock_retriever, generator_with_mock_llm, sample_documents):
    """Test that context is truncated when needed."""
    generator, _ = generator_with_mock_llm
    mock_retriever.retrieve.return_value = sample_documents
    
    with patch.object(generator, 'prompt_builder') as mock_prompt_builder:
        mock_prompt = Mock()
        mock_prompt_builder.build_prompt.return_value = mock_prompt
        mock_prompt_builder.max_context_tokens = 3000
        
        # Return truncated documents
        truncated_docs = sample_documents[:1]
        mock_prompt_builder.truncate_context.return_value = truncated_docs
        
        chain_mock = Mock()
        chain_mock.__or__ = Mock(return_value=chain_mock)
        chain_mock.invoke = Mock(return_value="Test answer")
        mock_prompt.__or__ = Mock(return_value=chain_mock)
        
        result = generator.generate("What is cholesterol?")
        
        # Verify truncate_context was called
        mock_prompt_builder.truncate_context.assert_called_once()
        assert result["metadata"]["retrieved_docs"] == len(truncated_docs)


def test_stream_generate_with_documents(mock_retriever, generator_with_mock_llm, sample_documents):
    """Test stream_generate() with documents."""
    generator, _ = generator_with_mock_llm
    mock_retriever.retrieve.return_value = sample_documents
    
    with patch.object(generator, 'prompt_builder') as mock_prompt_builder:
        mock_prompt = Mock()
        mock_prompt_builder.build_prompt.return_value = mock_prompt
        mock_prompt_builder.truncate_context.return_value = sample_documents
        mock_prompt_builder.max_context_tokens = 3000
        
        # Mock streaming chain
        chain_mock = Mock()
        chain_mock.__or__ = Mock(return_value=chain_mock)
        chain_mock.stream = Mock(return_value=["Chunk ", "1", " Chunk ", "2"])
        mock_prompt.__or__ = Mock(return_value=chain_mock)
        
        chunks = list(generator.stream_generate("What is cholesterol?"))
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)


def test_stream_generate_empty_context(mock_retriever, generator_with_mock_llm):
    """Test stream_generate() when no documents are retrieved."""
    generator, _ = generator_with_mock_llm
    mock_retriever.retrieve.return_value = []
    
    chunks = list(generator.stream_generate("What is cholesterol?"))
    
    assert len(chunks) == 1
    assert "couldn't find" in chunks[0].lower()


def test_generate_metadata_includes_query(mock_retriever, generator_with_mock_llm, sample_documents):
    """Test that metadata includes the query."""
    generator, _ = generator_with_mock_llm
    mock_retriever.retrieve.return_value = sample_documents
    
    with patch.object(generator, 'prompt_builder') as mock_prompt_builder:
        mock_prompt = Mock()
        mock_prompt_builder.build_prompt.return_value = mock_prompt
        mock_prompt_builder.truncate_context.return_value = sample_documents
        mock_prompt_builder.max_context_tokens = 3000
        
        chain_mock = Mock()
        chain_mock.__or__ = Mock(return_value=chain_mock)
        chain_mock.invoke = Mock(return_value="Test answer")
        mock_prompt.__or__ = Mock(return_value=chain_mock)
        
        query = "What is cholesterol?"
        result = generator.generate(query)
        
        assert result["metadata"]["query"] == query
        assert "model" in result["metadata"]
        assert "retrieved_docs" in result["metadata"]
