"""
Unit tests for PromptBuilder.
"""
import pytest
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from app.rag.prompt import PromptBuilder


@pytest.fixture
def prompt_builder():
    """Create PromptBuilder instance."""
    return PromptBuilder(max_context_tokens=3000)


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
        ),
        Document(
            page_content="Blood glucose measures sugar in your blood.",
            metadata={"source": "MedlinePlus", "test": "glucose", "similarity_score": 0.65}
        )
    ]


def test_prompt_builder_initialization():
    """Test PromptBuilder initialization."""
    builder = PromptBuilder(max_context_tokens=5000)
    assert builder.max_context_tokens == 5000
    
    # Test default
    builder_default = PromptBuilder()
    assert builder_default.max_context_tokens == 3000


def test_count_tokens_estimate(prompt_builder):
    """Test token count estimation."""
    # Test with various text lengths
    assert prompt_builder.count_tokens_estimate("") == 0
    assert prompt_builder.count_tokens_estimate("test") == 1  # 4 chars / 4 = 1
    assert prompt_builder.count_tokens_estimate("test test test test") == 4  # 19 chars / 4 = 4.75 -> 4
    assert prompt_builder.count_tokens_estimate("a" * 100) == 25  # 100 chars / 4 = 25


def test_build_prompt_basic(prompt_builder, sample_documents):
    """Test basic prompt building."""
    query = "What is cholesterol?"
    prompt = prompt_builder.build_prompt(query, sample_documents, include_examples=False)
    
    assert isinstance(prompt, ChatPromptTemplate)
    
    # Check that prompt can be formatted
    messages = prompt.format_messages()
    assert len(messages) >= 2  # System message + user message
    
    # Check system message contains system prompt
    system_msg = str(messages[0])
    assert "medical laboratory test assistant" in system_msg.lower()
    
    # Check user message contains query
    user_msg = str(messages[-1])
    assert query.lower() in user_msg.lower()


def test_build_prompt_with_examples(prompt_builder, sample_documents):
    """Test prompt building with few-shot examples."""
    query = "What is cholesterol?"
    prompt = prompt_builder.build_prompt(query, sample_documents, include_examples=True)
    
    messages = prompt.format_messages()
    
    # With examples, should have more messages (system + examples + user)
    assert len(messages) >= 4  # System + 2 example pairs (human+assistant) + user
    
    # Check that examples are included
    messages_str = " ".join(str(m) for m in messages)
    assert "cholesterol" in messages_str.lower() or "hba1c" in messages_str.lower()


def test_build_prompt_with_sources(prompt_builder, sample_documents):
    """Test prompt building includes source information."""
    query = "What is cholesterol?"
    prompt = prompt_builder.build_prompt(query, sample_documents, include_examples=False)
    
    messages = prompt.format_messages()
    system_msg = str(messages[0])
    
    # Should include source information
    assert "MedlinePlus" in system_msg or "Available sources" in system_msg


def test_build_prompt_empty_documents(prompt_builder):
    """Test prompt building with empty document list."""
    query = "What is cholesterol?"
    prompt = prompt_builder.build_prompt(query, [], include_examples=False)
    
    assert isinstance(prompt, ChatPromptTemplate)
    messages = prompt.format_messages()
    assert len(messages) >= 2  # System + user message


def test_build_prompt_context_truncation(prompt_builder):
    """Test prompt building with context truncation."""
    # Create a very long document that exceeds token limit
    long_content = "This is a test. " * 10000  # Very long content
    long_doc = Document(
        page_content=long_content,
        metadata={"source": "test", "similarity_score": 0.8}
    )
    
    # Set small max_context_tokens to force truncation
    builder = PromptBuilder(max_context_tokens=100)  # Very small limit
    query = "Test query"
    prompt = builder.build_prompt(query, [long_doc], include_examples=False)
    
    messages = prompt.format_messages()
    user_msg = str(messages[-1])
    
    # Should be truncated
    assert len(user_msg) < len(long_content)
    assert "[Context truncated due to length...]" in user_msg or len(user_msg) < len(long_content) * 0.5


def test_truncate_context_no_truncation(prompt_builder, sample_documents):
    """Test truncate_context when no truncation needed."""
    result = prompt_builder.truncate_context(sample_documents, max_tokens=10000)
    
    assert len(result) == len(sample_documents)
    assert all(doc in result for doc in sample_documents)


def test_truncate_context_with_truncation(prompt_builder):
    """Test truncate_context when truncation is needed."""
    # Create documents that exceed token limit
    docs = [
        Document(
            page_content="Short document. " * 10,  # ~150 chars = ~37 tokens
            metadata={"source": "test1"}
        ),
        Document(
            page_content="Medium document. " * 50,  # ~750 chars = ~187 tokens
            metadata={"source": "test2"}
        ),
        Document(
            page_content="Long document. " * 200,  # ~3000 chars = ~750 tokens
            metadata={"source": "test3"}
        )
    ]
    
    # Set max_tokens to only fit first two documents
    result = prompt_builder.truncate_context(docs, max_tokens=250)
    
    # Should truncate to fit within limit
    assert len(result) <= len(docs)
    total_tokens = sum(prompt_builder.count_tokens_estimate(doc.page_content) for doc in result)
    assert total_tokens <= 250


def test_truncate_context_partial_document(prompt_builder):
    """Test truncate_context with partial document inclusion."""
    # Create a document that's too long but can be partially included
    long_doc = Document(
        page_content="Word " * 1000,  # ~5000 chars = ~1250 tokens
        metadata={"source": "test"}
    )
    
    # Set max_tokens to allow partial document
    result = prompt_builder.truncate_context([long_doc], max_tokens=200)
    
    # Should include partial document if remaining_tokens > 100
    if len(result) > 0:
        assert result[0].metadata.get('truncated', False) is True
        assert len(result[0].page_content) < len(long_doc.page_content)


def test_truncate_context_empty_list(prompt_builder):
    """Test truncate_context with empty document list."""
    result = prompt_builder.truncate_context([], max_tokens=1000)
    assert result == []


def test_build_prompt_document_metadata(prompt_builder, sample_documents):
    """Test that document metadata is included in prompt."""
    query = "What is cholesterol?"
    prompt = prompt_builder.build_prompt(query, sample_documents, include_examples=False)
    
    messages = prompt.format_messages()
    user_msg = str(messages[-1])
    
    # Should include document metadata (source, test, similarity)
    assert "Document" in user_msg
    assert "Source:" in user_msg or "cholesterol" in user_msg.lower()
    assert "Relevance:" in user_msg or "0.85" in user_msg or "0.75" in user_msg


def test_build_prompt_multiple_sources(prompt_builder):
    """Test prompt building with documents from multiple sources."""
    docs = [
        Document(
            page_content="Content from source A",
            metadata={"source": "SourceA", "test": "test1", "similarity_score": 0.8}
        ),
        Document(
            page_content="Content from source B",
            metadata={"source": "SourceB", "test": "test2", "similarity_score": 0.7}
        ),
        Document(
            page_content="More content from source A",
            metadata={"source": "SourceA", "test": "test3", "similarity_score": 0.6}
        )
    ]
    
    query = "Test query"
    prompt = prompt_builder.build_prompt(query, docs, include_examples=False)
    
    messages = prompt.format_messages()
    system_msg = str(messages[0])
    
    # Should list unique sources
    assert "SourceA" in system_msg
    assert "SourceB" in system_msg
