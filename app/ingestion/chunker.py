from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from langchain_core.documents import Document
from typing import List, Optional

from app.config import (
    CHUNKING_STRATEGY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS
)


def build_documents(text: str, metadata: dict) -> list[Document]:
    return [
        Document(
            page_content=text,
            metadata=metadata
        )
    ]


def get_text_splitter(
    strategy: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    separators: Optional[List[str]] = None
):
    """
    Get text splitter based on configured strategy.
    
    Args:
        strategy: Chunking strategy name (defaults to config)
        chunk_size: Chunk size (defaults to config)
        chunk_overlap: Chunk overlap (defaults to config)
        separators: List of separators (defaults to config)
        
    Returns:
        Configured text splitter instance
        
    Raises:
        ValueError: If strategy is not supported
    """
    strategy = strategy or CHUNKING_STRATEGY
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP
    separators = separators or CHUNK_SEPARATORS
    
    if strategy == "recursive_character":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=separators
        )
    elif strategy == "character":
        # CharacterTextSplitter uses a single separator
        # Use the first separator or default to "\n\n"
        separator = separators[0] if separators else "\n\n"
        return CharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator
        )
    else:
        raise ValueError(
            f"Unsupported chunking strategy: {strategy}. "
            f"Supported strategies: 'recursive_character', 'character'"
        )


def chunk_documents(
    documents,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
    strategy: Optional[str] = None,
    separators: Optional[List[str]] = None
):
    """
    Chunk documents with overlap for context preservation.
    Uses configured chunking strategy from config.

    Args:
        documents: List of document texts
        chunk_size: Target chunk size in characters (defaults to config)
        overlap: Overlap between chunks (defaults to config)
        strategy: Chunking strategy name (defaults to config)
        separators: List of separators for recursive strategy (defaults to config)

    Returns:
        List of chunks with metadata
    """
    splitter = get_text_splitter(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators
    )

    chunked_docs = splitter.split_documents(documents)

    return chunked_docs