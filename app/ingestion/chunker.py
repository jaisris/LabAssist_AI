from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from typing import List, Optional
import logging

from app.config import (
    CHUNKING_STRATEGY,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    EMBEDDING_MODEL,
    SEMANTIC_CHUNK_BREAKPOINT_THRESHOLD_TYPE,
    SEMANTIC_CHUNK_BREAKPOINT_THRESHOLD_AMOUNT
)

logger = logging.getLogger(__name__)


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
    
    if strategy == "semantic":
        # Semantic chunking uses embeddings to group semantically similar sentences
        # This preserves semantic coherence better than fixed-size chunking
        from app.ingestion.indexer import get_embeddings
        
        logger.info(
            f"Initializing semantic chunker with embedding model: {EMBEDDING_MODEL}, "
            f"threshold_type={SEMANTIC_CHUNK_BREAKPOINT_THRESHOLD_TYPE}, "
            f"threshold_amount={SEMANTIC_CHUNK_BREAKPOINT_THRESHOLD_AMOUNT}"
        )
        embeddings = get_embeddings()
        
        # SemanticChunker parameters:
        # - breakpoint_threshold_type: "percentile" or "standard_deviation"
        #   - "percentile": Uses percentile of similarity distribution (recommended)
        #   - "standard_deviation": Uses standard deviation from mean similarity
        # - breakpoint_threshold_amount: threshold value
        #   - For percentile: 75 means 75th percentile (groups top 25% similar sentences)
        #   - For standard_deviation: typically 0.5-1.5
        # - add_start_index: whether to add start index to metadata
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=SEMANTIC_CHUNK_BREAKPOINT_THRESHOLD_TYPE,
            breakpoint_threshold_amount=SEMANTIC_CHUNK_BREAKPOINT_THRESHOLD_AMOUNT,
            add_start_index=True  # Add start index for better tracking
        )
    elif strategy == "recursive_character":
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
            f"Supported strategies: 'semantic', 'recursive_character', 'character'"
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
    effective_chunk_size = chunk_size or CHUNK_SIZE
    effective_overlap = overlap or CHUNK_OVERLAP
    effective_strategy = strategy or CHUNKING_STRATEGY
    
    if effective_strategy == "semantic":
        logger.info(
            f"Chunking {len(documents)} document(s) with strategy='{effective_strategy}' "
            f"(semantic chunking uses embeddings to group similar sentences, "
            f"reference chunk_size={effective_chunk_size})"
        )
    else:
        logger.info(
            f"Chunking {len(documents)} document(s) with strategy='{effective_strategy}', "
            f"chunk_size={effective_chunk_size}, overlap={effective_overlap}"
        )
    
    splitter = get_text_splitter(
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators
    )

    chunked_docs = splitter.split_documents(documents)
    
    # Log chunking statistics
    if chunked_docs:
        chunk_lengths = [len(doc.page_content) for doc in chunked_docs]
        avg_length = sum(chunk_lengths) / len(chunk_lengths)
        min_length = min(chunk_lengths)
        max_length = max(chunk_lengths)
        
        logger.info(
            f"Chunking complete: {len(chunked_docs)} chunks created, "
            f"length stats: avg={avg_length:.0f}, min={min_length}, max={max_length}"
        )
        
        # Log sample chunk for validation
        if logger.isEnabledFor(logging.DEBUG):
            sample_chunk = chunked_docs[0]
            logger.debug(
                f"Sample chunk: length={len(sample_chunk.page_content)}, "
                f"metadata={sample_chunk.metadata}, "
                f"preview='{sample_chunk.page_content[:100]}...'"
            )
    
    return chunked_docs