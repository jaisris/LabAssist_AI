"""
Configuration settings for the RAG Lab Tests application.
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = BASE_DIR / "chroma_db"

# Embedding configuration
EMBEDDING_MODEL = "text-embedding-3-large"  # Better quality embeddings for improved retrieval

# LLM configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1000

# Retrieval configuration
DEFAULT_TOP_K = 10  # Number of documents to retrieve
SIMILARITY_THRESHOLD = 0.7  # Minimum similarity score (0-1) to filter results
ENABLE_RERANKING = False

# Re-ranking configuration
# Options: "cohere", "voyage", None (for similarity-based ranking - default, no API key needed)
RERANK_MODEL = None  # Reranking model to use (None = similarity-based ranking only, no API key needed)
# For cohere: "rerank-v4.0-fast" or "rerank-v4.0-pro"
# For voyage: "rerank-2.5" or "rerank-2.5-lite"
RERANK_MODEL_NAME = "rerank-v4.0-fast"  # Model name for the selected reranker (only used if RERANK_MODEL is set)
RERANK_TOP_N = None  # Number of documents to rerank (None = rerank all retrieved documents)

# Chunking configuration
CHUNKING_STRATEGY = "semantic"  # Options: "semantic", "recursive_character", "character"
CHUNK_SIZE = 800  # Used as reference for semantic chunking (not strict limit)
CHUNK_OVERLAP = 150  # Used as reference for semantic chunking (not strict limit)
CHUNK_SEPARATORS = [
    "\n\n",   # paragraphs
    "\n",     # lines
    " | ",    # table rows
    ". ",     # sentences
    " "       # words
]

# Semantic chunking configuration
SEMANTIC_CHUNK_BREAKPOINT_THRESHOLD_TYPE = "percentile"  # Options: "percentile", "standard_deviation"
SEMANTIC_CHUNK_BREAKPOINT_THRESHOLD_AMOUNT = 75  # 75th percentile - groups similar sentences together

# Context management
MAX_CONTEXT_TOKENS = 3000

# Guardrails configuration
MIN_RELEVANCE_SCORE = 0.5

# Evaluation configuration
USE_RAGAS_METRICS = True  # Use RAGAS metrics for evaluation (requires LLM for evaluator)
EVALUATOR_LLM_MODEL = "gpt-4o-mini"  # LLM model for RAGAS evaluator
EVALUATOR_LLM_TEMPERATURE = 0.1  # Temperature for evaluator LLM

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")  # Required if using Cohere rerank
VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")  # Required if using Voyage AI rerank
