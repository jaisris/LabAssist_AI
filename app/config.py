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
EMBEDDING_MODEL = "text-embedding-3-small"

# LLM configuration
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1000

# Retrieval configuration
DEFAULT_TOP_K = 15  # Increased from 5 to capture range information that may be ranked lower
SIMILARITY_THRESHOLD = 0.3  # Lowered from 0.6 - L2 distance conversion produces lower similarity scores
ENABLE_RERANKING = True

# Chunking configuration
CHUNKING_STRATEGY = "recursive_character"  # Options: "recursive_character", "character"
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
CHUNK_SEPARATORS = [
    "\n\n",   # paragraphs
    "\n",     # lines
    " | ",    # table rows
    ". ",     # sentences
    " "       # words
]

# Context management
MAX_CONTEXT_TOKENS = 3000

# Guardrails configuration
MIN_RELEVANCE_SCORE = 0.5

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
