from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import logging
import shutil
from pathlib import Path

from app.config import CHROMA_PERSIST_DIR

logger = logging.getLogger(__name__)


def get_embeddings():
    """
    Create OpenAI embedding model.
    """
    from app.config import EMBEDDING_MODEL
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL}")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    logger.info(f"Embedding model initialized successfully")
    return embeddings


def build_vector_store(documents):
    """
    Build and persist a Chroma vector store from documents.
    Always deletes existing vector store before building a new one.
    
    Args:
        documents: List of documents to index
        
    Returns:
        Chroma vector store instance
    """
    logger.info(f"Building vector store from {len(documents)} documents")
    
    if not documents:
        logger.warning("No documents provided to build vector store")
        raise ValueError("Cannot build vector store with empty document list")
    
    embeddings = get_embeddings()
    
    # Always delete existing vector store before building new one
    persist_path = Path(CHROMA_PERSIST_DIR)
    if persist_path.exists() and persist_path.is_dir():
        logger.info(f"Deleting existing vector store at {persist_path}")
        shutil.rmtree(persist_path)
    
    logger.info(f"Creating embeddings for {len(documents)} documents...")
    logger.debug(f"Persist directory: {CHROMA_PERSIST_DIR}")
    
    # Ensure directory exists
    persist_path.mkdir(parents=True, exist_ok=True)
    
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR)
    )

    logger.info("Persisting vector store to disk...")
    vectordb.persist()
    
    # Verify the vector store
    collection_count = vectordb._collection.count()
    logger.info(f"Vector store built successfully: {collection_count} documents indexed")
    
    return vectordb


def load_vector_store():
    """
    Load an existing Chroma vector store from disk.
    """
    persist_path = Path(CHROMA_PERSIST_DIR)
    logger.info(f"Loading vector store from {persist_path}")
    
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Vector store not found at {persist_path}. "
            "Run ingestion first: python -m app.main --ingest"
        )
    
    embeddings = get_embeddings()

    try:
        vectordb = Chroma(
            persist_directory=str(persist_path),
            embedding_function=embeddings
        )
        
        # Verify the loaded vector store
        try:
            collection_count = vectordb._collection.count()
            logger.info(f"Vector store loaded successfully: {collection_count} documents available")
        except Exception as e:
            logger.warning(f"Could not verify vector store count: {e}")
        
        return vectordb
    except Exception as e:
        if "dimension" in str(e).lower() or "embedding" in str(e).lower():
            raise ValueError(
                f"Embedding dimension mismatch: {e}. "
                "The existing vector store was created with a different embedding model. "
                "Delete the chroma_db directory and rebuild: rm -rf chroma_db"
            ) from e
        raise
