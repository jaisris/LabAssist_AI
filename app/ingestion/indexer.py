from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_PERSIST_DIR = "chroma_db"


def get_embeddings():
    """
    Create OpenAI embedding model.
    """
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )


def build_vector_store(documents):
    """
    Build and persist a Chroma vector store from documents.
    """
    embeddings = get_embeddings()

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

    vectordb.persist()
    return vectordb


def load_vector_store():
    """
    Load an existing Chroma vector store from disk.
    """
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embeddings
    )
