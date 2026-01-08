import warnings
from pathlib import Path

from app.ingestion.loader import extract_text_and_tables, clean_text
from app.ingestion.chunker import build_documents, chunk_documents
from app.ingestion.indexer import build_vector_store, load_vector_store

# ------------------------------------------------------------------
# Suppress known benign PDF font warnings from pdfplumber
# ------------------------------------------------------------------
warnings.filterwarnings(
    "ignore",
    message="Could not get FontBBox from font descriptor.*"
)

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
DATA_DIR = Path("data")

DOCUMENT_METADATA_MAP = {
    "cholesterol.pdf": {
        "source": "MedlinePlus",
        "test": "cholesterol",
        "domain": "lab_tests"
    },
    "blood_glucose.pdf": {
        "source": "MedlinePlus",
        "test": "blood_glucose",
        "domain": "lab_tests"
    },
    "lab_results_overview.pdf": {
        "source": "MedlinePlus",
        "test": "general",
        "domain": "lab_tests"
    },
    "HbA1c.pdf": {
        "source": "MedlinePlus",
        "test": "HbA1c",
        "domain": "lab_tests"
    }
}

# ------------------------------------------------------------------
# Ingestion + Chunking Validation
# ------------------------------------------------------------------
def run_ingestion():
    all_chunks = []

    for pdf_path in DATA_DIR.glob("*.pdf"):
        print(f"\n Processing: {pdf_path.name}")

        # 1. Extract text and tables
        raw_text = extract_text_and_tables(pdf_path)

        if not raw_text.strip():
            print(" No text extracted, skipping.")
            continue

        # 2. Clean extracted text
        cleaned_text = clean_text(raw_text)

        # 3. Build LangChain Document
        metadata = DOCUMENT_METADATA_MAP.get(
            pdf_path.name,
            {"source": "unknown"}
        )

        documents = build_documents(
            text=cleaned_text,
            metadata=metadata
        )

        # 4. Chunk documents
        chunks = chunk_documents(documents)

        print(f"✅ Chunks created: {len(chunks)}")

        # 5. Preview first chunk for manual validation
        if chunks:
            print("\n--- Sample Chunk Preview ---")
            print(chunks[0].page_content[:600])
            print("\n--- Metadata ---")
            print(chunks[0].metadata)

        all_chunks.extend(chunks)

    print("\n==============================")
    print(f"📦 Total chunks across PDFs: {len(all_chunks)}")
    print("==============================")

    return all_chunks




# ------------------------------------------------------------------
# Script Entry Point
# ------------------------------------------------------------------
if __name__ == "__main__":
    all_chunks = run_ingestion()
    # After all_chunks collected
    print("\n💾 Building Chroma vector store...")
    build_vector_store(all_chunks)
    print("✅ Vector store persisted")
    db = load_vector_store()
    docs = db.similarity_search("What is normal LDL cholesterol?", k=3)

    for d in docs:
        print(d.page_content[:200])
        print(d.metadata)
