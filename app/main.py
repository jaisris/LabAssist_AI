"""
Main entry point for the RAG Lab Tests application.
Can be used as CLI or to run the API server.
"""
import argparse
import sys
import warnings
from pathlib import Path

from app.ingestion.loader import extract_text_and_tables, clean_text
from app.ingestion.chunker import build_documents, chunk_documents
from app.ingestion.indexer import build_vector_store, load_vector_store
from app.rag import initialize_rag_components, process_chat_query
from app.config import DATA_DIR

# Suppress known benign PDF font warnings from pdfplumber
warnings.filterwarnings(
    "ignore",
    message="Could not get FontBBox from font descriptor.*"
)

# Document metadata mapping
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


def run_ingestion_pipeline():
    """Run the document ingestion pipeline."""
    print("Starting ingestion pipeline...")
    all_chunks = []

    for pdf_path in DATA_DIR.glob("*.pdf"):
        print(f"\nProcessing: {pdf_path.name}")

        # 1. Extract text and tables
        raw_text = extract_text_and_tables(pdf_path)

        if not raw_text.strip():
            print("   Warning: No text extracted, skipping.")
            continue

        # 2. Clean extracted text
        cleaned_text = clean_text(raw_text)

        # 3. Build LangChain Document
        metadata = DOCUMENT_METADATA_MAP.get(
            pdf_path.name,
            {"source": "unknown", "domain": "lab_tests"}
        )

        documents = build_documents(
            text=cleaned_text,
            metadata=metadata
        )

        # 4. Chunk documents
        chunks = chunk_documents(documents)

        print(f"   Chunks created: {len(chunks)}")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("\nError: No chunks created. Exiting.")
        sys.exit(1)

    print("\n" + "=" * 50)
    print(f"Total chunks across PDFs: {len(all_chunks)}")
    print("=" * 50)

    print("\nBuilding Chroma vector store...")
    build_vector_store(all_chunks)
    print("Vector store persisted successfully!")


def run_chat_cli():
    """Run interactive CLI chat."""
    print("RAG Lab Tests Chat Assistant")
    print("Type 'exit' or 'quit' to end the conversation\n")
    
    # Initialize RAG components
    try:
        vector_store, retriever, generator, guardrails = initialize_rag_components()
    except Exception as e:
        print(f"Error loading vector store: {e}")
        print("Run ingestion first: python -m app.main --ingest")
        sys.exit(1)
    
    # Chat loop
    while True:
        try:
            query = input("\nYour question: ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            # Process query using shared function
            print("\nThinking...")
            result = process_chat_query(
                query=query,
                generator=generator,
                guardrails=guardrails,
                include_sources=True
            )
            
            # Display results
            if result.get("is_ambiguous"):
                print(f"Warning: {result['answer']}")
            elif not result.get("is_relevant"):
                print(f"\nWarning: {result['answer']}")
            else:
                print(f"\nAnswer:\n{result['answer']}")
                # if result["sources"]:
                #     sources = list(dict.fromkeys(s.get("source", "Unknown") for s in result["sources"]))
                #     print(f"\nSources: {', '.join(sources)}")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="RAG Lab Tests Application")
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run document ingestion pipeline"
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Run interactive CLI chat"
    )
    parser.add_argument(
        "--api",
        action="store_true",
        help="Run API server"
    )
    
    args = parser.parse_args()
    
    if args.ingest:
        run_ingestion_pipeline()
    elif args.chat:
        run_chat_cli()
    elif args.api:
        import uvicorn
        uvicorn.run("app.api:app", host="0.0.0.0", port=8000, reload=True)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
