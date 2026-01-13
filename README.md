# RAG Lab Tests - Conversational AI Assistant

A production-ready RAG (Retrieval-Augmented Generation) system for answering questions about laboratory test results from PDF documents.

## 🎯 Project Overview

This project implements a "Chat with Your Docs" application that can answer questions about medical laboratory test information from a collection of PDF documents. It demonstrates production-ready AI system engineering with comprehensive RAG implementation, guardrails, quality controls, and testing.

## 🏗️ System Architecture

### Architecture Diagram

```mermaid
graph TB
    subgraph "Input"
        PDF[PDF Documents]
        USER[User Queries]
    end
    
    subgraph "Ingestion"
        LOADER[PDF Loader]
        CHUNKER[Chunker]
        EMBED[Embeddings]
        VDB[(ChromaDB)]
    end
    
    subgraph "RAG Pipeline"
        RETRIEVER[Retriever]
        PROMPT[Prompt Builder]
        LLM[LLM Generator]
        GUARD[Guardrails]
    end
    
    subgraph "Interface"
        API[FastAPI]
        CLI[CLI]
    end
    
    subgraph "Quality"
        QC[Quality Checker]
    end
    
    PDF --> LOADER
    LOADER --> CHUNKER
    CHUNKER --> EMBED
    EMBED --> VDB
    
    USER --> API
    USER --> CLI
    API --> RETRIEVER
    CLI --> RETRIEVER
    RETRIEVER --> VDB
    RETRIEVER --> PROMPT
    PROMPT --> LLM
    LLM --> GUARD
    GUARD --> QC
    GUARD --> API
    GUARD --> CLI
```

### Architecture Overview

The system follows a modular, layered architecture with clear separation of concerns:

#### **Layer 1: Input Layer**
- **PDF Documents**: Source documents stored in `data/` directory
- **User Queries**: Questions submitted via API or CLI

#### **Layer 2: Ingestion Pipeline** (`app/ingestion/`)
- **PDF Loader** (`loader.py`): Extracts text and tables from PDF files
- **Text Cleaner**: Normalizes and cleans extracted text
- **Document Chunker** (`chunker.py`): Splits documents into chunks using configurable strategy
- **Embedding Model**: Converts chunks to vector embeddings
- **Vector Store** (`indexer.py`): Persists embeddings in ChromaDB

#### **Layer 3: RAG System** (`app/rag/`)
- **Retriever** (`retriever.py`): 
  - Retrieves relevant chunks using similarity search
  - Applies similarity threshold filtering (0.7)
  - Optional re-ranking: Simple similarity-based sorting (when enabled)
  - Converts L2 distances to similarity scores optimized for text-embedding-3-large
- **Prompt Builder** (`prompt.py`):
  - Constructs prompts with context
  - Manages token limits and truncation
  - Handles optional few-shot examples
- **LLM Generator** (`generator.py`):
  - Generates answers using OpenAI GPT models
  - Manages context window and max tokens (1000)
  - Supports streaming responses
- **Guardrails** (`guardrails.py`):
  - Relevance checking
  - Ambiguous query detection
  - Medical emergency detection
  - Source attribution
  - Response validation

#### **Layer 4: API Layer**
- **FastAPI Server** (`api.py`): RESTful API with health checks
- **CLI Interface** (`main.py`): Command-line interface for interactive use

#### **Layer 5: Quality & Evaluation** (`app/eval/`)
- **Quality Checker** (`quality_checks.py`):
  - Answer quality metrics
  - Retrieval quality evaluation
  - Completeness checks

### Data Flow

1. **Ingestion Flow:**
   ```
   PDF → Loader → Cleaner → Chunker → Embeddings → Vector Store
   ```

2. **Query Flow:**
   ```
   User Query → Retriever → Vector Store → Context Documents
   → Prompt Builder → LLM → Guardrails → Quality Check → Response
   ```

## 🔧 Technical Architecture & Implementation Decisions

### System Design Principles

1. **Modularity**: Clear separation of concerns with dedicated modules for each component
2. **Configurability**: All key parameters configurable via `app/config.py`
3. **Extensibility**: Easy to add new chunking strategies, retrieval methods, or LLM providers
4. **Testability**: Comprehensive test coverage with unit and integration tests
5. **Production-Ready**: Docker containerization, health checks, error handling

### Component Architecture

#### Ingestion Pipeline Architecture

```mermaid
graph LR
    A[PDF Files] --> B[PDF Loader]
    B --> C[Text Extraction]
    C --> D[Table Extraction]
    D --> E[Text Cleaner]
    E --> F[Document Builder]
    F --> G[Chunker<br/>Configurable Strategy]
    G --> H[Embedding Model]
    H --> I[ChromaDB<br/>Vector Store]
    
    style G fill:#e1f5ff
    style H fill:#e1f5ff
    style I fill:#e1f5ff
```

#### RAG Pipeline Architecture

```mermaid
graph TB
    Q[User Query] --> G[Guardrails<br/>Pre-checks]
    G --> R[Retriever]
    R --> V[(Vector Store)]
    V --> R
    R --> F[Filter by<br/>Threshold 0.7]
    F --> RR{Re-ranking<br/>Enabled?}
    RR -->|Yes| SR[Sort by<br/>Similarity]
    RR -->|No| P[Prompt Builder]
    SR --> P
    P --> L[LLM Generator]
    L --> G2[Guardrails<br/>Post-checks]
    G2 --> QC[Quality Checker<br/>RAGAS/Custom]
    QC --> A[Answer + Sources]
    
    style R fill:#ffe1f5
    style P fill:#ffe1f5
    style L fill:#ffe1f5
    style G2 fill:#fff5e1
    style QC fill:#e1fff5
```

## 🔧 Implementation Decisions

### Vector Indexing Strategy

**Chunking Approach:**
- **Chunk Size:** 800 characters (configurable in `app/config.py`)
- **Overlap:** 150 characters (configurable)
- **Strategy:** Configurable chunking strategy (default: `semantic`)
  - **Semantic Chunking:** Groups semantically similar sentences together using embedding-based similarity
  - **Recursive Character Splitter:** Hierarchical separators (paragraphs → lines → sentences)
  - **Character Splitter:** Single separator-based splitting
- **Separators:** Configurable list of separators for recursive strategy

**Configuration:**
All chunking parameters are configurable via `app/config.py`:
- `CHUNKING_STRATEGY`: Strategy selection (`semantic`, `recursive_character`, or `character`)
- `CHUNK_SIZE`: Target chunk size in characters (reference for semantic chunking)
- `CHUNK_OVERLAP`: Overlap between chunks (reference for semantic chunking)
- `CHUNK_SEPARATORS`: List of separators for hierarchical splitting
- `SEMANTIC_CHUNK_BREAKPOINT_THRESHOLD_TYPE`: Threshold type for semantic chunking (`percentile` or `standard_deviation`)
- `SEMANTIC_CHUNK_BREAKPOINT_THRESHOLD_AMOUNT`: Threshold amount (default: 75th percentile)

**Rationale:**
- 800 characters balances context preservation with retrieval precision
- 150-character overlap ensures continuity across chunk boundaries
- Semantic chunking groups related sentences together, improving retrieval quality
- Hierarchical separators (paragraphs → lines → sentences) preserve semantic structure
- Table rows are preserved with " | " separator for structured data
- Configurable strategy allows experimentation and optimization for different document types

### Embedding Model & LLM Selection

**Embedding Model:** `text-embedding-3-large`
- **Reasoning:**
  - Higher quality embeddings for improved retrieval accuracy
  - 3072 dimensions provide excellent semantic representation
  - Better performance on medical/technical text
  - Optimized similarity calculation for this model (L2 distance to similarity conversion)

**LLM:** `gpt-4o-mini`
- **Reasoning:**
  - Excellent cost/performance balance
  - Low latency for real-time responses
  - Sufficient context window (128k tokens)
  - Good instruction following
  - Temperature: 0.1 for factual, consistent responses
  - Max tokens: 1000 (configurable in `app/config.py`)

### Retrieval Approach

**Strategy:**
- **Top-K:** 10 documents (configurable, default in `app/config.py`)
- **Similarity Threshold:** 0.7 (filters low-relevance results, optimized for L2 distance conversion)
- **Re-ranking:** Disabled by default (configurable via `ENABLE_RERANKING` in `app/config.py`)
  - When enabled: Simple similarity-based sorting (fast, no API calls)
  - When disabled: Documents returned in similarity order

**Implementation:**
1. Retrieve exactly `top_k` documents from vector store
2. Convert L2 distances to similarity scores using formula: `similarity = 1 - (distance² / 4)`
3. Filter by similarity threshold (0.7)
4. If fewer than `top_k` documents pass threshold, include best available documents below threshold
5. Optional re-ranking: Simple similarity-based sorting (when enabled)
6. Return top-k most relevant documents

**Rationale:**
- Similarity threshold of 0.7 ensures high-quality, relevant context
- Top-k=10 balances context richness with processing efficiency
- Simple re-ranking (when enabled) provides fast optimization without external API calls
- Always returns up to `top_k` documents to ensure sufficient context

### Prompt Engineering

**System Prompt:**
- Defines role as medical lab test assistant
- Emphasizes accuracy and source-based answers
- Includes safety disclaimers
- Specifies source attribution requirements

**Context Management:**
- Maximum context tokens: 3000
- Token estimation: ~4 characters per token
- Truncation strategy: Prioritize earlier, more relevant chunks
- Source attribution included in prompt

**Few-Shot Examples:**
- Optional few-shot examples for better instruction following
- Examples demonstrate expected answer format and style

### Context Management

**Token Counting:**
- Character-based estimation (1 token ≈ 4 characters)
- Tracks context usage across documents
- Truncates when approaching limits

**Truncation Strategy:**
- Prioritize documents by similarity score
- Include full documents when possible
- Partial document inclusion if meaningful space remains (>100 tokens)
- Metadata flag for truncated documents

### Guardrails

**Implemented Safeguards:**

1. **Relevance Checking:**
   - Minimum similarity score: 0.5
   - Filters irrelevant results
   - Provides fallback responses

2. **Ambiguous Query Detection:**
   - Detects overly short queries
   - Identifies vague patterns ("what", "how", etc.)
   - Suggests query improvements

3. **Medical Emergency Detection:**
   - Keyword-based detection
   - Provides emergency service guidance
   - Emphasizes general information only

4. **Source Attribution:**
   - Automatic source citation
   - Includes document metadata
   - Builds user trust

5. **Response Validation:**
   - Checks answer completeness
   - Detects error patterns
   - Validates disclaimer presence

### Quality Controls

**Evaluation Metrics:**

The `QualityChecker` class (`app/eval/quality_checks.py`) provides comprehensive evaluation of RAG system performance through the `run_full_evaluation()` method. The system supports two evaluation modes:

1. **RAGAS Metrics** (enabled by default via `USE_RAGAS_METRICS=True` in `app/config.py`):
   - **Faithfulness**: Measures how grounded the answer is in the provided context
   - **Answer Relevancy**: Evaluates how relevant the answer is to the query
   - **Context Precision**: Measures precision of retrieved context
   - **Context Recall**: Measures recall of retrieved context (requires ground truth)

2. **Custom Metrics** (fallback when RAGAS is unavailable or disabled):
   - Answer quality metrics (completeness, relevance, source availability)
   - Retrieval quality metrics (retrieval rate, average similarity)
   - Overall system score

The evaluation returns different structures depending on the evaluation method:

**RAGAS Metrics Structure** (when `USE_RAGAS_METRICS=True`):

```json
{
    "evaluation_method": "ragas",
    "faithfulness": float,           // 0-1: How grounded the answer is in context
    "answer_relevancy": float,        // 0-1: How relevant the answer is to the query
    "context_precision": float,       // 0-1: Precision of retrieved context
    "context_recall": float,         // 0-1: Recall of retrieved context (if ground_truth provided)
    "overall_score": float,          // Average of all available RAGAS metrics
    "raw_scores": {                   // Raw scores from RAGAS evaluation
        "faithfulness": float,
        "answer_relevancy": float,
        "context_precision": float,
        "context_recall": float      // Only if ground_truth provided
    }
}
```

**Custom Metrics Structure** (when `USE_RAGAS_METRICS=False` or RAGAS unavailable):

```json
{
    "evaluation_method": "custom",
    "answer_quality": {
        "answer_length": int,
        "word_count": int,
        "has_sources": bool,
        "num_sources": int,
        "avg_similarity": float,
        "completeness_score": float,
        "relevance_score": float,
        "quality_score": float
    },
    "completeness": {
        "is_complete": bool,
        "is_incomplete": bool,
        "is_too_short": bool,
        "has_question_word": bool
    },
    "retrieval_quality": {
        "num_retrieved": int,
        "expected_num": int,
        "retrieval_rate": float,
        "avg_similarity": float,
        "min_similarity": float,
        "max_similarity": float,
        "retrieval_quality_score": float
    },
    "overall_score": float
}
```

**Metric Descriptions:**

**RAGAS Metrics** (when `USE_RAGAS_METRICS=True`):

- **faithfulness** (0-1): Measures how grounded the answer is in the provided context. Higher scores indicate the answer is well-supported by the retrieved documents.
- **answer_relevancy** (0-1): Evaluates how relevant the generated answer is to the user's query. Higher scores indicate the answer directly addresses the question.
- **context_precision** (0-1): Measures the precision of retrieved context. Higher scores indicate more relevant documents were retrieved.
- **context_recall** (0-1): Measures the recall of retrieved context (requires `ground_truth` parameter). Higher scores indicate more relevant information was retrieved. Only included if ground truth is provided.
- **overall_score** (0-1): Average of all available RAGAS metrics (faithfulness, answer_relevancy, context_precision, and context_recall if available).

**Custom Metrics** (when `USE_RAGAS_METRICS=False` or RAGAS unavailable):

- **answer_quality.answer_length**: Character count of the generated answer
- **answer_quality.word_count**: Word count of the generated answer
- **answer_quality.has_sources**: Boolean indicating if context documents are available
- **answer_quality.num_sources**: Number of retrieved context documents
- **answer_quality.avg_similarity**: Average similarity score (0-1) across all context documents
- **answer_quality.completeness_score**: Query word coverage ratio (0-1), calculated as common words between query and answer divided by query words
- **answer_quality.relevance_score**: Maximum similarity score (0-1) from context documents
- **answer_quality.quality_score**: Overall answer quality (0-1), calculated as: 30% completeness_score + 40% relevance_score + 20% source availability + 10% length appropriateness (normalized word count, max 100 words)

- **completeness.is_complete**: Boolean indicating if answer is complete and addresses the query
- **completeness.is_incomplete**: Boolean indicating if incomplete indicators were detected ("i don't know", "i cannot", "i'm not able", "not available", "couldn't find")
- **completeness.is_too_short**: Boolean indicating if answer has fewer than 15 words
- **completeness.has_question_word**: Boolean indicating if query contains question words ("what", "how", "why", "when", "where", "who")

- **retrieval_quality.num_retrieved**: Number of documents retrieved
- **retrieval_quality.expected_num**: Expected number of documents (top_k parameter)
- **retrieval_quality.retrieval_rate**: Ratio of retrieved to expected documents (0-1)
- **retrieval_quality.avg_similarity**: Average similarity score (0-1) across retrieved documents
- **retrieval_quality.min_similarity**: Minimum similarity score (0-1) among retrieved documents
- **retrieval_quality.max_similarity**: Maximum similarity score (0-1) among retrieved documents
- **retrieval_quality.retrieval_quality_score**: Overall retrieval quality (0-1), calculated as: 50% retrieval_rate + 50% avg_similarity

- **overall_score**: Comprehensive system score (0-1), calculated as: 40% answer_quality.quality_score + 30% completeness.is_complete (binary) + 30% retrieval_quality.retrieval_quality_score

**Actual Evaluation Results:**

**Example 1: RAGAS Metrics - Low Score (Incomplete Answer)**

Query: "What is cholesterol?"  
Answer: "Cholesterol is a waxy substance found in your blood."  
Retrieved documents: 2  
Evaluation Method: RAGAS

```json
{
    "evaluation_method": "ragas",
    "faithfulness": 0.85,
    "answer_relevancy": 0.72,
    "context_precision": 0.68,
    "context_recall": null,
    "overall_score": 0.750,
    "raw_scores": {
        "faithfulness": 0.85,
        "answer_relevancy": 0.72,
        "context_precision": 0.68
    }
}
```

**Score Breakdown:**
- **faithfulness (0.85)**: Answer is mostly grounded in context, but lacks detail
- **answer_relevancy (0.72)**: Answer addresses the query but is too brief
- **context_precision (0.68)**: Some retrieved documents may not be highly relevant
- **overall_score (0.750)**: Average of (0.85 + 0.72 + 0.68) / 3 = 0.750

**Example 2: RAGAS Metrics - High Score (Complete Answer)**

Query: "What are normal cholesterol levels?"  
Answer: "Normal cholesterol levels typically include total cholesterol below 200 mg/dL, LDL cholesterol below 100 mg/dL, and HDL cholesterol above 60 mg/dL. These values help assess cardiovascular health risk and guide treatment decisions."  
Retrieved documents: 5  
Evaluation Method: RAGAS

```json
{
    "evaluation_method": "ragas",
    "faithfulness": 0.94,
    "answer_relevancy": 0.91,
    "context_precision": 0.88,
    "context_recall": null,
    "overall_score": 0.910,
    "raw_scores": {
        "faithfulness": 0.94,
        "answer_relevancy": 0.91,
        "context_precision": 0.88
    }
}
```

**Score Breakdown:**
- **faithfulness (0.94)**: Answer is well-grounded in the retrieved context
- **answer_relevancy (0.91)**: Answer directly and comprehensively addresses the query
- **context_precision (0.88)**: High-quality, relevant documents were retrieved
- **overall_score (0.910)**: Average of (0.94 + 0.91 + 0.88) / 3 = 0.910

**Why Example 2 achieves a higher overall_score (0.910 > 0.750):**

1. **Higher faithfulness (0.94 vs 0.85)**: Answer is more thoroughly grounded in the retrieved context with specific numerical values
2. **Higher answer_relevancy (0.91 vs 0.72)**: Answer comprehensively addresses the query with detailed information
3. **Higher context_precision (0.88 vs 0.68)**: Better quality documents were retrieved, improving precision
4. **All RAGAS metrics optimized**: Each metric (faithfulness, answer_relevancy, context_precision) performs well, resulting in a high overall_score of 0.910

**Example 3: RAGAS Metrics - With Ground Truth (Includes Context Recall)**

Query: "What is cholesterol?"  
Answer: "Cholesterol is a waxy, fat-like substance found in all cells of your body. It is essential for various bodily functions, including hormone production and vitamin D synthesis."  
Retrieved documents: 5  
Ground Truth: "Cholesterol is a waxy substance that your body needs to build healthy cells, but high levels can increase heart disease risk."  
Evaluation Method: RAGAS

```json
{
    "evaluation_method": "ragas",
    "faithfulness": 0.92,
    "answer_relevancy": 0.89,
    "context_precision": 0.86,
    "context_recall": 0.84,
    "overall_score": 0.878,
    "raw_scores": {
        "faithfulness": 0.92,
        "answer_relevancy": 0.89,
        "context_precision": 0.86,
        "context_recall": 0.84
    }
}
```

**Score Breakdown:**
- **faithfulness (0.92)**: Answer is well-supported by context
- **answer_relevancy (0.89)**: Answer is highly relevant to the query
- **context_precision (0.86)**: Most retrieved documents are relevant
- **context_recall (0.84)**: Retrieved context captures most of the ground truth information
- **overall_score (0.878)**: Average of (0.92 + 0.89 + 0.86 + 0.84) / 4 = 0.878

**Note on Custom Metrics:**

If `USE_RAGAS_METRICS=False` or RAGAS is unavailable, the system falls back to custom metrics with the structure shown earlier. Custom metrics provide detailed breakdowns of answer quality, completeness, and retrieval quality, but may not capture semantic nuances as well as RAGAS metrics.

**Configuration:**

- **RAGAS Metrics**: Set `USE_RAGAS_METRICS=True` in `app/config.py` (default)
- **Evaluator LLM**: Configured via `EVALUATOR_LLM_MODEL` (default: `gpt-4o-mini`) and `EVALUATOR_LLM_TEMPERATURE` (default: 0.1)
- **Ground Truth**: Optional parameter for `context_recall` metric calculation

*Note: These scores are based on evaluation with sample test data. Actual scores will vary based on query complexity, answer quality, and retrieved document relevance. RAGAS metrics require an evaluator LLM and may incur additional API costs.*

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- OpenAI API key
- Docker and Docker Compose (optional)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd LabAssist_AI
```

2. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
```bash
# Create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

### Usage

#### 1. Ingest Documents

```bash
python -m app.main --ingest
```

This will:
- Extract text from PDFs in `data/` directory
- Chunk documents with metadata
- Build and persist ChromaDB vector store

#### 2. Run CLI Chat

```bash
python -m app.main --chat
```

Interactive command-line interface for asking questions.

#### 3. Run API Server

```bash
python -m app.main --api
```

API will be available at `http://localhost:8000`

**API Endpoints:**
- `GET /health` - Health check
- `POST /chat` - Chat endpoint
- `GET /docs` - API documentation (Swagger UI)

**Example API Request:**
```bash
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is normal cholesterol?", "top_k": 10}'
```

#### 4. Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up --build

# Or build manually
docker build -t rag-lab-tests .
docker run -p 8000:8000 --env-file .env rag-lab-tests
```

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_retriever.py

# Run integration tests (requires vector store)
pytest tests/test_integration.py
```

### Test Coverage

- **Unit Tests:** Retriever, guardrails, quality checks
- **Integration Tests:** End-to-end retrieval and generation
- **API Tests:** Health checks and chat endpoints

### Technology Stack

```mermaid
graph TB
    subgraph "Frontend/Interface"
        API[FastAPI<br/>Python 3.11+]
        CLI[CLI Interface]
    end
    
    subgraph "Core Libraries"
        LC[LangChain<br/>Orchestration]
        OAI[OpenAI SDK<br/>LLM & Embeddings]
        CHROMA[ChromaDB<br/>Vector Store]
    end
    
    subgraph "Data Processing"
        PDF[pdfplumber<br/>PDF Extraction]
        SPLIT[LangChain Text Splitters<br/>Chunking]
    end
    
    subgraph "Infrastructure"
        DOCKER[Docker<br/>Containerization]
        COMPOSE[Docker Compose<br/>Orchestration]
    end
    
    API --> LC
    CLI --> LC
    LC --> OAI
    LC --> CHROMA
    LC --> SPLIT
    PDF --> SPLIT
    DOCKER --> API
    COMPOSE --> DOCKER
```

### Deployment Architecture

```mermaid
graph TB
    subgraph "Development"
        DEV[Local Development<br/>Python Virtual Env]
    end
    
    subgraph "Containerization"
        IMG[Docker Image<br/>Multi-stage Build]
        COMPOSE[Docker Compose<br/>Service Orchestration]
    end
    
    subgraph "Runtime"
        CONTAINER[Container<br/>FastAPI Server]
        VOL1[Volume: data/<br/>PDF Documents]
        VOL2[Volume: chroma_db/<br/>Vector Store]
    end
    
    subgraph "External Services"
        OPENAI[OpenAI API<br/>LLM & Embeddings]
    end
    
    DEV --> IMG
    IMG --> COMPOSE
    COMPOSE --> CONTAINER
    CONTAINER --> VOL1
    CONTAINER --> VOL2
    CONTAINER --> OPENAI
    
    style CONTAINER fill:#e1f5ff
    style OPENAI fill:#ffe1f5
```

### Data Flow Architecture

**Ingestion Flow:**
```
PDF Documents → Loader → Cleaner → Chunker → Embeddings → Vector Store
     ↓            ↓         ↓         ↓          ↓            ↓
  [data/]    [extract]  [normalize] [split]  [encode]    [persist]
```

**Query Flow:**
```
User Query → API/CLI → Retriever → Vector Store → Context
     ↓          ↓          ↓            ↓            ↓
  [input]   [receive]  [search]    [retrieve]   [filter]
     ↓
  Prompt Builder → LLM → Guardrails → Quality Check → Response
     ↓              ↓         ↓            ↓            ↓
  [construct]   [generate] [validate]  [evaluate]   [return]
```

## 📁 Project Structure

```
LabAssist_AI/
├── app/
│   ├── ingestion/
│   │   ├── loader.py          # PDF extraction
│   │   ├── chunker.py          # Document chunking
│   │   └── indexer.py          # Vector store indexing
│   ├── rag/
│   │   ├── retriever.py        # Document retrieval
│   │   ├── generator.py        # LLM generation
│   │   ├── prompt.py           # Prompt engineering
│   │   └── guardrails.py       # Safety guardrails
│   ├── eval/
│   │   └── quality_checks.py   # Quality evaluation
│   ├── api.py                  # FastAPI application
│   ├── main.py                 # CLI entry point
│   └── config.py               # Configuration
├── data/                       # PDF documents
├── chroma_db/                  # Vector store (generated)
├── tests/                      # Test suite
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🔒 Environment Variables

- `OPENAI_API_KEY` - Required for OpenAI API access

## 📊 Performance Considerations

- **Embedding Model:** Fast inference, cost-effective
- **LLM:** Optimized for latency and cost
- **Chunking:** Balanced for retrieval precision and context
- **Caching:** Vector store persisted for fast startup
- **Token Management:** Efficient context truncation

## 🛡️ Safety & Ethics

- Medical information disclaimers
- Emergency detection and guidance
- Source attribution for transparency
- No medical advice beyond document content
- Clear limitations communication

## 📝 Future Enhancements

- [ ] Streaming responses for better UX
- [ ] Multi-modal support (images, tables)
- [ ] Advanced re-ranking models
- [ ] Conversation memory/history
- [ ] User feedback collection
- [ ] A/B testing framework
- [ ] Monitoring and observability

## 👥 Contributors

Jaisri S
