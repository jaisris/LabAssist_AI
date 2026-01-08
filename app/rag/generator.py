from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.rag.prompt import PromptBuilder
from app.rag.retriever import RAGRetriever


class RAGGenerator:
    """
    RAG generator that combines retrieval and generation.
    Handles LLM selection, context management, and response generation.
    """
    
    def __init__(
        self,
        retriever: RAGRetriever,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        max_context_tokens: int = 3000
    ):
        """
        Initialize RAG generator.
        
        Args:
            retriever: RAGRetriever instance
            model_name: OpenAI model name (gpt-4o-mini for cost/performance balance)
            temperature: LLM temperature (lower = more deterministic)
            max_tokens: Maximum tokens for response
            max_context_tokens: Maximum tokens for context
        """
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.prompt_builder = PromptBuilder(max_context_tokens=max_context_tokens)
        self.output_parser = StrOutputParser()
        
        # Reasoning for model selection:
        # - gpt-4o-mini: Good balance of cost, performance, and latency
        # - Lower temperature for factual, consistent responses
        # - Sufficient context window for RAG use case
    
    def generate(
        self,
        query: str,
        top_k: Optional[int] = None,
        include_sources: bool = True,
        include_examples: bool = False
    ) -> dict:
        """
        Generate answer for a query using RAG.
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            include_sources: Whether to include source attribution
            include_examples: Whether to use few-shot examples
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Retrieve relevant documents
        context_docs = self.retriever.retrieve(query, top_k=top_k)
        
        if not context_docs:
            return {
                "answer": "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your query or consult a healthcare professional.",
                "sources": [],
                "metadata": {
                    "retrieved_docs": 0,
                    "model": self.llm.model_name
                }
            }
        
        # Truncate context if needed
        context_docs = self.prompt_builder.truncate_context(
            context_docs,
            self.prompt_builder.max_context_tokens
        )
        
        # Build prompt
        prompt = self.prompt_builder.build_prompt(
            query,
            context_docs,
            include_examples=include_examples
        )
        
        # Create chain
        chain = prompt | self.llm | self.output_parser
        
        # Generate response
        answer = chain.invoke({})
        
        # Extract sources
        sources = []
        if include_sources:
            for doc in context_docs:
                source_info = {
                    "source": doc.metadata.get('source', 'Unknown'),
                    "test": doc.metadata.get('test', ''),
                    "similarity_score": doc.metadata.get('similarity_score', 0.0)
                }
                if source_info not in sources:
                    sources.append(source_info)
        
        result_dict = {
            "answer": answer,
            "sources": sources,
            "context_docs": context_docs,  # Include for guardrails and source attribution
            "metadata": {
                "retrieved_docs": len(context_docs),
                "model": self.llm.model_name,
                "query": query
            }
        }
        
        return result_dict
    
    def stream_generate(self, query: str, top_k: Optional[int] = None):
        """
        Stream response generation (for real-time UI).
        
        Args:
            query: User question
            top_k: Number of documents to retrieve
            
        Yields:
            Response chunks
        """
        # Retrieve relevant documents
        context_docs = self.retriever.retrieve(query, top_k=top_k)
        
        if not context_docs:
            yield "I couldn't find any relevant information in the documents."
            return
        
        # Truncate context if needed
        context_docs = self.prompt_builder.truncate_context(
            context_docs,
            self.prompt_builder.max_context_tokens
        )
        
        # Build prompt
        prompt = self.prompt_builder.build_prompt(query, context_docs)
        
        # Create chain
        chain = prompt | self.llm | self.output_parser
        
        # Stream response
        for chunk in chain.stream({}):
            yield chunk
