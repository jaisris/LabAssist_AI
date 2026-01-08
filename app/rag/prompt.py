from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage


class PromptBuilder:
    """
    Prompt engineering module with system prompts, context management,
    and few-shot examples.
    """
    
    SYSTEM_PROMPT = """You are a helpful medical laboratory test assistant. Your role is to answer questions about laboratory test results, their meanings, normal ranges, and interpretations based on the provided context documents.

Guidelines:
1. Base your answers strictly on the provided context documents
2. If the context doesn't contain enough information, say so clearly
3. Provide accurate, clear, and concise answers
4. Include relevant numerical values and ranges when available
5. Cite the source document when referencing specific information
6. If asked about something not in the documents, politely decline and suggest consulting a healthcare professional

Always prioritize accuracy and patient safety. Never provide medical advice beyond what's in the context documents."""

    FEW_SHOT_EXAMPLES = [
        {
            "question": "What is a normal cholesterol level?",
            "answer": "According to the context, normal total cholesterol levels are typically below 200 mg/dL. LDL (bad) cholesterol should be below 100 mg/dL, and HDL (good) cholesterol should be 60 mg/dL or higher."
        },
        {
            "question": "What does HbA1c measure?",
            "answer": "HbA1c (hemoglobin A1c) measures your average blood sugar levels over the past 2-3 months. It's expressed as a percentage, with normal levels typically below 5.7%."
        }
    ]
    
    def __init__(self, max_context_tokens: int = 3000):
        """
        Initialize prompt builder.
        
        Args:
            max_context_tokens: Maximum tokens to use for context
        """
        self.max_context_tokens = max_context_tokens
    
    def build_prompt(
        self,
        query: str,
        context_docs: List[Document],
        include_examples: bool = False
    ) -> ChatPromptTemplate:
        """
        Build a chat prompt with context and optional few-shot examples.
        
        Args:
            query: User question
            context_docs: Retrieved context documents
            include_examples: Whether to include few-shot examples
            
        Returns:
            Formatted chat prompt template
        """
        # Build context from documents
        context_parts = []
        sources = []
        
        for i, doc in enumerate(context_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            test = doc.metadata.get('test', '')
            similarity = doc.metadata.get('similarity_score', 0.0)
            
            context_parts.append(f"[Document {i} - Source: {source}, Test: {test}, Relevance: {similarity:.2f}]")
            context_parts.append(doc.page_content)
            context_parts.append("")  # Separator
            
            if source not in sources:
                sources.append(source)
        
        context_text = "\n".join(context_parts)
        
        # Truncate context if too long (simple character-based estimation)
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = self.max_context_tokens * 4
        original_length = len(context_text)
        if len(context_text) > max_chars:
            context_text = context_text[:max_chars] + "\n\n[Context truncated due to length...]"
        
        # Build messages
        messages = []
        
        # System message
        system_content = self.SYSTEM_PROMPT
        if sources:
            system_content += f"\n\nAvailable sources: {', '.join(sources)}"
        messages.append(("system", system_content))
        
        # Few-shot examples (optional)
        if include_examples:
            for example in self.FEW_SHOT_EXAMPLES:
                messages.append(("human", example["question"]))
                messages.append(("assistant", example["answer"]))
        
        # Current context and query
        user_content = f"""Context Documents:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context documents above. If the context doesn't contain enough information to answer the question, please say so."""
        
        messages.append(("human", user_content))
        
        # Create prompt template
        prompt = ChatPromptTemplate.from_messages(messages)
        
        return prompt
    
    def count_tokens_estimate(self, text: str) -> int:
        """
        Estimate token count (rough approximation).
        
        Args:
            text: Text to count
            
        Returns:
            Estimated token count
        """
        # Rough estimate: 1 token ≈ 4 characters for English text
        return len(text) // 4
    
    def truncate_context(
        self,
        documents: List[Document],
        max_tokens: int
    ) -> List[Document]:
        """
        Truncate context documents to fit within token limit.
        
        Args:
            documents: List of documents
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated list of documents
        """
        truncated = []
        current_tokens = 0
        
        for doc in documents:
            doc_tokens = self.count_tokens_estimate(doc.page_content)
            
            if current_tokens + doc_tokens <= max_tokens:
                truncated.append(doc)
                current_tokens += doc_tokens
            else:
                # Try to fit partial document
                remaining_tokens = max_tokens - current_tokens
                if remaining_tokens > 100:  # Only if meaningful space left
                    partial_doc = Document(
                        page_content=doc.page_content[:remaining_tokens * 4],
                        metadata=doc.metadata
                    )
                    partial_doc.metadata['truncated'] = True
                    truncated.append(partial_doc)
                break
        
        return truncated
