from typing import List, Dict, Optional
from langchain_core.documents import Document
import re


class Guardrails:
    """
    Guardrails for RAG system:
    - Relevance checking
    - Fallback responses
    - Source attribution
    - Handling ambiguous queries
    - Safety checks
    """
    
    # Medical terms that should trigger caution
    MEDICAL_EMERGENCY_KEYWORDS = [
        "emergency", "urgent", "severe", "critical", "immediate",
        "chest pain", "heart attack", "stroke", "difficulty breathing"
    ]
    
    # Ambiguous query patterns
    AMBIGUOUS_PATTERNS = [
        r"^what$",
        r"^how$",
        r"^why$",
        r"^tell me$",
        r"^explain$"
    ]
    
    def __init__(self, min_relevance_score: float = 0.5):
        """
        Initialize guardrails.
        
        Args:
            min_relevance_score: Minimum similarity score to consider relevant
        """
        self.min_relevance_score = min_relevance_score
    
    def check_relevance(
        self,
        query: str,
        documents: List[Document]
    ) -> Dict[str, any]:
        """
        Check if retrieved documents are relevant to the query.
        
        Args:
            query: User query
            documents: Retrieved documents
            
        Returns:
            Dictionary with relevance check results
        """
        if not documents:
            result = {
                "is_relevant": False,
                "reason": "No documents retrieved",
                "max_similarity": 0.0
            }
            return result
        
        # Get max similarity score
        similarity_scores = [doc.metadata.get('similarity_score', 0.0) for doc in documents]
        max_similarity = max(similarity_scores, default=0.0)
        
        is_relevant = max_similarity >= self.min_relevance_score
        
        result = {
            "is_relevant": is_relevant,
            "reason": "Low similarity score" if not is_relevant else "Relevant documents found",
            "max_similarity": max_similarity,
            "num_docs": len(documents)
        }
        
        return result
    
    def check_ambiguous_query(self, query: str) -> Dict[str, any]:
        """
        Check if query is too ambiguous or vague.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with ambiguity check results
        """
        query_lower = query.lower().strip()
        
        # Check for very short queries
        if len(query_lower.split()) < 2:
            return {
                "is_ambiguous": True,
                "reason": "Query too short or vague",
                "suggestion": "Please provide more specific details about what you'd like to know."
            }
        
        # Check for ambiguous patterns
        for pattern in self.AMBIGUOUS_PATTERNS:
            if re.match(pattern, query_lower):
                return {
                    "is_ambiguous": True,
                    "reason": "Query pattern too vague",
                    "suggestion": "Please specify what you'd like to know. For example: 'What is normal cholesterol?' instead of just 'What?'"
                }
        
        return {
            "is_ambiguous": False,
            "reason": "Query is specific enough"
        }
    
    def check_medical_emergency(self, query: str) -> Dict[str, any]:
        """
        Check if query might indicate a medical emergency.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with emergency check results
        """
        query_lower = query.lower()
        
        for keyword in self.MEDICAL_EMERGENCY_KEYWORDS:
            if keyword in query_lower:
                return {
                    "potential_emergency": True,
                    "keyword": keyword,
                    "message": "If this is a medical emergency, please contact emergency services immediately. This assistant provides general information only and cannot provide emergency medical advice."
                }
        
        return {
            "potential_emergency": False
        }
    
    def get_fallback_response(
        self,
        query: str,
        relevance_check: Dict,
        ambiguity_check: Dict
    ) -> str:
        """
        Generate appropriate fallback response based on checks.
        
        Args:
            query: User query
            relevance_check: Results from relevance check
            ambiguity_check: Results from ambiguity check
            
        Returns:
            Fallback response message
        """
        if ambiguity_check.get("is_ambiguous"):
            return f"I need more information to help you. {ambiguity_check.get('suggestion', 'Please provide more details about your question.')}"
        
        if not relevance_check.get("is_relevant"):
            return "I couldn't find relevant information in the documents to answer your question. Please try:\n1. Rephrasing your question\n2. Using more specific terms\n3. Consulting a healthcare professional for personalized advice."
        
        return "I'm having trouble processing your request. Please try rephrasing your question or consult a healthcare professional."
    
    def add_source_attribution(
        self,
        answer: str,
        documents: List[Document]
    ) -> str:
        """
        Add source attribution to answer.
        
        Args:
            answer: Generated answer
            documents: Source documents
            
        Returns:
            Answer with source attribution
        """
        if not documents:
            return answer
        
        # Strip any existing source attribution from the answer to avoid duplicates
        # This ensures we always have clean, deduplicated sources
        answer_lower = answer.lower()
        answer_cleaned = answer
        
        # Find and remove existing sources sections (various formats)
        import re
        # Pattern to match "Sources:" or "**Sources:**" followed by content until end or new section
        sources_patterns = [
            r'\n\*\*Sources:\*\*.*$',  # **Sources:** ... (to end)
            r'\nSources:.*$',  # Sources: ... (to end)
            r'\*\*Sources:\*\*.*$',  # **Sources:** ... (to end, no newline)
            r'Sources:.*$',  # Sources: ... (to end, no newline)
        ]
        
        sources_removed = False
        for pattern in sources_patterns:
            if re.search(pattern, answer, re.IGNORECASE | re.MULTILINE | re.DOTALL):
                answer_cleaned = re.sub(pattern, '', answer_cleaned, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
                sources_removed = True
                break
        
        # Use cleaned answer
        answer = answer_cleaned.strip()
        
        # Extract unique sources using a set for proper deduplication
        sources_set = set()
        source_extractions = []
        for doc in documents:
            source = doc.metadata.get('source', 'Unknown')
            test = doc.metadata.get('test', '')
            source_extractions.append({"source": source, "test": test})
            if source != 'Unknown':
                source_str = f"{source}"
                if test:
                    source_str += f" ({test})"
                sources_set.add(source_str)  # Set automatically handles deduplication
        
        # Convert to list to maintain order (Python 3.7+ sets maintain insertion order)
        sources = list(sources_set)
        
        if sources:
            attribution = "\n\nSources: " + ", ".join(sources)
            return answer + attribution
        
        return answer
    
    def validate_response(self, answer: str) -> Dict[str, any]:
        """
        Validate generated response for quality and safety.
        
        Args:
            answer: Generated answer
            
        Returns:
            Validation results
        """
        issues = []
        
        # Check for very short answers
        if len(answer.split()) < 10:
            issues.append("Answer is very short, may lack detail")
        
        # Check for common error patterns
        error_patterns = [
            r"i don't know",
            r"i cannot",
            r"i'm not able",
            r"i don't have"
        ]
        
        answer_lower = answer.lower()
        for pattern in error_patterns:
            if re.search(pattern, answer_lower):
                issues.append("Answer contains uncertainty indicators")
                break
        
        # Check for medical advice disclaimers
        has_disclaimer = any(
            phrase in answer_lower
            for phrase in ["consult", "healthcare professional", "medical advice", "doctor"]
        )
        
        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "has_disclaimer": has_disclaimer,
            "length": len(answer)
        }
