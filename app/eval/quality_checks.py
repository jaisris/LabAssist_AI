from typing import List, Dict, Optional
from langchain_core.documents import Document


class QualityChecker:
    """
    Quality control and evaluation logic for RAG system.
    Evaluates answer quality, relevance, and completeness.
    """
    
    def __init__(self):
        """Initialize quality checker."""
        pass
    
    def evaluate_answer_quality(
        self,
        query: str,
        answer: str,
        context_docs: List[Document]
    ) -> Dict[str, any]:
        """
        Evaluate the quality of a generated answer.
        
        Args:
            query: Original query
            answer: Generated answer
            context_docs: Retrieved context documents
            
        Returns:
            Quality evaluation metrics
        """
        metrics = {
            "answer_length": len(answer),
            "word_count": len(answer.split()),
            "has_sources": len(context_docs) > 0,
            "num_sources": len(context_docs),
            "avg_similarity": 0.0,
            "completeness_score": 0.0,
            "relevance_score": 0.0
        }
        
        # Calculate average similarity
        if context_docs:
            similarities = [
                doc.metadata.get('similarity_score', 0.0)
                for doc in context_docs
            ]
            metrics["avg_similarity"] = sum(similarities) / len(similarities) if similarities else 0.0
        
        # Completeness: Check if answer addresses the query
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        common_words = query_words.intersection(answer_words)
        metrics["completeness_score"] = len(common_words) / len(query_words) if query_words else 0.0
        
        # Relevance: Based on similarity scores
        if context_docs:
            max_similarity = max(
                (doc.metadata.get('similarity_score', 0.0) for doc in context_docs),
                default=0.0
            )
            metrics["relevance_score"] = max_similarity
        
        # Overall quality score (weighted combination)
        metrics["quality_score"] = (
            0.3 * metrics["completeness_score"] +
            0.4 * metrics["relevance_score"] +
            0.2 * (1.0 if metrics["has_sources"] else 0.0) +
            0.1 * min(metrics["word_count"] / 100.0, 1.0)  # Prefer detailed answers
        )
        
        return metrics
    
    def check_answer_completeness(
        self,
        query: str,
        answer: str
    ) -> Dict[str, any]:
        """
        Check if answer is complete and addresses the query.
        
        Args:
            query: Original query
            answer: Generated answer
            
        Returns:
            Completeness check results
        """
        # Check for common incomplete indicators
        incomplete_indicators = [
            "i don't know",
            "i cannot",
            "i'm not able",
            "not available",
            "couldn't find"
        ]
        
        answer_lower = answer.lower()
        is_incomplete = any(indicator in answer_lower for indicator in incomplete_indicators)
        
        # Check if answer is too short
        is_too_short = len(answer.split()) < 15
        
        # Check if answer contains question words from query
        question_words = ["what", "how", "why", "when", "where", "who"]
        query_lower = query.lower()
        has_question_word = any(word in query_lower for word in question_words)
        
        return {
            "is_complete": not is_incomplete and not is_too_short,
            "is_incomplete": is_incomplete,
            "is_too_short": is_too_short,
            "has_question_word": has_question_word
        }
    
    def evaluate_retrieval_quality(
        self,
        query: str,
        retrieved_docs: List[Document],
        top_k: int = 5
    ) -> Dict[str, any]:
        """
        Evaluate the quality of document retrieval.
        
        Args:
            query: User query
            retrieved_docs: Retrieved documents
            top_k: Expected number of documents
            
        Returns:
            Retrieval quality metrics
        """
        metrics = {
            "num_retrieved": len(retrieved_docs),
            "expected_num": top_k,
            "retrieval_rate": len(retrieved_docs) / top_k if top_k > 0 else 0.0,
            "avg_similarity": 0.0,
            "min_similarity": 0.0,
            "max_similarity": 0.0
        }
        
        if retrieved_docs:
            similarities = [
                doc.metadata.get('similarity_score', 0.0)
                for doc in retrieved_docs
            ]
            metrics["avg_similarity"] = sum(similarities) / len(similarities)
            metrics["min_similarity"] = min(similarities)
            metrics["max_similarity"] = max(similarities)
        
        # Overall retrieval quality
        metrics["retrieval_quality_score"] = (
            0.5 * metrics["retrieval_rate"] +
            0.5 * metrics["avg_similarity"]
        )
        
        return metrics
    
    def run_full_evaluation(
        self,
        query: str,
        answer: str,
        context_docs: List[Document],
        top_k: int = 5
    ) -> Dict[str, any]:
        """
        Run comprehensive evaluation of RAG system performance.
        
        Args:
            query: User query
            answer: Generated answer
            context_docs: Retrieved context documents
            top_k: Expected number of documents
            
        Returns:
            Complete evaluation results
        """
        answer_quality = self.evaluate_answer_quality(query, answer, context_docs)
        completeness = self.check_answer_completeness(query, answer)
        retrieval_quality = self.evaluate_retrieval_quality(query, context_docs, top_k)
        
        return {
            "answer_quality": answer_quality,
            "completeness": completeness,
            "retrieval_quality": retrieval_quality,
            "overall_score": (
                0.4 * answer_quality["quality_score"] +
                0.3 * (1.0 if completeness["is_complete"] else 0.0) +
                0.3 * retrieval_quality["retrieval_quality_score"]
            )
        }
