from typing import List, Dict, Optional
from langchain_core.documents import Document
import logging

from app.config import (
    USE_RAGAS_METRICS,
    EVALUATOR_LLM_MODEL,
    EVALUATOR_LLM_TEMPERATURE
)

logger = logging.getLogger(__name__)


class QualityChecker:
    """
    Quality control and evaluation logic for RAG system.
    Evaluates answer quality, relevance, and completeness.
    Supports both RAGAS metrics and custom metrics.
    """
    
    def __init__(self):
        """Initialize quality checker."""
        self.use_ragas = USE_RAGAS_METRICS
        self.ragas_metrics = None
        
        if self.use_ragas:
            try:
                self._initialize_ragas()
            except Exception as e:
                logger.warning(f"Failed to initialize RAGAS metrics: {e}, falling back to custom metrics")
                self.use_ragas = False
    
    def _initialize_ragas(self):
        """Initialize RAGAS metrics and evaluator."""
        try:
            from ragas.metrics import (
                Faithfulness,
                AnswerRelevancy,
                ContextPrecision,
                ContextRecall
            )
            from ragas import evaluate
            from langchain_openai import ChatOpenAI
            from ragas.llms import LangchainLLMWrapper
            
            # Initialize evaluator LLM
            evaluator_llm = ChatOpenAI(
                model=EVALUATOR_LLM_MODEL,
                temperature=EVALUATOR_LLM_TEMPERATURE
            )
            llm_wrapper = LangchainLLMWrapper(evaluator_llm)
            
            # Initialize RAGAS metrics with LLM wrapper
            self.ragas_metrics = {
                'faithfulness': Faithfulness(llm=llm_wrapper),
                'answer_relevancy': AnswerRelevancy(llm=llm_wrapper),
                'context_precision': ContextPrecision(llm=llm_wrapper),
                'context_recall': ContextRecall(llm=llm_wrapper),
                'evaluate': evaluate,
                'llm': llm_wrapper
            }
            
            logger.info(f"RAGAS metrics initialized with evaluator LLM: {EVALUATOR_LLM_MODEL}")
            
        except ImportError as e:
            raise ImportError(
                f"RAGAS not installed. Install with: pip install ragas datasets. Error: {e}"
            )
        except Exception as e:
            raise Exception(f"Failed to initialize RAGAS: {e}")
    
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
        logger.debug(f"Evaluating answer quality for query: '{query[:50]}...'")
        
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
            logger.debug(f"Average similarity: {metrics['avg_similarity']:.3f}")
        
        # Completeness: Check if answer addresses the query
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        common_words = query_words.intersection(answer_words)
        metrics["completeness_score"] = len(common_words) / len(query_words) if query_words else 0.0
        logger.debug(f"Completeness score: {metrics['completeness_score']:.3f} ({len(common_words)}/{len(query_words)} words matched)")
        
        # Relevance: Based on similarity scores
        if context_docs:
            max_similarity = max(
                (doc.metadata.get('similarity_score', 0.0) for doc in context_docs),
                default=0.0
            )
            metrics["relevance_score"] = max_similarity
            logger.debug(f"Relevance score (max similarity): {metrics['relevance_score']:.3f}")
        
        # Overall quality score (weighted combination)
        metrics["quality_score"] = (
            0.3 * metrics["completeness_score"] +
            0.4 * metrics["relevance_score"] +
            0.2 * (1.0 if metrics["has_sources"] else 0.0) +
            0.1 * min(metrics["word_count"] / 100.0, 1.0)  # Prefer detailed answers
        )
        
        logger.info(
            f"Answer quality evaluation: quality_score={metrics['quality_score']:.3f}, "
            f"completeness={metrics['completeness_score']:.3f}, "
            f"relevance={metrics['relevance_score']:.3f}, "
            f"avg_similarity={metrics['avg_similarity']:.3f}"
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
        logger.debug(f"Evaluating retrieval quality for query: '{query[:50]}...' (top_k={top_k})")
        
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
            logger.debug(
                f"Similarity stats: min={metrics['min_similarity']:.3f}, "
                f"max={metrics['max_similarity']:.3f}, "
                f"avg={metrics['avg_similarity']:.3f}"
            )
        else:
            logger.warning("No documents retrieved for quality evaluation")
        
        # Overall retrieval quality
        metrics["retrieval_quality_score"] = (
            0.5 * metrics["retrieval_rate"] +
            0.5 * metrics["avg_similarity"]
        )
        
        logger.info(
            f"Retrieval quality: score={metrics['retrieval_quality_score']:.3f}, "
            f"retrieved={metrics['num_retrieved']}/{top_k} "
            f"(rate={metrics['retrieval_rate']:.2f}), "
            f"avg_similarity={metrics['avg_similarity']:.3f}"
        )
        
        return metrics
    
    def run_full_evaluation(
        self,
        query: str,
        answer: str,
        context_docs: List[Document],
        top_k: int = 5,
        ground_truth: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Run comprehensive evaluation of RAG system performance.
        Uses RAGAS metrics if enabled, otherwise falls back to custom metrics.
        
        Args:
            query: User query
            answer: Generated answer
            context_docs: Retrieved context documents
            top_k: Expected number of documents
            ground_truth: Optional ground truth answer (required for some RAGAS metrics)
            
        Returns:
            Complete evaluation results
        """
        logger.info(f"Running full evaluation for query: '{query[:50]}...'")
        
        if self.use_ragas and self.ragas_metrics:
            return self._evaluate_with_ragas(query, answer, context_docs, ground_truth)
        else:
            return self._evaluate_with_custom_metrics(query, answer, context_docs, top_k)
    
    def _evaluate_with_ragas(
        self,
        query: str,
        answer: str,
        context_docs: List[Document],
        ground_truth: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Evaluate using RAGAS metrics.
        
        Args:
            query: User query
            answer: Generated answer
            context_docs: Retrieved context documents
            ground_truth: Optional ground truth answer
            
        Returns:
            Evaluation results with RAGAS metrics
        """
        logger.info("Using RAGAS metrics for evaluation")
        
        try:
            from datasets import Dataset
            
            # Prepare context as list of strings
            contexts = [[doc.page_content for doc in context_docs]]
            
            # Prepare data for RAGAS evaluation
            data = {
                "question": [query],
                "answer": [answer],
                "contexts": contexts
            }
            
            # Add ground truth if provided (for context_recall)
            if ground_truth:
                data["ground_truth"] = [ground_truth]
            
            dataset = Dataset.from_dict(data)
            
            # Select metrics based on available data
            metrics = [
                self.ragas_metrics['faithfulness'],
                self.ragas_metrics['answer_relevancy'],
                self.ragas_metrics['context_precision']
            ]
            
            # Add context_recall if ground truth is available
            if ground_truth:
                metrics.append(self.ragas_metrics['context_recall'])
            
            # Run evaluation
            logger.debug("Running RAGAS evaluation...")
            results = self.ragas_metrics['evaluate'](
                dataset=dataset,
                metrics=metrics
            )
            
            # Extract scores
            scores = results.to_pandas().iloc[0].to_dict()
            
            # Calculate overall score (average of available metrics)
            available_scores = [v for k, v in scores.items() if isinstance(v, (int, float)) and 0 <= v <= 1]
            overall_score = sum(available_scores) / len(available_scores) if available_scores else 0.0
            
            logger.info(
                f"RAGAS evaluation complete: overall_score={overall_score:.3f}, "
                f"faithfulness={scores.get('faithfulness', 'N/A')}, "
                f"answer_relevancy={scores.get('answer_relevancy', 'N/A')}, "
                f"context_precision={scores.get('context_precision', 'N/A')}"
            )
            
            return {
                "evaluation_method": "ragas",
                "faithfulness": scores.get('faithfulness', None),
                "answer_relevancy": scores.get('answer_relevancy', None),
                "context_precision": scores.get('context_precision', None),
                "context_recall": scores.get('context_recall', None) if ground_truth else None,
                "overall_score": overall_score,
                "raw_scores": scores
            }
            
        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}, falling back to custom metrics")
            return self._evaluate_with_custom_metrics(query, answer, context_docs, len(context_docs))
    
    def _evaluate_with_custom_metrics(
        self,
        query: str,
        answer: str,
        context_docs: List[Document],
        top_k: int = 5
    ) -> Dict[str, any]:
        """
        Evaluate using custom metrics (fallback when RAGAS is not available).
        
        Args:
            query: User query
            answer: Generated answer
            context_docs: Retrieved context documents
            top_k: Expected number of documents
            
        Returns:
            Evaluation results with custom metrics
        """
        logger.info("Using custom metrics for evaluation")
        
        answer_quality = self.evaluate_answer_quality(query, answer, context_docs)
        completeness = self.check_answer_completeness(query, answer)
        retrieval_quality = self.evaluate_retrieval_quality(query, context_docs, top_k)
        
        overall_score = (
            0.4 * answer_quality["quality_score"] +
            0.3 * (1.0 if completeness["is_complete"] else 0.0) +
            0.3 * retrieval_quality["retrieval_quality_score"]
        )
        
        logger.info(
            f"Custom evaluation complete: overall_score={overall_score:.3f}, "
            f"answer_quality={answer_quality['quality_score']:.3f}, "
            f"completeness={completeness['is_complete']}, "
            f"retrieval_quality={retrieval_quality['retrieval_quality_score']:.3f}"
        )
        
        return {
            "evaluation_method": "custom",
            "answer_quality": answer_quality,
            "completeness": completeness,
            "retrieval_quality": retrieval_quality,
            "overall_score": overall_score
        }
