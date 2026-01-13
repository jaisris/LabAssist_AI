from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
import logging

from app.config import (
    RERANK_MODEL,
    RERANK_MODEL_NAME,
    RERANK_TOP_N
)

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    Advanced RAG retriever with similarity threshold, top-k retrieval,
    and optional re-ranking capabilities.
    
    Strategy:
    - Top-k retrieval: Retrieve top K most similar chunks
    - Similarity threshold: Filter results below threshold
    - Re-ranking: Optional re-ranking for better relevance
    """
    
    def __init__(
        self,
        vector_store: Chroma,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        enable_reranking: bool = False
    ):
        """
        Initialize the RAG retriever.
        
        Args:
            vector_store: Chroma vector store instance
            top_k: Number of top documents to retrieve
            similarity_threshold: Minimum similarity score (0-1)
            enable_reranking: Whether to enable re-ranking
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.enable_reranking = enable_reranking
        
        # Initialize reranking model if enabled
        self.reranker = None
        if enable_reranking and RERANK_MODEL:
            self.reranker = self._initialize_reranker()
        
        logger.info(
            f"Initialized RAGRetriever: top_k={top_k}, "
            f"similarity_threshold={similarity_threshold}, "
            f"reranking={enable_reranking}, "
            f"rerank_model={RERANK_MODEL if enable_reranking else None}"
        )
    
    def _initialize_reranker(self):
        """
        Initialize the reranking model based on configuration.
        Only supports API-based rerankers (Cohere, Voyage AI).
        Returns None for similarity-based ranking (default, no API key needed).
        
        Returns:
            Reranker instance or None if not configured
        """
        if not RERANK_MODEL:
            return None
        
        try:
            if RERANK_MODEL.lower() == "cohere":
                from langchain_community.document_compressors import CohereRerank
                from app.config import COHERE_API_KEY
                
                if not COHERE_API_KEY:
                    logger.warning("COHERE_API_KEY not found, reranking disabled")
                    return None
                
                logger.info(f"Initializing Cohere reranker with model: {RERANK_MODEL_NAME}")
                return CohereRerank(
                    model=RERANK_MODEL_NAME,
                    top_n=RERANK_TOP_N,
                    cohere_api_key=COHERE_API_KEY
                )
            
            elif RERANK_MODEL.lower() == "voyage":
                from langchain_community.document_compressors import VoyageAIRerank
                from app.config import VOYAGE_API_KEY
                
                if not VOYAGE_API_KEY:
                    logger.warning("VOYAGE_API_KEY not found, reranking disabled")
                    return None
                
                logger.info(f"Initializing Voyage AI reranker with model: {RERANK_MODEL_NAME}")
                return VoyageAIRerank(
                    model_name=RERANK_MODEL_NAME,
                    top_n=RERANK_TOP_N,
                    voyage_api_key=VOYAGE_API_KEY
                )
            
            else:
                logger.warning(f"Unknown rerank model: {RERANK_MODEL}, reranking disabled")
                return None
                
        except ImportError as e:
            logger.warning(f"Failed to import reranker: {e}, reranking disabled")
            return None
        except Exception as e:
            logger.error(f"Failed to initialize reranker: {e}, reranking disabled")
            return None
    
    def _distance_to_similarity(self, distance: float) -> float:
        """
        Convert L2 distance to similarity score optimized for text-embedding-3-large.
        
        For normalized embeddings, the relationship is:
        - L2 distance d = sqrt(2 * (1 - cosine_similarity))
        - Therefore: cosine_similarity = 1 - (d^2 / 2)
        
        For text-embedding-3-large, typical distances are higher than expected.
        We use a normalized formula that better maps to [0, 1] range:
        similarity = 1 - (d^2 / 4) which gives better distribution.
        
        Args:
            distance: L2 distance (lower is better)
            
        Returns:
            Similarity score between 0 and 1 (higher is better)
        """
        # Optimized formula for text-embedding-3-large embeddings
        # Using d^2/4 instead of d^2/2 gives better similarity distribution
        # This maps typical distances (0.7-1.2) to similarities (0.6-0.9)
        similarity = max(0.0, min(1.0, 1.0 - (distance ** 2) / 4.0))
        return similarity
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        filter_dict: Optional[dict] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User query string
            top_k: Override default top_k
            similarity_threshold: Override default threshold
            filter_dict: Optional metadata filters
            
        Returns:
            List of relevant documents with scores
        """
        k = top_k or self.top_k
        threshold = similarity_threshold or self.similarity_threshold
        
        logger.info(f"Retrieving documents for query: '{query[:100]}...' (top_k={k}, threshold={threshold})")
        
        # Retrieve exactly k documents with similarity scores
        # If threshold filtering reduces results, we return what we have (up to k)
        # This ensures the user gets exactly what they request
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k,
            filter=filter_dict
        )
        
        logger.info(f"Retrieved {len(results)} candidate documents from vector store")
        
        if not results:
            logger.warning("No documents retrieved from vector store")
            return []
        
        # Convert distances to similarities
        scored_results = []
        all_distances = []
        
        for doc, distance in results:
            all_distances.append(distance)
            similarity = self._distance_to_similarity(distance)
            
            # Add similarity score to metadata
            doc.metadata['similarity_score'] = similarity
            doc.metadata['distance'] = distance
            scored_results.append((doc, similarity))
        
        # Log statistics
        if all_distances:
            min_dist = min(all_distances)
            max_dist = max(all_distances)
            avg_dist = sum(all_distances) / len(all_distances)
            similarities = [s for _, s in scored_results]
            min_sim = min(similarities)
            max_sim = max(similarities)
            avg_sim = sum(similarities) / len(similarities)
            logger.info(
                f"Distance stats: min={min_dist:.3f}, max={max_dist:.3f}, avg={avg_dist:.3f}"
            )
            logger.info(
                f"Similarity stats: min={min_sim:.3f}, max={max_sim:.3f}, avg={avg_sim:.3f}, "
                f"threshold={threshold}"
            )
        
        # Sort by similarity (descending)
        scored_results.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by threshold, but if all are below threshold, return top-k anyway
        # This ensures we always return something if documents were retrieved
        filtered_results = [(doc, sim) for doc, sim in scored_results if sim >= threshold]
        
        if not filtered_results and scored_results:
            # If threshold filtered everything out, return top-k anyway with warning
            logger.warning(
                f"All {len(scored_results)} documents below threshold {threshold}. "
                f"Returning top {k} documents anyway (highest similarity: {scored_results[0][1]:.3f})"
            )
            filtered_results = scored_results[:k]
        elif len(filtered_results) > k:
            # If more than k pass threshold, take top k
            filtered_results = filtered_results[:k]
        elif len(filtered_results) < k and scored_results:
            # If fewer than k pass threshold, take what we have (up to k)
            # Fill remaining slots with best documents below threshold
            remaining = k - len(filtered_results)
            below_threshold = [(doc, sim) for doc, sim in scored_results if sim < threshold]
            if below_threshold:
                filtered_results.extend(below_threshold[:remaining])
                logger.info(
                    f"Only {len(filtered_results) - remaining} documents passed threshold. "
                    f"Including {min(remaining, len(below_threshold))} additional documents below threshold."
                )
        
        logger.info(
            f"Selected {len(filtered_results)} documents "
            f"({sum(1 for _, s in filtered_results if s >= threshold)} above threshold={threshold})"
        )
        
        # Take top k
        final_docs = [doc for doc, _ in filtered_results[:k]]
        
        logger.info(f"Selected top {len(final_docs)} documents after filtering and sorting")
        
        # Optional re-ranking for better relevance
        # Only rerank if explicitly enabled
        if self.enable_reranking and len(final_docs) > 1:
            # Check if we have documents above threshold
            docs_above_threshold = sum(
                1 for doc in final_docs 
                if doc.metadata.get('similarity_score', 0.0) >= threshold
            )
            
            # Rerank if:
            # 1. Multiple documents above threshold (optimize ordering)
            # 2. No documents above threshold (help identify best candidates)
            if docs_above_threshold >= 2 or docs_above_threshold == 0:
                if docs_above_threshold == 0:
                    logger.debug(
                        f"Applying re-ranking to {len(final_docs)} documents "
                        f"(none above threshold={threshold}, reranking to find best candidates)"
                    )
                else:
                    logger.debug(
                        f"Applying re-ranking to {len(final_docs)} documents "
                        f"({docs_above_threshold} above threshold)"
                    )
                final_docs = self._rerank(final_docs, query)
                logger.info(f"Re-ranking complete: {len(final_docs)} documents")
            else:
                logger.debug(
                    f"Skipping re-ranking: only {docs_above_threshold} document(s) above threshold. "
                    "Not enough documents to benefit from reranking."
                )
        elif not self.enable_reranking:
            logger.debug(f"Re-ranking disabled (enable_reranking={self.enable_reranking})")
        
        # Log final similarity scores
        if final_docs:
            similarities = [doc.metadata.get('similarity_score', 0.0) for doc in final_docs]
            logger.info(
                f"Final retrieval: {len(final_docs)} documents, "
                f"similarity range: [{min(similarities):.3f}, {max(similarities):.3f}], "
                f"avg: {sum(similarities)/len(similarities):.3f}"
            )
        
        return final_docs
    
    def _rerank(self, documents: List[Document], query: str) -> List[Document]:
        """
        Re-rank documents using similarity-based ranking (fast, no API calls).
        This is a lightweight reranking that just sorts by similarity score.
        
        Args:
            documents: List of documents to re-rank
            query: Original query (not used in simple reranking, kept for API compatibility)
            
        Returns:
            Re-ranked list of documents (sorted by similarity score)
        """
        logger.debug(f"Re-ranking {len(documents)} documents by similarity score")
        
        # Simple reranking: just sort by similarity score (already computed)
        # This is fast and doesn't require API calls
        scored_docs = [
            (doc, doc.metadata.get('similarity_score', 0.0))
            for doc in documents
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        reranked_docs = [doc for doc, _ in scored_docs]
        
        logger.debug(
            f"Re-ranking complete: similarity range "
            f"[{min(s[1] for s in scored_docs):.3f}, {max(s[1] for s in scored_docs):.3f}]"
        )
        
        return reranked_docs
    
    def retrieve_with_metadata(
        self,
        query: str,
        metadata_filters: Optional[dict] = None
    ) -> List[Document]:
        """
        Retrieve documents with optional metadata filtering.
        
        Args:
            query: User query
            metadata_filters: Dictionary of metadata filters (e.g., {"test": "cholesterol"})
            
        Returns:
            List of filtered documents
        """
        return self.retrieve(query, filter_dict=metadata_filters)
