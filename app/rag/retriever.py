from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


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
        
        # Retrieve with similarity scores
        results = self.vector_store.similarity_search_with_score(
            query,
            k=k * 2,  # Retrieve more initially for filtering
            filter=filter_dict
        )
        
        # Filter by similarity threshold
        # ChromaDB returns L2 (Euclidean) distance scores (lower is better)
        # Convert distance to similarity score (higher is better, normalized to 0-1)
        # For L2 distance, we normalize by converting to similarity using: 1 / (1 + distance)
        # This maps distance [0, inf] to similarity [1, 0]
        filtered_results = []
        distances_and_similarities = []
        filtered_out_docs = []
        for doc, distance in results:
            # Convert L2 distance to similarity score
            # Using 1 / (1 + distance) to map [0, inf] to [1, 0]
            similarity = 1 / (1 + distance)
            distances_and_similarities.append({"distance": distance, "similarity": similarity, "passes_threshold": similarity >= threshold})
            
            if similarity >= threshold:
                # Add similarity score to metadata
                doc.metadata['similarity_score'] = similarity
                doc.metadata['distance'] = distance
                filtered_results.append((doc, similarity))
            else:
                # Track filtered out documents to see if they contain relevant info
                content_lower = doc.page_content.lower()
                filtered_out_docs.append({
                    "similarity": similarity,
                    "contains_range": "range" in content_lower,
                    "contains_numbers": any(char.isdigit() for char in doc.page_content[:200]),
                    "content_preview": doc.page_content[:150]
                })
        
        # Sort by similarity (descending)
        filtered_results.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k after filtering
        final_docs = [doc for doc, _ in filtered_results[:k]]
        
        # Optional re-ranking (simple length-based + score combination)
        if self.enable_reranking and len(final_docs) > 1:
            final_docs = self._rerank(final_docs, query)
        
        return final_docs
    
    def _rerank(self, documents: List[Document], query: str) -> List[Document]:
        """
        Simple re-ranking strategy combining similarity score and content length.
        Longer, more detailed chunks that are still relevant are preferred.
        
        Args:
            documents: List of documents to re-rank
            query: Original query
            
        Returns:
            Re-ranked list of documents
        """
        scored_docs = []
        
        for doc in documents:
            similarity = doc.metadata.get('similarity_score', 0.5)
            content_length = len(doc.page_content)
            
            # Combine similarity and content quality
            # Prefer longer chunks (more context) but still highly relevant
            # Normalize content length (assuming max ~2000 chars per chunk)
            normalized_length = min(content_length / 2000.0, 1.0)
            
            # Weighted score: 70% similarity, 30% content length
            rerank_score = 0.7 * similarity + 0.3 * normalized_length
            
            scored_docs.append((doc, rerank_score))
        
        # Sort by rerank score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, _ in scored_docs]
    
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
