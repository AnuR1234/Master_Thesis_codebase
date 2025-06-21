# -*- coding: utf-8 -*-
"""
FINAL FIXED retriever module with comprehensive debugging and enhanced RAG capabilities.

This module provides a sophisticated document retrieval system for RAG (Retrieval-
Augmented Generation) applications. It implements hybrid retrieval combining dense
and sparse embeddings, cross-encoder reranking, and comprehensive debugging
capabilities for robust document search and ranking.

Key Features:
    - Hybrid retrieval using dense (E5) and sparse (BM25) embeddings
    - Cross-encoder reranking for improved relevance scoring
    - Multi-collection support for different document types
    - Comprehensive error handling and fallback mechanisms
    - Detailed logging and debugging information
    - GPU acceleration for embedding generation and reranking
    - Dynamic collection switching for different search domains

Classes:
    RAGRetriever: Main retrieval class with hybrid search and reranking

Functions:
    None (all functionality encapsulated in RAGRetriever class)

Dependencies:
    - qdrant-client: Vector database client for similarity search
    - sentence-transformers: Cross-encoder models for reranking
    - numpy: Numerical computing for embedding operations
    - Custom modules: config, embedding (E5EmbeddingModel, SparseEmbeddingHelper)

Vector Database Integration:
    - Qdrant vector database for similarity search
    - Named vectors for dense embeddings (E5)
    - Sparse vectors for BM25-style keyword matching
    - RRF (Reciprocal Rank Fusion) for combining search results

Search Strategies:
    - Dense-only: E5 embeddings with cosine similarity
    - Hybrid: Dense + Sparse with RRF fusion
    - Reranking: Cross-encoder scoring for final ranking

Example Usage:
    >>> retriever = RAGRetriever()
    >>> retriever.set_collection("classes")
    >>> results = retriever.retrieve("How to create ABAP class?", limit=5)
    >>> reranked = retriever.rerank_results("How to create ABAP class?", results)
"""
import logging
import time
import re
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import CrossEncoder
import os

from config import (
    QDRANT_HOST,
    QDRANT_PORT,
    COLLECTIONS,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_RETRIEVAL_LIMIT,
    EMBEDDING_MODEL,
    SPARSE_MODEL,
    CROSS_ENCODER_RERANKER
)

from embedding import E5EmbeddingModel, SparseEmbeddingHelper, FallbackEmbeddingWrapper, clear_gpu_memory

# Set up logging
logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    FINAL FIXED retriever with comprehensive debugging and hybrid search capabilities.
    
    This class implements a sophisticated document retrieval system that combines
    dense vector search, sparse keyword matching, and cross-encoder reranking
    to provide highly relevant search results for RAG applications.
    
    The retriever supports multiple search strategies and automatically falls back
    to simpler methods if advanced features are unavailable. It provides extensive
    logging and debugging information for troubleshooting and optimization.
    
    Attributes:
        client (QdrantClient): Vector database client for similarity search
        primary_embedder (E5EmbeddingModel): Dense embedding model for semantic search
        sparse_helper (SparseEmbeddingHelper): Sparse embedding helper for keyword search
        reranker (CrossEncoder): Cross-encoder model for result reranking
        current_collection (str): Currently active document collection
        _has_reranker (bool): Flag indicating reranker availability
        
    Search Modes:
        - Dense-only: Uses E5 embeddings for semantic similarity search
        - Hybrid: Combines dense and sparse embeddings with RRF fusion
        - Reranked: Applies cross-encoder scoring to improve relevance
        
    Example:
        >>> retriever = RAGRetriever()
        >>> retriever.set_collection("classes")
        >>> 
        >>> # Basic retrieval
        >>> results = retriever.retrieve("ABAP class methods", limit=10)
        >>> 
        >>> # Hybrid retrieval with reranking
        >>> results = retriever.retrieve("ABAP class methods", use_hybrid=True)
        >>> reranked = retriever.rerank_results("ABAP class methods", results, top_k=5)
    """
    
    def __init__(self):
        """
        Initialize FINAL FIXED retriever with comprehensive component setup.
        
        This method initializes all retrieval components including the Qdrant
        client, embedding models, sparse helpers, and cross-encoder reranker.
        It performs connection validation and component availability checks.
        
        Initialization Steps:
            1. Connect to Qdrant vector database
            2. Initialize dense embedding model (E5)
            3. Initialize sparse embedding helper (BM25)
            4. Initialize cross-encoder reranker
            5. Set default collection
            6. Validate collection availability
            
        Raises:
            ConnectionError: If Qdrant connection fails
            ModelError: If critical models cannot be loaded
        """
        logger.info("FINAL FIXED: Initializing RAG Retriever")
        
        # Initialize Qdrant client
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info(f"FINAL FIXED: Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        
        # Initialize components
        self._initialize_embeddings()
        self._initialize_reranker()
        
        # Default collection
        self.current_collection = DEFAULT_COLLECTION_NAME
        
        # Check collections
        self._check_collections()
    
    def _initialize_embeddings(self) -> None:
        """
        FINAL FIXED: Initialize dense and sparse embedding models.
        
        This method sets up both dense (E5) and sparse (BM25) embedding models
        with fallback mechanisms to ensure retrieval functionality even if
        some components fail to load.
        
        Dense Embeddings:
            - Primary: E5-Large-v2 for semantic understanding
            - Fallback: Basic embedding wrapper if E5 fails
            - Device: CUDA for GPU acceleration
            
        Sparse Embeddings:
            - BM25-style sparse vectors for keyword matching
            - Optional component with graceful degradation
            - Enhances hybrid search capabilities
        """
        device = "cuda"
        
        # Model paths for local loading
        model_paths = {
            "intfloat/e5-large-v2": "/home/user/rag_model/intfloat_e5-large-v2"
        }
        
        # Initialize dense embeddings
        try:
            local_path = model_paths.get(EMBEDDING_MODEL)
            model_to_load = local_path if local_path and os.path.exists(local_path) else EMBEDDING_MODEL
            
            self.primary_embedder = E5EmbeddingModel(
                model_name=model_to_load, 
                device=device
            )
            logger.info(f"FINAL FIXED: Dense embeddings initialized: {EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"FINAL FIXED: Could not initialize dense embeddings: {e}")
            self.primary_embedder = FallbackEmbeddingWrapper(dimension=1024)
        
        # Initialize sparse embeddings
        try:
            self.sparse_helper = SparseEmbeddingHelper(model_name=SPARSE_MODEL)
            if self.sparse_helper.is_available():
                logger.info(f"FINAL FIXED: Sparse embeddings initialized: {SPARSE_MODEL}")
            else:
                logger.warning("FINAL FIXED: Sparse embeddings not available")
        except Exception as e:
            logger.error(f"FINAL FIXED: Could not initialize sparse embeddings: {e}")
            self.sparse_helper = None
    
    def _initialize_reranker(self) -> None:
        """
        FINAL FIXED: Initialize cross-encoder reranker for result refinement.
        
        This method sets up the cross-encoder model used for reranking search
        results to improve relevance scoring. The reranker provides more
        accurate relevance assessment than initial retrieval scores.
        
        Reranker Features:
            - Cross-encoder architecture for query-document pair scoring
            - GPU acceleration for fast inference
            - Graceful degradation if model loading fails
            - Binary relevance scoring for ranking
        """
        try:
            self.reranker = CrossEncoder(CROSS_ENCODER_RERANKER, device="cuda")
            self._has_reranker = True
            logger.info(f"FINAL FIXED: Reranker initialized: {CROSS_ENCODER_RERANKER}")
        except Exception as e:
            logger.error(f"FINAL FIXED: Could not initialize reranker: {e}")
            self.reranker = None
            self._has_reranker = False
    
    def _check_collections(self) -> None:
        """
        FINAL FIXED: Check collection availability with detailed logging.
        
        This method validates that all configured document collections are
        available in the Qdrant database and logs their status for debugging.
        
        Collection Validation:
            - Queries Qdrant for available collections
            - Compares with configured collections from config
            - Logs availability status for each collection
            - Identifies missing collections for troubleshooting
        """
        try:
            collections = self.client.get_collections().collections
            available_collections = [c.name for c in collections]
            
            logger.info(f"FINAL FIXED: Available collections: {available_collections}")
            
            for collection_type, info in COLLECTIONS.items():
                collection_name = info["name"]
                if collection_name in available_collections:
                    logger.info(f"FINAL FIXED: ? Collection {collection_name} ({collection_type}) is available")
                else:
                    logger.error(f"FINAL FIXED: ? Collection {collection_name} ({collection_type}) is NOT available")
                    
        except Exception as e:
            logger.error(f"FINAL FIXED: Error checking collections: {e}")
    
    def set_collection(self, collection_type: str) -> str:
        """
        FINAL FIXED: Set the active document collection for retrieval.
        
        This method switches the retriever to use a different document collection,
        allowing searches across different types of documents (e.g., classes vs reports).
        
        Args:
            collection_type (str): Type of collection to switch to (e.g., "classes", "reports")
            
        Returns:
            str: Name of the currently active collection
            
        Collection Types:
            - "classes": SAP ABAP class documentation
            - "reports": SAP ABAP report documentation
            - Additional types as configured in config.py
            
        Example:
            >>> retriever.set_collection("classes")
            'abap_classes_collection'
            >>> retriever.set_collection("reports")  
            'abap_reports_collection'
        """
        if collection_type in COLLECTIONS:
            old_collection = self.current_collection
            self.current_collection = COLLECTIONS[collection_type]["name"]
            logger.info(f"FINAL FIXED: Changed collection from {old_collection} to {self.current_collection}")
            return self.current_collection
        else:
            logger.error(f"FINAL FIXED: Unknown collection type: {collection_type}")
            return self.current_collection
    
    @property
    def has_reranker(self) -> bool:
        """
        Check if cross-encoder reranker is available for use.
        
        Returns:
            bool: True if reranker is loaded and available, False otherwise
            
        Example:
            >>> if retriever.has_reranker:
            ...     reranked = retriever.rerank_results(query, results)
        """
        return self._has_reranker
    
    def retrieve(self, 
                query: str, 
                limit: int = DEFAULT_RETRIEVAL_LIMIT, 
                use_hybrid: bool = True,
                collection_name: Optional[str] = None) -> List[Any]:
        """
        FINAL FIXED: Comprehensive retrieval with detailed logging and fallback strategies.
        
        This method performs document retrieval using the most appropriate strategy
        based on available components. It attempts hybrid search first, then falls
        back to dense-only search if needed.
        
        Args:
            query (str): Search query text
            limit (int): Maximum number of results to return
            use_hybrid (bool): Whether to attempt hybrid (dense + sparse) search
            collection_name (Optional[str]): Specific collection to search, 
                                           defaults to current collection
        
        Returns:
            List[Any]: List of search results with scores and payloads
            
        Retrieval Strategy:
            1. Generate dense embedding for query
            2. If hybrid enabled and sparse available: perform hybrid search
            3. If hybrid fails or disabled: perform dense-only search
            4. Return results with comprehensive logging
            
        Search Methods:
            - Hybrid: Combines dense (E5) and sparse (BM25) with RRF fusion
            - Dense-only: E5 embeddings with cosine similarity
            - Fallback: Graceful degradation on component failures
            
        Example:
            >>> results = retriever.retrieve("ABAP class constructor", limit=10)
            >>> results = retriever.retrieve("method parameters", use_hybrid=False)
        """
        if collection_name is None:
            collection_name = self.current_collection
            
        logger.info(f"FINAL FIXED: ===== RETRIEVAL START =====")
        logger.info(f"FINAL FIXED: Query: '{query}'")
        logger.info(f"FINAL FIXED: Collection: {collection_name}")
        logger.info(f"FINAL FIXED: Limit: {limit}, Hybrid: {use_hybrid}")
        
        try:
            # Generate dense embedding
            logger.info(f"FINAL FIXED: Generating dense embedding...")
            dense_embedding = self._generate_dense_embedding(query)
            
            if dense_embedding is None:
                logger.error(f"FINAL FIXED: Failed to generate dense embedding")
                return []
            
            dense_vec = dense_embedding.tolist() if not isinstance(dense_embedding, list) else dense_embedding
            logger.info(f"FINAL FIXED: Dense embedding generated, shape: {len(dense_vec)}")
            
            # Try retrieval methods
            results = []
            
            # Try hybrid first if available
            if use_hybrid and self.sparse_helper is not None and self.sparse_helper.is_available():
                logger.info(f"FINAL FIXED: Attempting hybrid retrieval...")
                results = self._perform_hybrid_retrieval(query, dense_vec, collection_name, limit)
                
                if results and len(results) > 0:
                    logger.info(f"FINAL FIXED: Hybrid retrieval successful: {len(results)} results")
                else:
                    logger.warning(f"FINAL FIXED: Hybrid retrieval failed, trying dense...")
                    results = self._perform_dense_retrieval(query, dense_vec, collection_name, limit)
            else:
                logger.info(f"FINAL FIXED: Using dense-only retrieval...")
                results = self._perform_dense_retrieval(query, dense_vec, collection_name, limit)
            
            # Log final results
            logger.info(f"FINAL FIXED: ===== RETRIEVAL RESULTS =====")
            logger.info(f"FINAL FIXED: Total results: {len(results)}")
            
            for i, result in enumerate(results[:5]):
                title = result.payload.get("title", "No title")
                score = getattr(result, 'score', 0)
                logger.info(f"FINAL FIXED: Result {i+1}: '{title}' (score: {score:.4f})")
            
            logger.info(f"FINAL FIXED: ===== RETRIEVAL END =====")
            return results
                    
        except Exception as e:
            logger.error(f"FINAL FIXED: Error in retrieve: {e}", exc_info=True)
            return []

    def _generate_dense_embedding(self, query: str) -> Optional[np.ndarray]:
        """
        FINAL FIXED: Generate dense embedding with comprehensive error handling.
        
        This method generates dense vector embeddings for the input query using
        the E5 embedding model. It handles various model types and provides
        detailed error reporting.
        
        Args:
            query (str): Input query text to embed
            
        Returns:
            Optional[np.ndarray]: Dense embedding vector or None if generation fails
            
        Embedding Process:
            - Uses E5EmbeddingModel for semantic embeddings
            - Batch size 1 for single query processing
            - Returns numpy array for downstream processing
            - Comprehensive error handling and logging
            
        Example:
            >>> embedding = retriever._generate_dense_embedding("ABAP class")
            >>> print(f"Embedding shape: {embedding.shape}")
        """
        try:
            if isinstance(self.primary_embedder, E5EmbeddingModel):
                embedding = self.primary_embedder.encode([query], batch_size=1)[0]
            else:
                embedding = self.primary_embedder.encode([query], batch_size=1)[0]
            
            logger.info(f"FINAL FIXED: Dense embedding generated successfully")
            return embedding
            
        except Exception as e:
            logger.error(f"FINAL FIXED: Error generating dense embedding: {e}")
            return None

    def _perform_hybrid_retrieval(self, 
                                query: str, 
                                dense_vec: List[float], 
                                collection_name: str, 
                                limit: int) -> List[Any]:
        """
        FINAL FIXED: Hybrid retrieval combining dense and sparse search with comprehensive logging.
        
        This method implements hybrid search by combining dense vector similarity
        with sparse keyword matching using Reciprocal Rank Fusion (RRF) for
        optimal result ranking.
        
        Args:
            query (str): Search query text
            dense_vec (List[float]): Pre-computed dense embedding vector
            collection_name (str): Target collection for search
            limit (int): Maximum number of final results
            
        Returns:
            List[Any]: Hybrid search results with RRF-fused scores
            
        Hybrid Search Process:
            1. Generate sparse embedding from query text
            2. Create prefetch queries for both dense and sparse vectors
            3. Execute parallel searches with generous prefetch limits
            4. Apply RRF fusion to combine and rank results
            5. Return top results based on fused scores
            
        RRF Fusion:
            - Combines dense semantic similarity with sparse keyword matching
            - Provides balanced relevance scoring
            - Handles different score scales automatically
            
        Example:
            >>> results = retriever._perform_hybrid_retrieval(
            ...     "ABAP constructor", dense_vec, "classes", 10
            ... )
        """
        try:
            logger.info(f"FINAL FIXED: Generating sparse embedding for hybrid search...")
            sparse_vector = self.sparse_helper.generate_sparse_embedding(query)
            
            if not sparse_vector:
                logger.error(f"FINAL FIXED: Failed to generate sparse vector")
                return []
            
            logger.info(f"FINAL FIXED: Sparse vector generated: {len(sparse_vector.indices)} terms")
            
            # Build prefetch with generous limits
            prefetch_limit = limit * 8
            
            prefetch = [
                models.Prefetch(
                    query=models.SparseVector(
                        indices=sparse_vector.indices, 
                        values=sparse_vector.values
                    ),
                    using="BM25",
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=dense_vec,
                    using="E5",
                    limit=prefetch_limit,
                ),
            ]
            
            logger.info(f"FINAL FIXED: Executing hybrid search with RRF fusion...")
            logger.info(f"FINAL FIXED: Prefetch limit: {prefetch_limit}, Final limit: {limit}")
            
            response = self.client.query_points(
                collection_name=collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
            
            results = self._extract_results_from_response(response)
            
            if results and len(results) > 0:
                logger.info(f"FINAL FIXED: Hybrid search successful: {len(results)} results")
                return results
            else:
                logger.error(f"FINAL FIXED: Hybrid search returned no results")
                return []
                
        except Exception as e:
            logger.error(f"FINAL FIXED: Hybrid search failed: {e}", exc_info=True)
            return []

    def _perform_dense_retrieval(self, 
                            query: str, 
                            dense_vec: List[float], 
                            collection_name: str, 
                            limit: int) -> List[Any]:
        """
        FINAL FIXED: Dense-only retrieval using semantic embeddings with comprehensive logging.
        
        This method performs pure dense vector search using E5 embeddings and
        cosine similarity for semantic matching. It serves as the fallback
        method when hybrid search is unavailable or fails.
        
        Args:
            query (str): Search query text (for logging)
            dense_vec (List[float]): Pre-computed dense embedding vector
            collection_name (str): Target collection for search
            limit (int): Maximum number of results to return
            
        Returns:
            List[Any]: Dense search results with cosine similarity scores
            
        Dense Search Process:
            1. Create named vector for E5 embedding space
            2. Execute similarity search in vector database
            3. Return results ranked by cosine similarity
            4. Provide comprehensive logging and error handling
            
        Similarity Scoring:
            - Cosine similarity between query and document embeddings
            - Range: -1 to 1 (higher values indicate better matches)
            - Normalized scores for consistent ranking
            
        Example:
            >>> results = retriever._perform_dense_retrieval(
            ...     "ABAP method", dense_vec, "classes", 5
            ... )
        """
        try:
            logger.info(f"FINAL FIXED: Executing dense search...")
            logger.info(f"FINAL FIXED: Vector dimension: {len(dense_vec)}")
            
            # Create named vector
            named_vector = models.NamedVector(
                name="E5",
                vector=dense_vec
            )
            
            # Perform search
            results = self.client.search(
                collection_name=collection_name,
                query_vector=named_vector,
                limit=limit,
                with_payload=True
            )
            
            if results:
                logger.info(f"FINAL FIXED: Dense search successful: {len(results)} results")
                return results
            else:
                logger.error(f"FINAL FIXED: Dense search returned no results")
                return []
            
        except Exception as e:
            logger.error(f"FINAL FIXED: Dense search failed: {e}", exc_info=True)
            return []

    def _extract_results_from_response(self, response) -> Optional[List[Any]]:
        """
        FINAL FIXED: Extract search results from Qdrant response with flexible parsing.
        
        This method handles different response formats from Qdrant to extract
        the actual search results. It provides robust parsing for various
        response structures.
        
        Args:
            response: Qdrant response object with various possible structures
            
        Returns:
            Optional[List[Any]]: Extracted results list or None if parsing fails
            
        Response Parsing:
            - Handles direct result attributes
            - Supports nested response.body structures
            - Provides fallback for different Qdrant client versions
            - Logs extraction success/failure for debugging
            
        Example:
            >>> results = retriever._extract_results_from_response(qdrant_response)
            >>> print(f"Extracted {len(results)} results")
        """
        results = None
        
        if hasattr(response, 'result'):
            results = response.result
        elif hasattr(response, 'points'):
            results = response.points
        elif hasattr(response, 'body') and response.body:
            if hasattr(response.body, 'result'):
                results = response.body.result
            elif hasattr(response.body, 'points'):
                results = response.body.points
        
        logger.info(f"FINAL FIXED: Extracted {len(results) if results else 0} results from response")
        return results
    
    def rerank_results(self, 
                    query: str, 
                    results: List[Any], 
                    top_k: int = 5) -> List[Tuple[Any, float]]:
        """
        FINAL FIXED: Rerank search results using cross-encoder with comprehensive logging.
        
        This method applies cross-encoder reranking to improve the relevance
        ordering of search results. It provides more accurate relevance scores
        than initial retrieval by considering query-document interactions.
        
        Args:
            query (str): Original search query
            results (List[Any]): Search results to rerank
            top_k (int): Number of top results to return after reranking
            
        Returns:
            List[Tuple[Any, float]]: Reranked results as (result, score) tuples
            
        Reranking Process:
            1. Select more results than needed for reranking (4x top_k or 20 minimum)
            2. Apply cross-encoder scoring if available
            3. Sort results by new relevance scores
            4. Return top_k highest-scoring results
            5. Fallback to original scores if reranker unavailable
            
        Cross-Encoder Scoring:
            - Binary relevance classification between query and document
            - Considers semantic relationships and content relevance
            - Provides scores typically in range [0, 1]
            - GPU-accelerated inference for performance
            
        Example:
            >>> results = retriever.retrieve("ABAP class methods", limit=20)
            >>> reranked = retriever.rerank_results("ABAP class methods", results, top_k=5)
            >>> for result, score in reranked:
            ...     print(f"Title: {result.payload['title']}, Score: {score:.4f}")
        """
        if not results:
            logger.warning("FINAL FIXED: No results to rerank")
            return []
        
        logger.info(f"FINAL FIXED: ===== RERANKING START =====")
        logger.info(f"FINAL FIXED: Input results: {len(results)}")
        logger.info(f"FINAL FIXED: Target top_k: {top_k}")
        
        # Take more results for reranking
        results_to_rerank = results[:min(len(results), max(top_k * 4, 20))]
        logger.info(f"FINAL FIXED: Reranking {len(results_to_rerank)} results")
        
        if self.reranker is not None:
            reranked = self._rerank_with_cross_encoder(query, results_to_rerank, top_k)
            logger.info(f"FINAL FIXED: Reranking completed: {len(reranked)} results")
            return reranked
        else:
            logger.warning("FINAL FIXED: No reranker available")
            fallback = [(result, getattr(result, 'score', 0.0)) for result in results[:top_k]]
            return fallback

    def _rerank_with_cross_encoder(self, 
                                query: str, 
                                results: List[Any], 
                                top_k: int) -> List[Tuple[Any, float]]:
        """
        FINAL FIXED: Cross-encoder reranking with optimized text preparation and batch processing.
        
        This method implements the actual cross-encoder reranking by preparing
        query-document pairs, computing relevance scores, and ranking results
        by their cross-encoder scores.
        
        Args:
            query (str): Search query for comparison
            results (List[Any]): Results to rerank
            top_k (int): Number of top results to return
            
        Returns:
            List[Tuple[Any, float]]: Reranked (result, score) tuples
            
        Text Preparation:
            - Combines title, content, and code snippets
            - Truncates content to prevent memory issues
            - Creates structured text for cross-encoder input
            - Handles missing fields gracefully
            
        Batch Processing:
            - Processes multiple query-document pairs efficiently
            - Uses batch size 4 for optimal GPU utilization
            - Handles different score formats (tensor, list, scalar)
            - Provides timing information for performance monitoring
            
        Fallback Handling:
            - Graceful degradation on reranking failures
            - Uses original similarity scores as fallback
            - Maintains result ordering on errors
            
        Example:
            >>> reranked = retriever._rerank_with_cross_encoder(
            ...     "ABAP constructor method", results, top_k=5
            ... )
            >>> for result, score in reranked:
            ...     print(f"Cross-encoder score: {score:.4f}")
        """
        if not results:
            return []
        
        try:
            start_time = time.time()
            
            # Prepare pairs
            rerank_pairs = []
            for result in results:
                payload = result.payload
                
                # Build reranking text
                text_parts = []
                
                if payload.get('title'):
                    title = payload['title']
                    text_parts.append(f"Title: {title}")
                
                if payload.get('text'):
                    text = payload['text'][:2000]
                    text_parts.append(f"Content: {text}")
                
                if payload.get('code_snippet'):
                    code = payload['code_snippet'][:1500]
                    text_parts.append(f"Code: {code}")
                
                rerank_text = "\n".join(text_parts)
                rerank_pairs.append((query, rerank_text))
            
            logger.info(f"FINAL FIXED: Prepared {len(rerank_pairs)} rerank pairs")
            
            # Compute scores
            cross_encoder_scores = self.reranker.predict(rerank_pairs, batch_size=4)
            
            if hasattr(cross_encoder_scores, 'tolist'):
                cross_encoder_scores = cross_encoder_scores.tolist()
            elif not isinstance(cross_encoder_scores, list):
                cross_encoder_scores = [float(cross_encoder_scores)]
            
            # Create results
            final_results = []
            for result, score in zip(results, cross_encoder_scores):
                final_results.append((result, float(score)))
            
            # Sort by score
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            rerank_time = time.time() - start_time
            logger.info(f"FINAL FIXED: Reranking completed in {rerank_time:.2f}s")
            
            # Log reranked results
            for i, (result, score) in enumerate(final_results[:3]):
                title = result.payload.get("title", "No title")
                logger.info(f"FINAL FIXED: Reranked {i+1}: '{title}' (score: {score:.4f})")
            
            return final_results[:top_k]
            
        except Exception as e:
            logger.error(f"FINAL FIXED: Reranking failed: {e}", exc_info=True)
            # Fallback
            fallback = [(result, abs(getattr(result, 'score', 0.0))) for result in results]
            fallback.sort(key=lambda x: x[1], reverse=True)
            return fallback[:top_k]