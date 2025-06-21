#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Document Retrieval Module for RAG Pipeline.

This module provides comprehensive document retrieval capabilities for the RAG pipeline
with improved BM25 sparse embedding handling, hybrid search support, and robust
fallback mechanisms for various embedding and reranking scenarios.

The module implements:
- Dense vector retrieval using OpenAI embeddings or fallback
- Sparse vector retrieval using BM25 through FastEmbed
- Hybrid search combining dense and sparse vectors
- Cross-encoder reranking for improved result quality
- Robust error handling and fallback strategies
- Multiple collection support for different document types

Classes:
    RobustSparseEmbeddingHelper: Handles BM25 sparse embeddings with robust initialization
    DummySparseHelper: Fallback helper when BM25 is not available
    RAGRetriever: Main retriever class coordinating all retrieval operations

Author: SAP ABAP RAG Team
Version: 1.0.0
Date: 2025
License: MIT
"""

# Standard library imports
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party imports
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import CrossEncoder

# Local imports
from config import (
    COLLECTIONS,
    DEFAULT_COLLECTION_NAME,
    DEFAULT_RETRIEVAL_LIMIT,
    DENSE_VECTOR_WEIGHT,
    QDRANT_HOST,
    QDRANT_PORT,
    RERANKER_MODEL,
    SPARSE_VECTOR_WEIGHT,
)
from embedding import FallbackEmbeddingWrapper, OpenAIEmbeddingWrapper

# Import FastEmbed for sparse embeddings with graceful fallback
try:
    from fastembed import SparseTextEmbedding
    from qdrant_client.models import SparseVector
    FASTEMBED_AVAILABLE = True
except ImportError:
    FASTEMBED_AVAILABLE = False
    SparseTextEmbedding = None
    SparseVector = None

# Set up logging
logger = logging.getLogger(__name__)


class RobustSparseEmbeddingHelper:
    """
    Robust sparse embedding helper specifically designed for BM25 compatibility.
    
    This class provides robust initialization and handling of BM25 sparse embeddings
    using FastEmbed, with multiple fallback strategies to handle various failure modes
    including network issues, cache problems, and model loading failures.
    
    Features:
        - Multiple initialization strategies with exponential backoff
        - Comprehensive error handling and logging
        - Model testing and validation
        - Graceful degradation when BM25 is unavailable
        
    Attributes:
        model_name (str): Name of the sparse embedding model
        sparse_model: The FastEmbed SparseTextEmbedding instance
    """
    
    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        """
        Initialize the sparse embedding helper with BM25 model.
        
        Attempts multiple initialization strategies to ensure robust setup
        of the BM25 sparse embedding model.
        
        Args:
            model_name: Name of the sparse embedding model. Should remain
                       "Qdrant/bm25" for compatibility with existing vector data.
                       
        Note:
            The initialization tries three different strategies:
            1. Cache directory initialization
            2. Retry logic with exponential backoff
            3. Basic initialization as last resort
        """
        self.model_name = model_name
        self.sparse_model = None
        
        if not FASTEMBED_AVAILABLE:
            logger.error("FastEmbed not available - sparse embeddings disabled")
            return
        
        # Try multiple initialization approaches
        success = (self._try_initialize_with_cache() or 
                  self._try_initialize_with_retry() or 
                  self._try_initialize_basic())
        
        if success:
            logger.info(f"? BM25 sparse embedding model '{model_name}' initialized successfully")
        else:
            logger.error(f"? All BM25 initialization attempts failed for model '{model_name}'")

    def _try_initialize_with_cache(self) -> bool:
        """
        Try to initialize with explicit cache directory.
        
        This method attempts to initialize the BM25 model with an explicit
        cache directory, which can help resolve permission and path issues.
        
        Returns:
            True if initialization successful, False otherwise
            
        Note:
            Creates the cache directory if it doesn't exist and tests
            the model after initialization to ensure it's working properly.
        """
        try:
            logger.info("Attempting BM25 initialization with cache directory...")
            
            # Set explicit cache directory
            cache_dir = os.path.expanduser("~/.cache/fastembed")
            os.makedirs(cache_dir, exist_ok=True)
            
            self.sparse_model = SparseTextEmbedding(
                model_name=self.model_name,
                cache_dir=cache_dir,
                threads=1
            )
            
            # Test the model
            if self._test_model():
                logger.info("? BM25 initialized successfully with cache directory")
                return True
            else:
                self.sparse_model = None
                return False
                
        except Exception as e:
            logger.warning(f"Cache directory initialization failed: {e}")
            self.sparse_model = None
            return False

    def _try_initialize_with_retry(self) -> bool:
        """
        Try to initialize with retry logic for network issues.
        
        This method implements exponential backoff retry logic to handle
        transient network issues during model download or initialization.
        
        Returns:
            True if initialization successful on any attempt, False otherwise
            
        Note:
            Makes up to 3 attempts with exponential backoff delay between
            attempts to handle temporary network or service issues.
        """
        for attempt in range(3):
            try:
                logger.info(f"Attempting BM25 initialization (attempt {attempt + 1}/3)...")
                
                self.sparse_model = SparseTextEmbedding(
                    model_name=self.model_name,
                    threads=1
                )
                
                if self._test_model():
                    logger.info(f"? BM25 initialized successfully on attempt {attempt + 1}")
                    return True
                else:
                    self.sparse_model = None
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                self.sparse_model = None
                
                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(2 ** attempt)  # Exponential backoff
        
        return False

    def _try_initialize_basic(self) -> bool:
        """
        Try basic initialization as last resort.
        
        This method attempts the most basic initialization without any
        special parameters as a final fallback option.
        
        Returns:
            True if basic initialization successful, False otherwise
            
        Note:
            This is the last resort initialization method with minimal
            configuration to maximize chances of success.
        """
        try:
            logger.info("Attempting basic BM25 initialization...")
            
            self.sparse_model = SparseTextEmbedding(model_name=self.model_name)
            
            if self._test_model():
                logger.info("? BM25 basic initialization successful")
                return True
            else:
                self.sparse_model = None
                return False
                
        except Exception as e:
            logger.error(f"Basic initialization failed: {e}")
            self.sparse_model = None
            return False

    def _test_model(self) -> bool:
        """
        Test the model with a simple embedding to ensure it works.
        
        This method validates that the initialized model can actually
        generate embeddings by testing it with a simple text input.
        
        Returns:
            True if model passes test (can generate valid embeddings), False otherwise
            
        Note:
            Tests the model by generating an embedding for "test" and
            verifying the result has the expected structure and content.
        """
        try:
            if not self.sparse_model:
                return False
                
            test_result = list(self.sparse_model.embed(["test"]))
            if test_result and len(test_result) > 0:
                embedding = test_result[0]
                if hasattr(embedding, 'indices') and hasattr(embedding, 'values'):
                    return len(embedding.indices) > 0
            return False
        except Exception as e:
            logger.debug(f"Model test failed: {e}")
            return False

    def is_available(self) -> bool:
        """
        Check if sparse embedding model is available.
        
        Returns:
            True if the BM25 sparse embedding model is initialized and ready to use,
            False otherwise
            
        Examples:
            >>> helper = RobustSparseEmbeddingHelper()
            >>> if helper.is_available():
            ...     vector = helper.generate_sparse_embedding("query text")
        """
        return self.sparse_model is not None

    def process_sparse_vector(self, sparse_vec_result: Any) -> Optional[SparseVector]:
        """
        Process BM25 sparse vector from fastembed into Qdrant format.
        
        This method converts the output from FastEmbed's embed() method into
        the SparseVector format required by Qdrant for vector search operations.
        
        Args:
            sparse_vec_result: Result from FastEmbed embed() method, which is
                             typically a generator or list of SparseEmbedding objects
                             
        Returns:
            SparseVector in Qdrant format with indices and values, or None if
            processing fails due to invalid input or conversion errors
            
        Note:
            FastEmbed returns a generator of SparseEmbedding objects, each containing
            indices and values arrays. This method extracts the first embedding
            and converts it to Qdrant's SparseVector format.
            
        Examples:
            >>> helper = RobustSparseEmbeddingHelper()
            >>> result = helper.sparse_model.embed(["example text"])
            >>> vector = helper.process_sparse_vector(result)
            >>> print(f"Generated vector with {len(vector.indices)} features")
        """
        if sparse_vec_result is None:
            logger.warning("Received None sparse vector result")
            return None
            
        try:
            # FastEmbed embed() returns a generator, convert to list first
            if hasattr(sparse_vec_result, '__iter__'):
                sparse_list = list(sparse_vec_result)
                if not sparse_list:
                    logger.warning("Empty sparse vector list returned")
                    return None
                    
                # Take the first (and usually only) embedding
                sparse_embedding = sparse_list[0]
            else:
                sparse_embedding = sparse_vec_result
            
            # Extract indices and values from the SparseEmbedding object
            if hasattr(sparse_embedding, 'indices') and hasattr(sparse_embedding, 'values'):
                # Convert numpy arrays to Python lists if needed
                if hasattr(sparse_embedding.indices, 'tolist'):
                    indices = sparse_embedding.indices.tolist()
                else:
                    indices = list(sparse_embedding.indices)
                    
                if hasattr(sparse_embedding.values, 'tolist'):
                    values = sparse_embedding.values.tolist()
                else:
                    values = list(sparse_embedding.values)
                
                # Validate the data
                if len(indices) != len(values):
                    logger.error(f"Mismatch between indices ({len(indices)}) and values ({len(values)})")
                    return None
                    
                if len(indices) == 0:
                    logger.warning("Empty sparse vector (no features)")
                    return None
                
                # Create and return SparseVector
                return SparseVector(indices=indices, values=values)
                
            else:
                logger.error(f"Unexpected sparse embedding structure: {type(sparse_embedding)}")
                logger.error(f"Available attributes: {[attr for attr in dir(sparse_embedding) if not attr.startswith('_')]}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing BM25 sparse vector: {e}")
            logger.error(f"Input type: {type(sparse_vec_result)}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None

    def generate_sparse_embedding(self, text: str) -> Optional[SparseVector]:
        """
        Generate a BM25 sparse embedding for a text string.
        
        This method creates a BM25 sparse vector representation of the input text
        that can be used for sparse vector search in Qdrant.
        
        Args:
            text: Input text to embed. Should be non-empty and contain meaningful content
                 for best BM25 performance
                 
        Returns:
            SparseVector in Qdrant format containing BM25 term weights, or None if
            generation fails due to model unavailability, empty text, or processing errors
            
        Note:
            BM25 (Best Matching 25) is a ranking function used for information retrieval
            that assigns weights to terms based on their frequency and inverse document
            frequency characteristics.
            
        Examples:
            >>> helper = RobustSparseEmbeddingHelper()
            >>> vector = helper.generate_sparse_embedding("ABAP class implementation")
            >>> if vector:
            ...     print(f"Generated BM25 vector with {len(vector.indices)} terms")
        """
        if not self.is_available():
            logger.debug("BM25 sparse embedding model is not available")
            return None
            
        if not text or not text.strip():
            logger.warning("Empty or whitespace-only text provided")
            return None
            
        try:
            # Clean the text
            cleaned_text = text.strip()
            logger.debug(f"Generating BM25 sparse embedding for text: '{cleaned_text[:100]}...'")
            
            # Generate sparse embedding - embed() expects a list of texts
            sparse_result = self.sparse_model.embed([cleaned_text])
            
            # Process the result
            processed_vector = self.process_sparse_vector(sparse_result)
            
            if processed_vector is None:
                logger.warning(f"Failed to process BM25 sparse vector for text: '{cleaned_text[:50]}...'")
                return None
                
            logger.debug(f"Successfully generated BM25 sparse embedding with {len(processed_vector.indices)} features")
            return processed_vector
            
        except Exception as e:
            logger.error(f"Error generating BM25 sparse embedding for text '{text[:50]}...': {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            return None

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current sparse embedding model.
        
        Returns:
            Dictionary containing model information including:
            - model_name: Name of the BM25 model
            - is_available: Whether the model is ready for use
            - model_type: Type description of the model
            - fastembed_available: Whether FastEmbed library is available
            
        Examples:
            >>> helper = RobustSparseEmbeddingHelper()
            >>> info = helper.get_model_info()
            >>> print(f"Model: {info['model_name']}, Available: {info['is_available']}")
        """
        return {
            "model_name": self.model_name,
            "is_available": self.is_available(),
            "model_type": "BM25 SparseTextEmbedding",
            "fastembed_available": FASTEMBED_AVAILABLE
        }


class DummySparseHelper:
    """
    Dummy sparse helper for graceful fallback when BM25 is not available.
    
    This class provides a no-op implementation of the sparse embedding interface
    to ensure the system can continue operating even when BM25 sparse embeddings
    are not available due to library issues or initialization failures.
    
    Attributes:
        model_name (str): Always set to "unavailable" to indicate fallback mode
    """
    
    def __init__(self) -> None:
        """
        Initialize the dummy sparse helper.
        
        Creates a fallback helper that provides the same interface as
        RobustSparseEmbeddingHelper but always returns unavailable status.
        """
        self.model_name = "unavailable"
    
    def is_available(self) -> bool:
        """
        Check if sparse embedding model is available.
        
        Returns:
            Always False since this is a dummy fallback implementation
        """
        return False
    
    def generate_sparse_embedding(self, text: str) -> None:
        """
        Generate a sparse embedding for a text string.
        
        Args:
            text: Input text (ignored in dummy implementation)
            
        Returns:
            Always None since no sparse embedding is available
        """
        return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the sparse embedding model.
        
        Returns:
            Dictionary indicating this is a dummy fallback implementation
        """
        return {
            "model_name": self.model_name,
            "is_available": False,
            "model_type": "Dummy (Fallback)",
            "fastembed_available": FASTEMBED_AVAILABLE
        }


class RAGRetriever:
    """
    Retriever component for the RAG pipeline with robust BM25 sparse embedding support.
    
    This class coordinates all document retrieval operations for the RAG pipeline,
    including dense vector search, sparse vector search, hybrid search combining both,
    and cross-encoder reranking for improved result quality.
    
    Features:
        - Dense vector retrieval using OpenAI embeddings or fallback
        - Sparse vector retrieval using BM25 through FastEmbed
        - Hybrid search with multiple fallback strategies
        - Cross-encoder reranking for result optimization
        - Multi-collection support for different document types
        - Comprehensive error handling and graceful degradation
        
    Attributes:
        client (QdrantClient): Qdrant vector database client
        dense_embedder: Dense embedding model (OpenAI or fallback)
        sparse_helper: BM25 sparse embedding helper
        reranker: Cross-encoder model for reranking
        current_collection (str): Currently active collection name
        hybrid_search_supported (bool): Whether hybrid search is available
    """
    
    def __init__(self) -> None:
        """
        Initialize the retriever with necessary components.
        
        Sets up all required components for document retrieval including:
        - Qdrant client connection
        - Dense embedding models (OpenAI with fallback)
        - Sparse embedding models (BM25 with fallback)
        - Cross-encoder reranker (optional)
        - Collection validation and setup
        
        Raises:
            ConnectionError: If unable to connect to Qdrant
            RuntimeError: If no embedding models can be initialized
            
        Note:
            The initialization is designed to be robust with multiple fallback
            options. Even if some components fail, the retriever will attempt
            to continue with available functionality.
        """
        logger.info("Initializing RAG Retriever")
        
        # Initialize Qdrant client
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        
        # Check for hybrid search support
        self.hybrid_search_supported = self._check_hybrid_search_support()
        
        # Initialize sparse embedding helper - MUST use BM25 for compatibility with existing data
        logger.info("Initializing BM25 sparse embedding helper...")
        try:
            self.sparse_helper = RobustSparseEmbeddingHelper(model_name="Qdrant/bm25")
            
            if self.sparse_helper.is_available():
                logger.info("? BM25 sparse embedding model initialized successfully")
                # Test it with a simple query
                test_vector = self.sparse_helper.generate_sparse_embedding("test query")
                if test_vector:
                    logger.info(f"? BM25 model test passed - generated {len(test_vector.indices)} features")
                else:
                    logger.warning("? BM25 model test failed - sparse embedding generation not working")
            else:
                logger.warning("? BM25 sparse embedding model not available - will use dense-only retrieval")
                
        except Exception as e:
            logger.error(f"? Failed to initialize BM25 sparse embedding helper: {e}")
            # Create a dummy helper for graceful fallback
            self.sparse_helper = DummySparseHelper()
        
        # Initialize dense embedding models
        try:
            self.dense_embedder = OpenAIEmbeddingWrapper()
            logger.info("? Using OpenAI for dense embeddings")
        except Exception as e:
            logger.warning(f"Could not initialize OpenAI embeddings, using fallback: {e}")
            # Use fallback with same dimension as in the vector DB
            self.dense_embedder = FallbackEmbeddingWrapper(dimension=1536)
            logger.warning("? Using random embeddings as fallback - FOR TESTING ONLY")
        
        # Initialize reranker (optional)
        try:
            self.reranker = CrossEncoder(RERANKER_MODEL)
            self._has_reranker = True
            logger.info(f"? Loaded reranker model: {RERANKER_MODEL}")
        except Exception as e:
            logger.warning(f"Could not load reranker model: {e}")
            self._has_reranker = False
        
        # Default collection name
        self.current_collection = DEFAULT_COLLECTION_NAME
        
        # Check available collections
        self._check_collections()
        
        # Log final initialization status
        logger.info("=== RAG RETRIEVER INITIALIZATION SUMMARY ===")
        logger.info(f"Qdrant client: {'? Connected' if self.client else '? Not Connected'}")
        logger.info(f"Hybrid search support: {'? Available' if getattr(self, 'hybrid_search_supported', False) else '? Not Available (older Qdrant client)'}")
        logger.info(f"Dense embeddings: {'? Available' if self.dense_embedder else '? Not Available'}")
        logger.info(f"Sparse embeddings (BM25): {'? Available' if self.sparse_helper.is_available() else '? Not Available'}")
        logger.info(f"Reranker: {'? Available' if self._has_reranker else '? Not Available'}")
        logger.info(f"FastEmbed library: {'? Available' if FASTEMBED_AVAILABLE else '? Not Available'}")
        logger.info("===================================================")
            
    def _check_collections(self) -> None:
        """
        Check if the configured collections exist in Qdrant.
        
        Validates that all collections defined in the configuration are
        actually available in the connected Qdrant instance and logs
        their availability status.
        
        Note:
            This method only logs warnings for missing collections but
            does not prevent the retriever from functioning. Missing
            collections will cause runtime errors during retrieval.
        """
        try:
            collections = self.client.get_collections().collections
            available_collections = [c.name for c in collections]
            
            for collection_type, info in COLLECTIONS.items():
                collection_name = info["name"]
                if collection_name in available_collections:
                    logger.info(f"? Collection {collection_name} ({collection_type}) is available")
                else:
                    logger.warning(f"? Collection {collection_name} ({collection_type}) is NOT available in Qdrant")
        except Exception as e:
            logger.error(f"Error checking collections: {e}")
    
    def set_collection(self, collection_type: str) -> str:
        """
        Set the collection to use for retrieval.
        
        Changes the active collection for all subsequent retrieval operations
        based on the specified collection type.
        
        Args:
            collection_type: Type of collection to use ('classes' or 'reports').
                           Must be a key in the COLLECTIONS configuration.
                           
        Returns:
            The actual collection name that will be used for retrieval
            
        Examples:
            >>> retriever = RAGRetriever()
            >>> collection_name = retriever.set_collection("classes")
            >>> print(f"Now using collection: {collection_name}")
        """
        if collection_type in COLLECTIONS:
            self.current_collection = COLLECTIONS[collection_type]["name"]
            logger.info(f"Set current collection to {self.current_collection} ({collection_type})")
            return self.current_collection
        else:
            logger.warning(f"Unknown collection type: {collection_type}, using default")
            self.current_collection = DEFAULT_COLLECTION_NAME
            return self.current_collection
        
    @property
    def has_reranker(self) -> bool:
        """
        Check if reranker is available.
        
        Returns:
            True if cross-encoder reranker is loaded and available, False otherwise
        """
        return self._has_reranker
    
    def _check_hybrid_search_support(self) -> bool:
        """
        Check if the current Qdrant client supports hybrid search.
        
        Inspects the Qdrant client's search method to determine if it supports
        the parameters needed for hybrid search (combining dense and sparse vectors).
        
        Returns:
            True if hybrid search parameters are supported, False otherwise
            
        Note:
            This check is necessary because hybrid search support was added
            in newer versions of the Qdrant client library.
        """
        try:
            # Try to inspect the search method signature
            import inspect
            search_sig = inspect.signature(self.client.search)
            params = list(search_sig.parameters.keys())
            
            # Check for hybrid search parameters
            has_query_sparse_vector = 'query_sparse_vector' in params
            has_vector_weights = any('weight' in param.lower() for param in params)
            
            hybrid_supported = has_query_sparse_vector or has_vector_weights
            logger.info(f"Hybrid search support detected: {hybrid_supported}")
            return hybrid_supported
            
        except Exception as e:
            logger.warning(f"Could not detect hybrid search support: {e}")
            return False

    def retrieve(self, 
                query: str, 
                limit: int = DEFAULT_RETRIEVAL_LIMIT, 
                use_hybrid: bool = True,
                collection_name: Optional[str] = None) -> List[Any]:
        """
        Retrieve relevant documents from Qdrant with robust sparse embedding handling.
        
        This is the main retrieval method that coordinates dense vector search,
        sparse vector search, and hybrid search based on available capabilities
        and user preferences.
        
        Args:
            query: User query text to search for. Should be meaningful text
                  that represents the information need.
            limit: Number of results to retrieve. Must be positive integer.
            use_hybrid: Whether to use hybrid retrieval (dense + sparse) or
                       just dense vectors. Will fall back to dense-only if
                       sparse embeddings are not available.
            collection_name: Optional collection name to override current collection.
                           If None, uses the currently set collection.
                           
        Returns:
            List of retrieved documents with scores and metadata. Each document
            contains fields like id, score, payload with document content.
            
        Raises:
            ValueError: If query is empty or embedding generation fails
            ConnectionError: If Qdrant connection fails
            
        Examples:
            >>> retriever = RAGRetriever()
            >>> results = retriever.retrieve("ABAP class implementation", limit=5)
            >>> for result in results:
            ...     print(f"Document: {result.payload['title']}, Score: {result.score}")
        """
        # Use specified collection or current collection
        if collection_name is None:
            collection_name = self.current_collection
            
        logger.info(f"Retrieving documents for query: '{query}' from collection {collection_name}")
        logger.info(f"Parameters: limit={limit}, use_hybrid={use_hybrid}")
        
        # Generate embedding for the query (OpenAI or fallback)
        embedding = self.dense_embedder.encode([query])[0]
        
        # Check if embedding generation failed
        if embedding is None:
            logger.error("Failed to generate embedding for query")
            raise ValueError("Failed to generate embedding. Check API keys and embedding service.")
        
        # Create named vector properly formatted for Qdrant
        named_vector = models.NamedVector(
            name="OpenAI",
            vector=embedding
        )
        
        # Determine if we can use hybrid search
        can_use_hybrid = use_hybrid and self.sparse_helper.is_available()
        sparse_vector = None
        
        if can_use_hybrid:
            # Generate BM25 sparse vector for the query
            sparse_vector = self.sparse_helper.generate_sparse_embedding(query)
            
            # Check if sparse vector generation failed
            if sparse_vector is None:
                logger.warning("Failed to generate sparse vector, falling back to dense only")
                can_use_hybrid = False
            else:
                logger.debug(f"Generated sparse vector with {len(sparse_vector.indices)} features")
        
        # Execute search with appropriate method based on availability
        search_mode = "Dense only"  # Default
        
        try:
            if can_use_hybrid and sparse_vector is not None:
                # Try hybrid search first
                results = self._perform_hybrid_search(
                    collection_name, named_vector, sparse_vector, limit
                )
                search_mode = "Hybrid (Dense + Sparse)"
                
            else:
                # Fall back to dense-only search
                results = self._perform_dense_search(
                    collection_name, named_vector, limit
                )
                search_mode = "Dense only"
            
            # Log the results
            self._log_search_results(results, collection_name, search_mode)
            return results
            
        except Exception as e:
            logger.error(f"Error during retrieval from {collection_name}: {e}")
            
            # If hybrid search failed, try dense-only as fallback
            if can_use_hybrid:
                logger.warning("Hybrid search failed, falling back to dense-only search")
                try:
                    results = self._perform_dense_search(collection_name, named_vector, limit)
                    self._log_search_results(results, collection_name, "Dense only (fallback)")
                    return results
                except Exception as fallback_error:
                    logger.error(f"Dense-only fallback also failed: {fallback_error}")
                    raise
            else:
                raise

    def _perform_hybrid_search(self, collection_name: str, named_vector: models.NamedVector, 
                             sparse_vector: SparseVector, limit: int) -> List[Any]:
        """
        Perform hybrid search with multiple fallback strategies.
        
        Attempts hybrid search using multiple strategies to handle different
        Qdrant client versions and API variations.
        
        Args:
            collection_name: Name of the collection to search
            named_vector: Dense vector in Qdrant NamedVector format
            sparse_vector: Sparse vector in Qdrant SparseVector format
            limit: Maximum number of results to return
            
        Returns:
            List of search results from Qdrant
            
        Raises:
            Exception: If all hybrid search strategies fail
            
        Note:
            Tries three different hybrid search strategies:
            1. Modern API with vector weights
            2. Modern API without weights
            3. Manual combination of separate searches
        """
        # Strategy 1: Try modern hybrid search API
        try:
            named_sparse_vector = models.NamedSparseVector(
                name="BM25",
                vector=sparse_vector
            )
            
            search_params = {
                "collection_name": collection_name,
                "query_vector": named_vector,
                "query_sparse_vector": named_sparse_vector,
                "limit": limit,
                "with_payload": True,
                "vector_weights": {
                    "OpenAI": DENSE_VECTOR_WEIGHT,
                    "BM25": SPARSE_VECTOR_WEIGHT
                }
            }
            
            logger.debug("Attempting modern hybrid search API")
            return self.client.search(**search_params)
            
        except Exception as e:
            logger.debug(f"Modern hybrid search failed: {e}")
            
        # Strategy 2: Try hybrid search without vector_weights
        try:
            search_params = {
                "collection_name": collection_name,
                "query_vector": named_vector,
                "query_sparse_vector": named_sparse_vector,
                "limit": limit,
                "with_payload": True
            }
            
            logger.debug("Attempting hybrid search without weights")
            return self.client.search(**search_params)
            
        except Exception as e:
            logger.debug(f"Hybrid search without weights failed: {e}")
            
        # Strategy 3: Try separate searches and combine manually
        try:
            logger.debug("Attempting manual hybrid search combination")
            return self._manual_hybrid_search(collection_name, named_vector, sparse_vector, limit)
            
        except Exception as e:
            logger.debug(f"Manual hybrid search failed: {e}")
            raise Exception("All hybrid search strategies failed")

    def _perform_dense_search(self, collection_name: str, named_vector: models.NamedVector, 
                            limit: int) -> List[Any]:
        """
        Perform dense-only vector search.
        
        Executes a standard dense vector similarity search using only
        the dense embeddings (OpenAI or fallback).
        
        Args:
            collection_name: Name of the collection to search
            named_vector: Dense vector in Qdrant NamedVector format
            limit: Maximum number of results to return
            
        Returns:
            List of search results from Qdrant dense vector search
        """
        search_params = {
            "collection_name": collection_name,
            "query_vector": named_vector,
            "limit": limit,
            "with_payload": True
        }
        
        logger.debug("Performing dense vector search only")
        return self.client.search(**search_params)

    def _manual_hybrid_search(self, collection_name: str, named_vector: models.NamedVector, 
                            sparse_vector: SparseVector, limit: int) -> List[Any]:
        """
        Manually combine dense and sparse search results.
        
        Performs separate dense and sparse searches then combines the results
        manually using weighted scoring. This is a fallback for when the
        Qdrant client doesn't support native hybrid search.
        
        Args:
            collection_name: Name of the collection to search
            named_vector: Dense vector in Qdrant NamedVector format
            sparse_vector: Sparse vector in Qdrant SparseVector format
            limit: Maximum number of results to return
            
        Returns:
            List of combined and reranked search results
            
        Note:
            Combines results by adding weighted scores from dense and sparse
            searches. Documents appearing in both result sets get combined scores.
        """
        # Perform dense search
        dense_results = self._perform_dense_search(collection_name, named_vector, limit)
        
        # Try sparse search if the client supports it
        sparse_results = []
        try:
            # Create sparse-only search parameters
            named_sparse_vector = models.NamedSparseVector(
                name="BM25",
                vector=sparse_vector
            )
            
            sparse_search_params = {
                "collection_name": collection_name,
                "query_vector": named_sparse_vector,
                "limit": limit,
                "with_payload": True
            }
            
            sparse_results = self.client.search(**sparse_search_params)
            logger.debug(f"Sparse search returned {len(sparse_results)} results")
            
        except Exception as e:
            logger.debug(f"Sparse-only search failed: {e}")
            # If sparse search fails, just return dense results
            return dense_results
        
        # Combine and rerank results manually
        combined_results = self._combine_search_results(
            dense_results, sparse_results, limit
        )
        
        return combined_results

    def _combine_search_results(self, dense_results: List[Any], sparse_results: List[Any], 
                              limit: int) -> List[Any]:
        """
        Manually combine dense and sparse search results using simple scoring.
        
        Merges results from separate dense and sparse searches by combining
        their scores with configured weights.
        
        Args:
            dense_results: Results from dense vector search
            sparse_results: Results from sparse vector search
            limit: Maximum number of final results to return
            
        Returns:
            List of combined results sorted by combined score
            
        Note:
            Documents appearing in both result sets have their scores combined
            using the configured dense and sparse vector weights. New documents
            from either search are included with their weighted scores.
        """
        # Create a dictionary to store combined scores
        result_scores = {}
        all_results = {}
        
        # Process dense results
        for result in dense_results:
            doc_id = result.id
            dense_score = result.score * DENSE_VECTOR_WEIGHT
            result_scores[doc_id] = dense_score
            all_results[doc_id] = result
        
        # Process sparse results and combine scores
        for result in sparse_results:
            doc_id = result.id
            sparse_score = result.score * SPARSE_VECTOR_WEIGHT
            
            if doc_id in result_scores:
                # Combine scores
                result_scores[doc_id] += sparse_score
            else:
                # New result from sparse search
                result_scores[doc_id] = sparse_score
                all_results[doc_id] = result
        
        # Sort by combined score and return top results
        sorted_results = sorted(
            result_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:limit]
        
        # Create final result list with updated scores
        final_results = []
        for doc_id, combined_score in sorted_results:
            result = all_results[doc_id]
            # Update the score to reflect the combined score
            result.score = combined_score
            final_results.append(result)
        
        logger.debug(f"Combined {len(dense_results)} dense + {len(sparse_results)} sparse results into {len(final_results)} final results")
        return final_results

    def _log_search_results(self, results: List[Any], collection_name: str, search_mode: str) -> None:
        """
        Log search results in a standardized format.
        
        Provides detailed logging of search results including scores,
        document titles, and search statistics for debugging and monitoring.
        
        Args:
            results: List of search results from Qdrant
            collection_name: Name of the collection that was searched
            search_mode: Description of the search method used
            
        Note:
            Logs both to the Python logger and prints to stdout for
            immediate visibility during development and debugging.
        """
        logger.info(f"Retrieved {len(results)} documents from {collection_name}")
        print(f"\n==== RETRIEVAL STATS ====")
        print(f"Collection: {collection_name}")
        print(f"Search mode: {search_mode}")
        print(f"Total number of results retrieved: {len(results)}")
        
        # Count and log how many results have positive scores
        positive_scores = [r for r in results if r.score > 0]
        print(f"Number of results with positive scores: {len(positive_scores)}")
        
        # Log all the scores
        print("\nAll document scores:")
        for i, result in enumerate(results):
            doc_id = result.id
            score = result.score
            # Try to get a title or identifier from the payload if available
            title = result.payload.get('title', f'Doc {doc_id}')
            print(f"  {i+1}. {title}: {score}")
        
        print("==== END STATS ====\n")
    
    def rerank_results(self, 
                      query: str, 
                      results: List[Any], 
                      top_k: int = 5) -> List[Tuple[Any, float]]:
        """
        Rerank results using a cross-encoder.
        
        Applies cross-encoder reranking to improve the quality of retrieved
        results by considering the full interaction between query and document.
        
        Args:
            query: Original user query for context
            results: Results from vector retrieval to be reranked
            top_k: Number of top results to keep after reranking
            
        Returns:
            List of tuples (result, reranker_score) sorted by reranker score,
            limited to top_k results
            
        Note:
            Cross-encoder reranking considers the full query-document interaction
            rather than just vector similarity, often providing better relevance
            ranking especially for complex queries.
            
        Examples:
            >>> retriever = RAGRetriever()
            >>> results = retriever.retrieve("ABAP class methods")
            >>> reranked = retriever.rerank_results("ABAP class methods", results, top_k=3)
            >>> for result, score in reranked:
            ...     print(f"Document: {result.payload['title']}, Reranker Score: {score}")
        """
        if not self._has_reranker:
            logger.warning("Reranker not available, returning original results")
            return [(result, result.score) for result in results[:top_k]]
        
        logger.info(f"Reranking {len(results)} results, keeping top {top_k}")
        
        # Prepare input pairs for the reranker
        pairs = []
        for result in results:
            text = result.payload.get("text", "")
            pairs.append((query, text))
        
        # Get scores from the reranker
        try:
            scores = self.reranker.predict(pairs)
            
            # Create new result objects with reranker scores
            reranked_results = []
            for i, (result, score) in enumerate(zip(results, scores)):
                reranked_results.append((result, float(score)))
            
            # Sort by reranker score and keep top_k
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            
            # Log reranked scores
            print("\n==== RERANKER STATS ====")
            print(f"Reranked {len(results)} documents, keeping top {top_k}")
            print("\nReranked document scores:")
            for i, (result, score) in enumerate(reranked_results[:top_k]):
                doc_id = result.id
                # Try to get a title or identifier from the payload if available
                title = result.payload.get('title', f'Doc {doc_id}')
                print(f"  {i+1}. {title}: {score}")
            print("==== END RERANKER STATS ====\n")
            
            logger.info(f"Reranking complete, top score: {reranked_results[0][1] if reranked_results else 0}")
            return reranked_results[:top_k]
        except Exception as e:
            logger.error(f"Error during reranking: {e}")
            # Fall back to original ranking
            return [(result, result.score) for result in results[:top_k]]

    def get_retriever_info(self) -> Dict[str, Any]:
        """
        Get information about the retriever components.
        
        Returns a comprehensive summary of the retriever's configuration
        and capabilities for debugging and monitoring purposes.
        
        Returns:
            Dictionary containing:
            - dense_embedder: Type of dense embedding model used
            - sparse_helper: Information about sparse embedding model
            - reranker_available: Whether cross-encoder reranking is available
            - current_collection: Currently active collection name
            - qdrant_host: Qdrant server host
            - qdrant_port: Qdrant server port
            
        Examples:
            >>> retriever = RAGRetriever()
            >>> info = retriever.get_retriever_info()
            >>> print(f"Dense model: {info['dense_embedder']}")
            >>> print(f"Sparse available: {info['sparse_helper']['is_available']}")
        """
        return {
            "dense_embedder": type(self.dense_embedder).__name__,
            "sparse_helper": self.sparse_helper.get_model_info(),
            "reranker_available": self._has_reranker,
            "current_collection": self.current_collection,
            "qdrant_host": QDRANT_HOST,
            "qdrant_port": QDRANT_PORT
        }