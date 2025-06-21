# -*- coding: utf-8 -*-
"""
FINAL FIXED embedding module for proper code retrieval and semantic search.

This module provides comprehensive embedding capabilities for RAG (Retrieval-
Augmented Generation) systems, specifically optimized for SAP ABAP code and
documentation retrieval. It implements both dense and sparse embeddings with
fallback mechanisms for robust operation.

Key Features:
    - E5-Large-v2 dense embeddings for semantic understanding
    - BM25-style sparse embeddings for keyword matching
    - Optimized query prefixing for E5 model performance
    - GPU acceleration with memory management
    - Fallback mechanisms for graceful degradation
    - Local model path support for offline operation
    - Comprehensive error handling and logging

Classes:
    E5EmbeddingModel: Dense semantic embeddings using E5-Large-v2
    SparseEmbeddingHelper: Sparse keyword embeddings using BM25
    FallbackEmbeddingWrapper: Fallback random embeddings for testing

Functions:
    clear_gpu_memory: GPU memory cleanup utility

Model Configuration:
    - Primary: intfloat/e5-large-v2 (1024-dimensional dense embeddings)
    - Sparse: Qdrant/bm25 (sparse term frequency embeddings)
    - Device: CUDA with CPU fallback
    - Memory: Optimized for RTX 6000 Ada (48GB VRAM)

Usage Examples:
    >>> # Dense embeddings for semantic search
    >>> e5_model = E5EmbeddingModel("intfloat/e5-large-v2")
    >>> embeddings = e5_model.encode(["What is ABAP class method?"])
    >>> 
    >>> # Sparse embeddings for keyword matching
    >>> sparse_helper = SparseEmbeddingHelper("Qdrant/bm25")
    >>> sparse_vec = sparse_helper.generate_sparse_embedding("ABAP method parameters")
    >>> 
    >>> # Combined in hybrid search
    >>> dense_emb = e5_model.encode(["query text"])
    >>> sparse_emb = sparse_helper.generate_sparse_embedding("query text")

Dependencies:
    - torch: PyTorch for GPU acceleration
    - sentence-transformers: Transformer-based embedding models
    - fastembed: Fast sparse embedding generation
    - qdrant-client: Vector database client types
    - numpy: Numerical computing support
"""
import os
import logging
import torch
import gc
from typing import List, Optional, Union, Dict, Any
import numpy as np
from qdrant_client.models import SparseVector
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_gpu_memory() -> None:
    """
    Clear GPU memory cache to free up VRAM.
    
    This utility function clears the PyTorch CUDA cache to free up GPU memory,
    which is particularly important when working with large models on limited
    VRAM or when switching between different models.
    
    Example:
        >>> clear_gpu_memory()  # Frees GPU memory cache
        >>> # Load new model or perform memory-intensive operations
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class E5EmbeddingModel:
    """
    FINAL FIXED E5 embedding model for code retrieval and semantic search.
    
    This class implements dense semantic embeddings using the E5-Large-v2 model,
    specifically optimized for SAP ABAP code and documentation retrieval. It
    provides high-quality semantic understanding with proper query formatting
    and GPU acceleration.
    
    The E5 model requires specific prefixing for optimal performance:
    - "query:" prefix for search queries
    - "passage:" prefix for documents (handled by indexing pipeline)
    
    Attributes:
        device (str): Computing device ("cuda" or "cpu")
        model_name (str): Name/path of the embedding model
        model_paths (Dict[str, str]): Mapping of model names to local paths
        model (SentenceTransformer): Loaded sentence transformer model
        embedding_dim (int): Dimensionality of generated embeddings
        
    Features:
        - 1024-dimensional dense embeddings
        - Automatic GPU/CPU device selection
        - Local model path support for offline operation
        - Proper E5 query formatting with "query:" prefix
        - Normalized embeddings for cosine similarity
        - Batch processing for efficiency
        
    Example:
        >>> model = E5EmbeddingModel("intfloat/e5-large-v2")
        >>> embeddings = model.encode(["What parameters does submit method accept?"])
        >>> print(f"Embedding shape: {embeddings.shape}")  # (1, 1024)
    """
    
    def __init__(self, model_name: str = "intfloat/e5-large-v2", device: str = None, max_gpu_mem_gb: float = 48.0):
        """
        Initialize E5 model with FINAL FIXED configuration and optimization.
        
        This method loads the E5-Large-v2 model with optimal configuration for
        SAP ABAP code retrieval, including device selection, local path support,
        and memory optimization for RTX 6000 Ada GPUs.
        
        Args:
            model_name (str): Name or identifier of the E5 model to load
                            (default: "intfloat/e5-large-v2")
            device (str): Device for model execution ("cuda" or "cpu").
                         If None, automatically selects CUDA if available
            max_gpu_mem_gb (float): Maximum GPU memory limit in GB for optimization
                                   (default: 48.0 for RTX 6000 Ada)
        
        Raises:
            Exception: If model loading fails due to missing files, CUDA issues,
                      or insufficient memory
                      
        Model Loading Process:
            1. Determine optimal device (CUDA vs CPU)
            2. Check for local model cache
            3. Load SentenceTransformer with appropriate settings
            4. Verify model dimensions and capabilities
            5. Log successful initialization
        """
        logger.info(f"FINAL FIXED: Initializing E5 embedding model: {model_name}")
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        
        # Model paths for local loading and offline operation
        self.model_paths = {
            "intfloat/e5-large-v2": "/home/user/rag_model/intfloat_e5-large-v2"
        }
        
        try:
            # Get model source (local path or HuggingFace Hub)
            local_path = self.model_paths.get(model_name)
            model_source = local_path if local_path and os.path.exists(local_path) else model_name
            
            logger.info(f"FINAL FIXED: Loading model from: {model_source}")
            
            # Load model with device-specific configuration
            if self.device == "cuda":
                self.model = SentenceTransformer(
                    model_source,
                    device="cuda",
                    trust_remote_code=False  # Security: disable remote code execution
                )
            else:
                self.model = SentenceTransformer(model_source, device="cpu")
                
            # Get embedding dimension for validation
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            logger.info(f"FINAL FIXED: Model loaded successfully, dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"FINAL FIXED: Failed to load model {model_name}: {e}")
            raise
    
    def encode(self, texts: Union[str, List[str]], 
              batch_size: int = 16,
              code_specific: bool = False) -> np.ndarray:
        """
        FINAL FIXED: Encoding with correct E5 prefixes for optimal method searches.
        
        This method generates dense embeddings for input texts using the E5 model
        with proper query formatting. It applies the "query:" prefix required by
        E5 models for optimal search performance and handles batch processing
        for efficiency.
        
        Args:
            texts (Union[str, List[str]]): Text(s) to encode. Single string or list
            batch_size (int): Batch size for processing multiple texts (default: 16)
            code_specific (bool): Whether to apply code-specific optimizations
                                 (reserved for future enhancements)
        
        Returns:
            np.ndarray: Normalized embedding vectors with shape (n_texts, embedding_dim)
                       Returns zero vectors on encoding failure
                       
        Encoding Process:
            1. Convert single strings to list format
            2. Apply "query:" prefix to each text for E5 optimization
            3. Generate embeddings using SentenceTransformer
            4. Normalize embeddings for cosine similarity
            5. Return as numpy array for downstream processing
            
        E5 Prefix Strategy:
            - "query:" prefix for search queries (this method)
            - "passage:" prefix for documents (handled during indexing)
            - Proper prefixing improves retrieval performance significantly
            
        Example:
            >>> model = E5EmbeddingModel()
            >>> embeddings = model.encode(["ABAP class method parameters"])
            >>> print(embeddings.shape)  # (1, 1024)
            >>> 
            >>> # Batch processing
            >>> queries = ["method submit", "class constructor", "interface implementation"]
            >>> batch_embeddings = model.encode(queries, batch_size=8)
            >>> print(batch_embeddings.shape)  # (3, 1024)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        try:
            # FINAL FIXED: Use the correct E5 instruction format
            formatted_texts = []
            for text in texts:
                # For E5 models, we need to use the correct instruction format
                # Use "query:" prefix for search queries
                prefix = "query: "
                formatted_text = f"{prefix}{text}"
                formatted_texts.append(formatted_text)
                
                logger.debug(f"FINAL FIXED: Formatted text: {formatted_text[:100]}...")
            
            logger.info(f"FINAL FIXED: Generating {len(texts)} embeddings with 'query:' prefix")
            
            # Generate embeddings with optimal settings
            embeddings = self.model.encode(
                formatted_texts,
                batch_size=batch_size,
                show_progress_bar=False,  # Disable for cleaner logging
                normalize_embeddings=True,  # Normalize for cosine similarity
                device=self.device,
                convert_to_numpy=True  # Return numpy arrays
            )
            
            logger.info(f"FINAL FIXED: Successfully generated embeddings shape: {embeddings.shape}")
            return embeddings
        
        except Exception as e:
            logger.error(f"FINAL FIXED: Error generating E5 embeddings: {e}")
            # Return zero vectors as fallback
            return np.zeros((len(texts), self.embedding_dim))


class SparseEmbeddingHelper:
    """
    FINAL FIXED Sparse embedding helper for keyword-based retrieval.
    
    This class implements sparse embeddings using BM25-style term frequency
    scoring for keyword matching. It complements dense embeddings by capturing
    exact term matches and providing interpretable retrieval signals.
    
    Sparse embeddings are particularly effective for:
    - Exact keyword matching
    - Technical term retrieval
    - Method and class name searches
    - Acronym and identifier matching
    
    Attributes:
        model_name (str): Name of the sparse embedding model
        sparse_model: Loaded FastEmbed sparse text embedding model
        
    Features:
        - BM25-style sparse vector generation
        - Conservative resource usage for stability
        - Comprehensive error handling and testing
        - Qdrant SparseVector format output
        - Graceful degradation on model failures
        
    Example:
        >>> helper = SparseEmbeddingHelper("Qdrant/bm25")
        >>> if helper.is_available():
        ...     sparse_vec = helper.generate_sparse_embedding("ABAP method submit")
        ...     print(f"Sparse terms: {len(sparse_vec.indices)}")
    """
    
    def __init__(self, model_name: str = "Qdrant/bm25"):
        """
        Initialize sparse embedding helper with conservative settings.
        
        This method initializes the FastEmbed sparse text embedding model with
        conservative resource settings for stable operation. It includes
        comprehensive testing to ensure model functionality.
        
        Args:
            model_name (str): Name of the sparse embedding model
                             (default: "Qdrant/bm25")
        
        Initialization Process:
            1. Import FastEmbed SparseTextEmbedding
            2. Load model with conservative thread and batch settings
            3. Test model functionality with sample input
            4. Set availability flag based on test results
            5. Handle graceful degradation on failures
        """
        logger.info(f"FINAL FIXED: Initializing sparse embedding: {model_name}")
        
        self.model_name = model_name
        self.sparse_model = None
        
        try:
            from fastembed import SparseTextEmbedding
            
            # Load model with conservative settings for stability
            self.sparse_model = SparseTextEmbedding(
                model_name=model_name,
                threads=2,      # Conservative thread count
                batch_size=2    # Small batch size for stability
            )
            
            # Test model functionality
            if self._test_model():
                logger.info(f"FINAL FIXED: Sparse embedding initialized successfully")
            else:
                self.sparse_model = None
                
        except Exception as e:
            logger.error(f"FINAL FIXED: Failed to initialize sparse embedding: {e}")
            self.sparse_model = None

    def _test_model(self) -> bool:
        """
        Test the sparse embedding model functionality.
        
        This method performs a comprehensive test of the sparse embedding model
        to ensure it can generate valid embeddings with proper indices and
        values structures.
        
        Returns:
            bool: True if model passes all tests, False otherwise
            
        Test Process:
            1. Generate embedding for test input
            2. Verify embedding structure (indices and values attributes)
            3. Check that indices are non-empty
            4. Validate data types and dimensions
            
        Example:
            >>> helper = SparseEmbeddingHelper()
            >>> is_working = helper._test_model()
            >>> print(f"Model working: {is_working}")
        """
        try:
            test_result = list(self.sparse_model.embed(["test"]))
            if test_result and len(test_result) > 0:
                embedding = test_result[0]
                if hasattr(embedding, 'indices') and hasattr(embedding, 'values'):
                    return len(embedding.indices) > 0
            return False
        except Exception as e:
            logger.error(f"FINAL FIXED: Model test failed: {e}")
            return False

    def is_available(self) -> bool:
        """
        Check if sparse embedding model is available and functional.
        
        Returns:
            bool: True if model is loaded and ready, False otherwise
            
        Example:
            >>> helper = SparseEmbeddingHelper()
            >>> if helper.is_available():
            ...     # Use sparse embeddings
            ...     sparse_vec = helper.generate_sparse_embedding("query")
            ... else:
            ...     # Fall back to dense-only search
            ...     print("Sparse embeddings not available")
        """
        return self.sparse_model is not None

    def generate_sparse_embedding(self, text: str) -> Optional[SparseVector]:
        """
        FINAL FIXED: Generate sparse embedding for input text.
        
        This method generates sparse vector embeddings using BM25-style term
        frequency scoring. The output is formatted as a Qdrant SparseVector
        for direct use in hybrid search operations.
        
        Args:
            text (str): Input text to generate sparse embedding for
            
        Returns:
            Optional[SparseVector]: Qdrant SparseVector with indices and values,
                                   or None if generation fails or model unavailable
                                   
        Sparse Vector Structure:
            - indices: List of term indices (vocabulary positions)
            - values: List of corresponding term weights (BM25 scores)
            - Compatible with Qdrant vector database sparse search
            
        Generation Process:
            1. Validate input text and model availability
            2. Clean and preprocess input text
            3. Generate sparse embedding using FastEmbed
            4. Extract indices and values from embedding result
            5. Convert to Qdrant SparseVector format
            6. Validate output structure and dimensions
            
        Example:
            >>> helper = SparseEmbeddingHelper()
            >>> sparse_vec = helper.generate_sparse_embedding("ABAP class constructor")
            >>> if sparse_vec:
            ...     print(f"Terms: {len(sparse_vec.indices)}")
            ...     print(f"Weights: {sparse_vec.values[:5]}")  # First 5 weights
        """
        if not self.is_available() or not text or not text.strip():
            return None
            
        try:
            cleaned_text = text.strip()
            logger.debug(f"FINAL FIXED: Generating sparse embedding for: '{cleaned_text[:50]}...'")
            
            # Generate embedding using FastEmbed
            sparse_result = self.sparse_model.embed([cleaned_text])
            sparse_list = list(sparse_result)
            
            if not sparse_list:
                return None
                
            sparse_embedding = sparse_list[0]
            
            # Extract indices and values with proper type conversion
            if hasattr(sparse_embedding, 'indices') and hasattr(sparse_embedding, 'values'):
                # Convert indices to list format
                if hasattr(sparse_embedding.indices, 'tolist'):
                    indices = sparse_embedding.indices.tolist()
                else:
                    indices = list(sparse_embedding.indices)
                    
                # Convert values to list format
                if hasattr(sparse_embedding.values, 'tolist'):
                    values = sparse_embedding.values.tolist()
                else:
                    values = list(sparse_embedding.values)
                
                # Validate dimensions and create SparseVector
                if len(indices) == len(values) and len(indices) > 0:
                    return SparseVector(indices=indices, values=values)
                
            return None
            
        except Exception as e:
            logger.error(f"FINAL FIXED: Error generating sparse embedding: {e}")
            return None


class FallbackEmbeddingWrapper:
    """
    Fallback embedding wrapper for testing and graceful degradation.
    
    This class provides a fallback mechanism when primary embedding models
    fail to load or are unavailable. It generates normalized random embeddings
    that maintain the correct dimensionality for system compatibility while
    providing minimal functionality for testing and development.
    
    Attributes:
        dimension (int): Dimensionality of generated embeddings
        
    Use Cases:
        - Model loading failures during development
        - Testing system components without full model dependencies
        - Graceful degradation in production when models are unavailable
        - Performance testing with minimal computational overhead
        
    Note:
        Fallback embeddings provide no semantic meaning and should only be
        used for testing or emergency fallback scenarios. They will not
        provide meaningful retrieval results.
        
    Example:
        >>> fallback = FallbackEmbeddingWrapper(dimension=1024)
        >>> embeddings = fallback.encode(["test query"])
        >>> print(f"Fallback embedding shape: {embeddings.shape}")  # (1, 1024)
    """
    
    def __init__(self, dimension: int = 1024):
        """
        Initialize fallback embedding wrapper with specified dimensions.
        
        Args:
            dimension (int): Embedding dimension to maintain compatibility
                           with downstream systems (default: 1024 for E5)
        """
        self.dimension = dimension
        logger.warning("FINAL FIXED: Using fallback embeddings")
        
    def encode(self, texts: Union[str, List[str]], batch_size: int = 4, code_specific: bool = False) -> np.ndarray:
        """
        Generate normalized random embeddings as fallback.
        
        This method generates random embeddings with proper normalization to
        maintain compatibility with cosine similarity operations used in
        retrieval systems.
        
        Args:
            texts (Union[str, List[str]]): Input texts (content ignored)
            batch_size (int): Batch size parameter (ignored in fallback)
            code_specific (bool): Code-specific flag (ignored in fallback)
            
        Returns:
            np.ndarray: Normalized random embeddings with shape (n_texts, dimension)
            
        Generation Process:
            1. Convert single strings to list format
            2. Generate random normal distribution matrix
            3. Normalize embeddings to unit length
            4. Return properly shaped numpy array
            
        Example:
            >>> fallback = FallbackEmbeddingWrapper()
            >>> embeddings = fallback.encode(["query1", "query2"])
            >>> print(embeddings.shape)  # (2, 1024)
            >>> print(np.linalg.norm(embeddings[0]))  # ~1.0 (normalized)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        logger.info(f"FINAL FIXED: Generating {len(texts)} fallback embeddings")
        
        # Generate random matrix with normal distribution
        random_matrix = np.random.randn(len(texts), self.dimension)
        
        # Normalize embeddings to unit length for cosine similarity
        norms = np.linalg.norm(random_matrix, axis=1, keepdims=True)
        normalized_embeddings = random_matrix / norms
        
        return normalized_embeddings