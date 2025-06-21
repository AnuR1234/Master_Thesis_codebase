#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit-Compatible Embedding Module for RAG Pipeline.

This module provides comprehensive embedding capabilities for the RAG pipeline,
supporting both dense and sparse vector embeddings optimized for Streamlit environments.
The module implements robust fallback mechanisms and is designed to work seamlessly
with SAP AI Core for production deployments while providing testing alternatives.

Key Features:
    - SAP AI Core integration for production-grade dense embeddings
    - OpenAI-compatible API with multiple deployment fallback
    - BM25 sparse embeddings for lexical matching
    - Streamlit-compatible synchronous interfaces
    - Robust error handling with graceful degradation
    - Testing fallbacks with random embeddings
    - Environment-based configuration management

The module supports multiple embedding strategies:
    - Dense Embeddings: High-quality semantic embeddings via SAP AI Core
    - Sparse Embeddings: BM25-based lexical matching for precise term matching
    - Fallback Embeddings: Random vectors for testing and development

Architecture:
    The embedding module follows a wrapper pattern that abstracts the complexity
    of different embedding providers and provides a unified interface for the
    RAG pipeline components.

Classes:
    OpenAIEmbeddingWrapper: Main production embedding class with SAP AI Core integration
    FallbackEmbeddingWrapper: Testing fallback with random embeddings
    SparseEmbeddingHelper: BM25 sparse embedding support

Dependencies:
    - gen_ai_hub: SAP AI Core native client integration
    - fastembed: FastEmbed library for BM25 sparse embeddings
    - qdrant_client: Vector database client for sparse vector formats
    - dotenv: Environment variable management

Author: SAP ABAP RAG Team
Version: 1.0.0
Date: 2025
License: MIT
"""

# Standard library imports
import asyncio
import logging
import os
from typing import List, Optional

# Third-party imports
import nest_asyncio
import requests
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client.models import SparseVector

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
RAG_ENV_PATH = r"/home/user/Desktop/RAG_pipeline_enhanced_conversational_claude/claude.env"
load_dotenv(RAG_ENV_PATH)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenAIEmbeddingWrapper:
    """
    Wrapper for embeddings with SAP AI Core integration.
    
    This class provides production-grade dense embeddings through SAP AI Core's
    native OpenAI integration. It implements robust fallback mechanisms across
    multiple deployment instances and handles authentication automatically.
    
    The wrapper is designed to be Streamlit-compatible by providing synchronous
    interfaces while using asynchronous operations internally for optimal performance.
    
    Features:
        - Multiple deployment instance fallback for high availability
        - Automatic authentication token management
        - Streamlit-compatible synchronous interface
        - Comprehensive error handling and logging
        - OpenAI API compatibility through SAP AI Core
        
    Attributes:
        model_instances (List[Tuple[str, str]]): Available deployment instances
        model (str): OpenAI model name for embeddings
        client_class: SAP AI Core AsyncOpenAI client class
        token (str): Authentication token for API access
        
    Examples:
        >>> wrapper = OpenAIEmbeddingWrapper()
        >>> embeddings = wrapper.encode(["Hello world", "ABAP programming"])
        >>> print(f"Generated {len(embeddings)} embeddings")
        >>> print(f"Embedding dimension: {len(embeddings[0])}")
    """
    
    def __init__(self) -> None:
        """
        Initialize with direct environment variable access and SAP AI Core integration.
        
        Sets up the embedding wrapper by:
        - Loading deployment IDs from environment variables
        - Initializing SAP AI Core client
        - Obtaining authentication token
        - Validating all required configurations
        
        Raises:
            ValueError: If no embedding deployment IDs are found
            ImportError: If SAP AI Core client is not available
            Exception: If authentication token retrieval fails
            
        Note:
            Expects environment variables in the format:
            AICORE_DEPLOYMENT_ID_embed_1, AICORE_DEPLOYMENT_ID_embed_2, etc.
            Also requires AICORE_AUTH_URL, AICORE_CLIENT_ID, AICORE_CLIENT_SECRET
        """
        logger.info("Initializing embedding wrapper with direct env var access")
        
        # Find embedding deployment IDs
        self.model_instances = []
        for i in range(1, 4):
            env_var = f"AICORE_DEPLOYMENT_ID_embed_{i}"
            if os.getenv(env_var):
                deployment_id = os.getenv(env_var)
                self.model_instances.append((f"embed_{i}", deployment_id))
                logger.info(f"Found embedding deployment: {env_var} = {deployment_id}")
        
        if not self.model_instances:
            logger.error("No embedding deployment IDs found")
            raise ValueError("No embedding deployment IDs found")
        
        # Standard OpenAI embedding model name
        self.model = "text-embedding-ada-002"
        
        # Import SAP AI Core dependencies
        try:
            from gen_ai_hub.proxy.native.openai import AsyncOpenAI as SapAsyncOpenAI
            self.client_class = SapAsyncOpenAI
            logger.info("Successfully imported SAP AI Core client")
        except ImportError as e:
            logger.error(f"Failed to import SAP AI Core client: {e}")
            raise ImportError("SAP AI Core client not available")
        
        # Get auth token once at initialization
        self.token = self._get_auth_token()
        logger.info("Successfully initialized embedding wrapper")
    
    def _get_auth_token(self) -> str:
        """
        Get authentication token from SAP AI Core OAuth endpoint.
        
        Performs OAuth2 client credentials flow to obtain an access token
        for authenticating with SAP AI Core services.
        
        Returns:
            Valid access token for SAP AI Core API calls
            
        Raises:
            ValueError: If required authentication credentials are missing
            Exception: If token retrieval fails due to network or auth issues
            
        Note:
            Uses client credentials grant type with the following parameters:
            - grant_type: client_credentials
            - client_id: From AICORE_CLIENT_ID environment variable
            - client_secret: From AICORE_CLIENT_SECRET environment variable
        """
        auth_url = os.getenv('AICORE_AUTH_URL')
        client_id = os.getenv('AICORE_CLIENT_ID')
        client_secret = os.getenv('AICORE_CLIENT_SECRET')
        
        # Direct validation
        if not auth_url or not client_id or not client_secret:
            logger.error(f"Missing auth credentials: URL={auth_url is not None}, ID={client_id is not None}, Secret={client_secret is not None}")
            raise ValueError("Missing SAP AI Core authentication credentials")
        
        logger.info(f"Getting auth token from {auth_url}")
        
        try:
            response = requests.post(
                auth_url,
                data={
                    'grant_type': 'client_credentials',
                    'client_id': client_id,
                    'client_secret': client_secret
                },
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                token = token_data.get('access_token')
                logger.info(f"Successfully got auth token (length: {len(token)})")
                return token
            else:
                error_msg = f"Auth failed: {response.status_code} - {response.text}"
                logger.error(error_msg)
                raise Exception(error_msg)
        except Exception as e:
            logger.error(f"Error getting auth token: {e}")
            raise
    
    def _create_client(self, model_instance: tuple[str, str]):
        """
        Create SAP AI Core client for a specific model instance.
        
        Creates an authenticated AsyncOpenAI client configured for the specified
        deployment instance with proper headers for SAP AI Core integration.
        
        Args:
            model_instance: Tuple of (instance_name, deployment_id) for the target deployment
            
        Returns:
            Configured AsyncOpenAI client ready for API calls
            
        Raises:
            ValueError: If required environment variables are missing
            
        Note:
            The client is configured with:
            - Bearer token authentication
            - AI-Resource-Group header for resource isolation
            - AI-Deployment-ID header for specific deployment targeting
        """
        instance_name, deployment_id = model_instance
        
        base_url = os.getenv('AICORE_BASE_URL')
        resource_group = os.getenv('AICORE_RESOURCE_GROUP', 'default')
        
        if not base_url:
            raise ValueError("AICORE_BASE_URL is missing")
        
        client = self.client_class(
            base_url=base_url,
            api_key="dummy",  # Not used with SAP AI Core
            default_headers={
                "Authorization": f"Bearer {self.token}",
                "AI-Resource-Group": resource_group,
                "AI-Deployment-ID": deployment_id
            }
        )
        
        logger.info(f"Created client for {instance_name} with deployment ID {deployment_id}")
        return client
    
    async def _get_embedding(self, text: str, model_instance: tuple[str, str]) -> Optional[List[float]]:
        """
        Get embedding asynchronously from a specific model instance.
        
        Generates a dense vector embedding for the input text using the specified
        SAP AI Core deployment instance. Handles API communication and response parsing.
        
        Args:
            text: Input text to embed (will be converted to string if not already)
            model_instance: Tuple of (instance_name, deployment_id) to use
            
        Returns:
            List of floats representing the embedding vector, or None if generation fails
            
        Note:
            Uses the OpenAI embeddings API format through SAP AI Core proxy.
            The embedding dimension depends on the model configuration (typically 1536).
        """
        try:
            # Ensure text is a string
            if not isinstance(text, str):
                text = str(text)
            
            # Create client for this model instance
            client = self._create_client(model_instance)
            
            # Generate embedding
            logger.info(f"Generating embedding using {model_instance[0]}")
            
            response = await client.embeddings.create(
                input=text,
                model=self.model
            )
            
            if hasattr(response, 'data') and len(response.data) > 0:
                logger.info(f"Successfully generated embedding with {model_instance[0]}")
                return response.data[0].embedding
            else:
                logger.error(f"Unexpected response format from {model_instance[0]}")
                return None
                
        except Exception as e:
            logger.error(f"Error with {model_instance[0]}: {e}")
            return None
    
    async def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        Try to generate embedding using any available model instance with fallback mechanisms.
        
        Attempts to generate an embedding using the available deployment instances
        in order, providing automatic failover if one instance is unavailable.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as list of floats, or None if all instances fail
            
        Note:
            Implements graceful degradation by trying each deployment instance
            in sequence until one succeeds or all fail. This ensures high
            availability even if some deployments are temporarily unavailable.
        """
        for model_instance in self.model_instances:
            try:
                embedding = await self._get_embedding(text, model_instance)
                if embedding is not None:
                    return embedding
            except Exception as e:
                logger.error(f"Error with {model_instance[0]}: {e}")
                continue
        
        # If all instances fail
        logger.error(f"All embedding instances failed for text: {text[:50]}...")
        return None
    
    def encode(self, texts: List[str] | str) -> List[Optional[List[float]]]:
        """
        Streamlit-compatible encoding function for generating embeddings.
        
        This is the main interface for generating embeddings in a Streamlit environment.
        It handles both single strings and lists of strings, providing a synchronous
        interface while using asynchronous operations internally.
        
        Args:
            texts: Single string or list of strings to embed
            
        Returns:
            List of embedding vectors (each as list of floats) or None for failed embeddings
            
        Examples:
            >>> wrapper = OpenAIEmbeddingWrapper()
            >>> # Single text
            >>> embeddings = wrapper.encode("ABAP class implementation")
            >>> print(f"Single embedding dimension: {len(embeddings[0])}")
            
            >>> # Multiple texts
            >>> texts = ["ABAP classes", "Interface methods", "Data structures"]
            >>> embeddings = wrapper.encode(texts)
            >>> print(f"Generated {len(embeddings)} embeddings")
            
        Note:
            Uses nest_asyncio to handle asyncio operations within Streamlit's
            event loop. Failed embeddings are returned as None in the result list.
        """
        # Ensure texts is always a list
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return []
            
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            
            # Process each text synchronously to avoid event loop issues
            embeddings = []
            for i, text in enumerate(texts):
                # Run the async function using nest_asyncio
                embedding = asyncio.run(self._generate_embedding(text))
                embeddings.append(embedding)
                
                if embedding is None:
                    logger.error(f"Failed to generate embedding for text {i+1}")
            
            failed = sum(1 for e in embeddings if e is None)
            if failed > 0:
                logger.error(f"Failed to generate {failed}/{len(texts)} embeddings")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in encoding: {e}")
            import traceback
            traceback.print_exc()
            return [None] * len(texts)


class FallbackEmbeddingWrapper:
    """
    Fallback wrapper that generates random embeddings for testing.
    
    This class provides a testing and development fallback when production
    embedding services are not available. It generates normalized random
    vectors that maintain the same interface as the production embeddings.
    
    Warning:
        This class is intended for testing and development only. Random embeddings
        do not capture semantic meaning and will not provide meaningful retrieval
        results in production scenarios.
        
    Attributes:
        dimension (int): Dimension of the generated random embeddings
        
    Examples:
        >>> fallback = FallbackEmbeddingWrapper(dimension=1536)
        >>> embeddings = fallback.encode(["test text"])
        >>> print(f"Random embedding dimension: {len(embeddings[0])}")
    """
    
    def __init__(self, dimension: int = 1536) -> None:
        """
        Initialize the fallback wrapper with specified embedding dimension.
        
        Args:
            dimension: Dimension of the random embeddings to generate.
                      Should match the dimension of production embeddings (default: 1536)
                      
        Note:
            Logs a warning to ensure users are aware they're using random embeddings
            which are only suitable for testing purposes.
        """
        self.dimension = dimension
        logger.warning("Using FallbackEmbeddingWrapper with random embeddings - FOR TESTING ONLY")
        
    def encode(self, texts: List[str] | str) -> List[List[float]]:
        """
        Generate random embeddings for testing purposes.
        
        Creates normalized random vectors for each input text. The vectors are
        properly normalized to unit length to simulate realistic embedding behavior.
        
        Args:
            texts: Single string or list of strings (content is ignored)
            
        Returns:
            List of normalized random embedding vectors
            
        Note:
            The actual content of the texts is ignored since these are random embeddings.
            Each vector is normalized to unit length to match typical embedding behavior.
        """
        import numpy as np
        logger.info(f"Generating random embeddings for {len(texts) if isinstance(texts, list) else 1} texts")
        
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        for _ in texts:
            # Generate random vector and normalize it
            random_vector = np.random.randn(self.dimension)
            normalized_vector = random_vector / np.linalg.norm(random_vector)
            embeddings.append(normalized_vector.tolist())
            
        return embeddings


class SparseEmbeddingHelper:
    """
    Helper class for sparse embeddings - BM25 compatible version.
    
    This class provides BM25-based sparse embeddings for lexical matching in the
    RAG pipeline. It integrates with FastEmbed to generate sparse vectors that
    complement dense embeddings for hybrid search scenarios.
    
    BM25 (Best Matching 25) is a probabilistic ranking function used in information
    retrieval that provides excellent performance for exact term matching and
    keyword-based queries.
    
    Features:
        - FastEmbed integration for BM25 embeddings
        - Qdrant SparseVector format compatibility
        - Robust error handling and model validation
        - Thread-safe single-threaded operation
        - Comprehensive logging and debugging support
        
    Attributes:
        model_name (str): Name of the BM25 model being used
        sparse_model: FastEmbed SparseTextEmbedding instance
        
    Examples:
        >>> helper = SparseEmbeddingHelper()
        >>> if helper.is_available():
        ...     vector = helper.generate_sparse_embedding("ABAP class method")
        ...     print(f"Generated sparse vector with {len(vector.indices)} features")
    """
    
    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        """
        Initialize the sparse embedding helper with BM25 model.
        
        Sets up the FastEmbed BM25 model with configuration optimized for
        stability and compatibility with the RAG pipeline.
        
        Args:
            model_name: Name of the sparse embedding model. Should remain
                       "Qdrant/bm25" for compatibility with existing vector data.
                       
        Note:
            Uses single-threaded operation and default cache to avoid
            concurrency issues in Streamlit environments. Performs model
            validation to ensure proper functionality.
        """
        self.model_name = model_name
        self.sparse_model = None
        
        try:
            logger.info(f"Initializing sparse embedding model: {model_name}")
            # Add cache_dir and other parameters to improve stability
            self.sparse_model = SparseTextEmbedding(
                model_name=model_name,
                cache_dir=None,  # Use default cache
                threads=1,       # Use single thread to avoid concurrency issues
            )
            logger.info(f"Successfully initialized sparse embedding model: {model_name}")
            
            # Test the model with a simple text to ensure it works
            test_result = self._test_model()
            if not test_result:
                logger.error("Model test failed, marking as unavailable")
                self.sparse_model = None
                
        except Exception as e:
            logger.error(f"Failed to initialize sparse embedding model {model_name}: {e}")
            self.sparse_model = None

    def _test_model(self) -> bool:
        """
        Test the model with a simple embedding to ensure it works correctly.
        
        Performs a validation test by generating an embedding for a simple
        test string and verifying the output has the expected structure.
        
        Returns:
            True if the model passes validation tests, False otherwise
            
        Note:
            Checks that the model returns embeddings with the expected
            SparseEmbedding structure containing 'indices' and 'values' attributes.
        """
        try:
            test_text = "test"
            result = list(self.sparse_model.embed([test_text]))
            if result and len(result) > 0:
                # Check if the result has the expected structure
                embedding = result[0]
                if hasattr(embedding, 'indices') and hasattr(embedding, 'values'):
                    logger.info("Model test passed - BM25 model is working correctly")
                    return True
                else:
                    logger.error(f"Model test failed - unexpected embedding structure: {type(embedding)}")
                    return False
            else:
                logger.error("Model test failed - no embeddings returned")
                return False
        except Exception as e:
            logger.error(f"Model test failed with exception: {e}")
            return False

    def is_available(self) -> bool:
        """
        Check if sparse embedding model is available and ready for use.
        
        Returns:
            True if the BM25 model is loaded and operational, False otherwise
            
        Examples:
            >>> helper = SparseEmbeddingHelper()
            >>> if helper.is_available():
            ...     print("BM25 embeddings are ready")
            ... else:
            ...     print("BM25 embeddings unavailable, using dense only")
        """
        return self.sparse_model is not None

    def process_sparse_vector(self, sparse_vec_result) -> Optional[SparseVector]:
        """
        Process BM25 sparse vector from FastEmbed into Qdrant format.
        
        Converts the output from FastEmbed's embed() method into the SparseVector
        format required by Qdrant for sparse vector search operations.
        
        Args:
            sparse_vec_result: Result from FastEmbed embed() method, typically
                             a generator or list of SparseEmbedding objects
                             
        Returns:
            SparseVector in Qdrant format with indices and values, or None if
            processing fails due to invalid input or conversion errors
            
        Note:
            FastEmbed returns SparseEmbedding objects with indices and values
            as numpy arrays. This method converts them to Python lists as
            required by Qdrant's SparseVector format.
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
                logger.error(f"Available attributes: {dir(sparse_embedding)}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing BM25 sparse vector: {e}")
            logger.error(f"Input type: {type(sparse_vec_result)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None

    def generate_sparse_embedding(self, text: str) -> Optional[SparseVector]:
        """
        Generate a BM25 sparse embedding for a text string.
        
        Creates a BM25 sparse vector representation of the input text that
        can be used for sparse vector search in Qdrant. BM25 provides
        excellent performance for keyword matching and term-based queries.
        
        Args:
            text: Input text to embed. Should be non-empty and contain
                 meaningful content for optimal BM25 performance.
                 
        Returns:
            SparseVector in Qdrant format containing BM25 term weights,
            or None if generation fails due to model issues or empty input
            
        Examples:
            >>> helper = SparseEmbeddingHelper()
            >>> vector = helper.generate_sparse_embedding("ABAP class implementation")
            >>> if vector:
            ...     print(f"Generated BM25 vector with {len(vector.indices)} terms")
            ...     print(f"Top terms: {vector.indices[:5]}")
            
        Note:
            BM25 generates sparse vectors where indices represent vocabulary
            terms and values represent their BM25 scores. Empty or whitespace-only
            text will result in None return value.
        """
        if not self.is_available():
            logger.error("BM25 sparse embedding model is not available")
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
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    def get_model_info(self) -> Dict[str, any]:
        """
        Get information about the current sparse embedding model.
        
        Returns comprehensive information about the BM25 model configuration
        and availability status for debugging and monitoring purposes.
        
        Returns:
            Dictionary containing:
            - model_name: Name of the BM25 model
            - is_available: Whether the model is ready for use
            - model_type: Description of the embedding type
            
        Examples:
            >>> helper = SparseEmbeddingHelper()
            >>> info = helper.get_model_info()
            >>> print(f"Model: {info['model_name']}")
            >>> print(f"Available: {info['is_available']}")
            >>> print(f"Type: {info['model_type']}")
        """
        return {
            "model_name": self.model_name,
            "is_available": self.is_available(),
            "model_type": "BM25 SparseTextEmbedding"
        }