# -*- coding: utf-8 -*-
"""
Complete Enhanced RAG pipeline with re-enabled query enhancement and improved response quality.

This module provides a comprehensive RAG (Retrieval-Augmented Generation) pipeline
specifically designed for SAP ABAP documentation systems. It integrates query 
enhancement, anti-hallucination mechanisms, intelligent retrieval, and response
generation with extensive monitoring and validation capabilities.

Key Features:
    - Enhanced query understanding and expansion for better retrieval
    - Multi-mode query enhancement (disabled, conservative, aggressive)
    - Anti-hallucination detection with pattern-based validation
    - Intelligent document filtering and relevance scoring
    - Context-aware system prompt selection
    - Comprehensive confidence scoring and risk assessment
    - Real-time performance monitoring and logging
    - Hybrid retrieval with dense and sparse embeddings
    - Cross-encoder reranking for improved relevance
    - Multi-collection support (ABAP classes vs reports)

Pipeline Components:
    - Query Enhancement: Intelligent query expansion and context addition
    - Retrieval: Hybrid search with dense (E5) and sparse (BM25) embeddings
    - Reranking: Cross-encoder relevance refinement
    - Filtering: Document relevance and quality assessment
    - Generation: Context-aware response synthesis
    - Validation: Anti-hallucination and confidence assessment

Anti-Hallucination Features:
    - External knowledge phrase detection
    - Parameter and method validation against source documents
    - Response length and context correlation analysis
    - Risk level assessment (LOW/MEDIUM/HIGH)
    - Confidence adjustment based on validation results

Query Analysis Types:
    - Method queries: Function and procedure documentation
    - Interface queries: Interface implementation and structure
    - Parameter queries: Method parameters and argument analysis
    - Implementation queries: Usage examples and code patterns
    - Comparison queries: Feature and option comparisons
    - Error handling queries: Exception management and debugging

Example Usage:
    >>> pipeline = EnhancedRAGPipeline()
    >>> pipeline.set_enhancement_mode("conservative")
    >>> 
    >>> result = await pipeline.process_query(
    ...     "What parameters does submit method accept?",
    ...     use_hybrid=True,
    ...     use_reranker=True
    ... )
    >>> 
    >>> print(f"Enhanced Query: {result['enhanced_query']}")
    >>> print(f"Confidence: {result['confidence_level']}")
    >>> print(f"Sources: {result['context_count']}")

Dependencies:
    - retriever: RAGRetriever for document search and ranking
    - generator: EnhancedRAGGenerator for response synthesis
    - query_enhancer: EnhancedQueryEnhancer for query optimization
    - config: System configuration and prompts
"""
import logging
import json
import asyncio
import nest_asyncio
import re
from typing import Dict, Any, List, Optional, Tuple, Union
import os
import time

# Apply nest_asyncio for Streamlit compatibility
nest_asyncio.apply()

from config import (
    DEFAULT_TOP_K,
    USE_HYBRID_DEFAULT,
    USE_RERANKER_DEFAULT,
    LOG_LEVEL,
    LOG_FILE,
    COLLECTIONS,
    DEFAULT_COLLECTION_TYPE,
    STRICT_DOCUMENT_ADHERENCE_PROMPT,
    ENHANCED_RESPONSE_PROMPT,
    CONVERSATION_SYSTEM_PROMPT,
    OUT_OF_SCOPE_PROMPT,
    OUT_OF_SCOPE_CONVERSATION_PROMPT,
    STRUCTURE_SYSTEM_PROMPT,
    INTERFACE_SYSTEM_PROMPT,
    ENHANCED_METHOD_SYSTEM_PROMPT,
    CONFIDENCE_THRESHOLDS,
    MIN_DOCUMENT_SCORE,
    MAX_CONTEXT_DOCS
)

from retriever import RAGRetriever
from generator import EnhancedRAGGenerator
from query_enhancer import EnhancedQueryEnhancer

# Configure comprehensive logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EnhancedRAGPipeline:
    """
    Complete Enhanced RAG pipeline with query enhancement and improved response quality.
    
    This class orchestrates the complete RAG workflow from query enhancement through
    response generation, with comprehensive anti-hallucination measures and quality
    controls. It provides intelligent query analysis, adaptive retrieval strategies,
    and context-aware response generation optimized for SAP ABAP documentation.
    
    The pipeline operates in multiple stages:
    1. Query Enhancement: Intelligent query expansion and context addition
    2. Query Analysis: Type classification and complexity assessment
    3. Retrieval: Hybrid search with adaptive limits based on query type
    4. Reranking: Cross-encoder relevance refinement
    5. Filtering: Document quality and relevance validation
    6. Generation: Context-aware response synthesis with appropriate prompts
    7. Validation: Anti-hallucination detection and confidence assessment
    
    Attributes:
        retriever (RAGRetriever): Document retrieval and ranking component
        generator (EnhancedRAGGenerator): Response generation component
        query_enhancer (EnhancedQueryEnhancer): Query enhancement component
        current_collection_type (str): Active document collection type
        query_enhancement_enabled (bool): Query enhancement activation flag
        max_tokens (int): Maximum response length in tokens
        formatting_mode (str): Response formatting style
        hallucination_keywords (List[str]): Phrases indicating external knowledge
        validation_patterns (Dict[str, List[str]]): Regex patterns for validation
        
    Example:
        >>> pipeline = EnhancedRAGPipeline()
        >>> pipeline.set_enhancement_mode("conservative")
        >>> pipeline.set_generation_params(max_tokens=1200, formatting_mode="Enhanced")
        >>> 
        >>> result = await pipeline.process_query(
        ...     "How to implement ABAP class constructor?",
        ...     use_hybrid=True,
        ...     use_reranker=True,
        ...     top_k=5
        ... )
        >>> 
        >>> print(f"Confidence: {result['confidence_level']}")
        >>> print(f"Enhancement: {result['query_enhancement_used']}")
    """
    
    def __init__(self):
        """
        Initialize the enhanced pipeline with comprehensive anti-hallucination measures.
        
        This method sets up all pipeline components with optimal configurations for
        SAP ABAP documentation retrieval, including query enhancement, anti-hallucination
        detection, and quality monitoring systems.
        
        Initialization Process:
            1. Initialize core components (retriever, generator, query enhancer)
            2. Configure query enhancement with conservative defaults
            3. Set up anti-hallucination detection patterns
            4. Configure generation parameters and formatting
            5. Initialize validation and monitoring systems
            
        Raises:
            Exception: If any core component fails to initialize
        """
        logger.info("Initializing Complete Enhanced Anti-Hallucination RAG Pipeline with Query Enhancement")
        
        try:
            # Initialize core components
            self.retriever = RAGRetriever()
            self.generator = EnhancedRAGGenerator()
            self.current_collection_type = DEFAULT_COLLECTION_TYPE
            
            # RE-ENABLED: Query enhancement with enhanced functionality
            self.query_enhancement_enabled = True
            self.query_enhancer = EnhancedQueryEnhancer(self.generator)
            self.query_enhancer.set_enhancement_mode("conservative")  # Default mode
            logger.info("ENHANCED: Query enhancement ENABLED with improved functionality")
            
            # Enhanced generation parameters
            self.max_tokens = 1200
            self.formatting_mode = "Enhanced"
            
            # Anti-hallucination tracking (kept for compatibility)
            self.hallucination_keywords = [
                "generally", "typically", "usually", "standard SAP", "common practice",
                "by default", "in most cases", "standard procedure", "normally",
                "standard ABAP", "typical SAP", "commonly", "often", "traditionally"
            ]
            
            # Enhanced pattern validators for hallucination detection
            self.validation_patterns = {
                "external_knowledge": [
                    r"generally(?:\s+speaking)?",
                    r"typically",
                    r"usually",
                    r"standard\s+sap",
                    r"common\s+practice",
                    r"by\s+default",
                    r"in\s+most\s+cases",
                    r"standard\s+procedure",
                    r"normally",
                    r"standard\s+abap",
                    r"typical\s+sap"
                ],
                "parameter_patterns": r"(?:iv_|et_|ct_|it_)[a-z_]+",
                "method_patterns": r"\b[A-Z_]{2,}~[A-Z_]{2,}\b",
                "class_patterns": r"\bCL_[A-Z_]+\b",
                "interface_patterns": r"\bIF_[A-Z_]+\b"
            }
            
            logger.info(f"Enhanced Pipeline initialized successfully with collection: {self.current_collection_type}")
            
        except Exception as e:
            logger.error(f"Failed to initialize pipeline: {e}")
            raise
    
    def set_enhancement_mode(self, mode: str) -> None:
        """
        Set query enhancement mode for the pipeline.
        
        This method configures the query enhancement behavior for subsequent
        query processing. Different modes provide varying levels of query
        modification to balance enhancement with preservation.
        
        Args:
            mode (str): Enhancement mode - "disabled", "conservative", or "aggressive"
            
        Enhancement Modes:
            - "disabled": No query modification, preserves original queries
            - "conservative": Minimal enhancement, adds context only when needed
            - "aggressive": Comprehensive enhancement with detailed context
            
        Example:
            >>> pipeline.set_enhancement_mode("conservative")
            >>> pipeline.set_enhancement_mode("aggressive")
        """
        if hasattr(self, 'query_enhancer'):
            self.query_enhancer.set_enhancement_mode(mode)
            logger.info(f"Query enhancement mode set to: {mode}")
        else:
            logger.warning("Query enhancer not initialized")
    
    def set_generation_params(self, max_tokens: int = None, formatting_mode: str = None, **kwargs) -> None:
        """
        Set generation parameters for response synthesis.
        
        This method configures the response generation behavior including
        token limits, formatting styles, and other generation parameters.
        
        Args:
            max_tokens (int, optional): Maximum response length in tokens
            formatting_mode (str, optional): Response formatting style
                ("Enhanced", "Basic", "Raw")
            **kwargs: Additional generation parameters passed to generator
            
        Example:
            >>> pipeline.set_generation_params(
            ...     max_tokens=1600,
            ...     formatting_mode="Enhanced",
            ...     temperature=0.1
            ... )
        """
        if max_tokens:
            self.max_tokens = max_tokens
            if hasattr(self.generator, 'set_generation_config'):
                self.generator.set_generation_config(max_new_tokens=max_tokens)
            logger.info(f"Max tokens set to: {max_tokens}")
        
        if formatting_mode:
            self.formatting_mode = formatting_mode
            if hasattr(self.generator, 'set_generation_config'):
                self.generator.set_generation_config(formatting_mode=formatting_mode)
            logger.info(f"Formatting mode set to: {formatting_mode}")
        
        # Pass any additional parameters to generator
        if kwargs and hasattr(self.generator, 'update_generation_settings'):
            self.generator.update_generation_settings(**kwargs)
    
    def _calculate_confidence_level(self, context_results: List[Tuple[Any, float]], query: str) -> str:
        """
        Enhanced confidence calculation with balanced thresholds and query complexity factors.
        
        This method calculates confidence levels based on retrieval scores, context
        quantity, and query complexity. It provides more nuanced confidence assessment
        that considers both the quality of retrieved documents and query characteristics.
        
        Args:
            context_results (List[Tuple[Any, float]]): Retrieved documents with scores
            query (str): Original query for complexity analysis
            
        Returns:
            str: Confidence level - "HIGH", "MEDIUM", or "LOW"
            
        Confidence Factors:
            - Average and top retrieval scores (40% weight)
            - Context document count (30% weight)
            - Query complexity and specificity (30% weight)
            
        Thresholds:
            - HIGH: Strong scores + sufficient context + appropriate complexity
            - MEDIUM: Moderate scores + adequate context
            - LOW: Weak scores or insufficient context
            
        Example:
            >>> confidence = pipeline._calculate_confidence_level(
            ...     [(doc1, 0.85), (doc2, 0.72)],
            ...     "What parameters does submit method accept?"
            ... )
            >>> print(confidence)  # "HIGH"
        """
        if not context_results:
            return "LOW"
        
        # Get scoring metrics
        scores = [abs(score) for _, score in context_results]
        top_score = max(scores) if scores else 0
        avg_score = sum(scores) / len(scores) if scores else 0
        context_count = len(context_results)
        
        # Query complexity factor (simpler queries should get higher confidence)
        query_words = len(query.split())
        complexity_factor = max(0.3, min(1.0, 15.0 / query_words))
        
        # Calculate weighted confidence (more balanced)
        score_weight = 0.4
        count_weight = 0.3
        complexity_weight = 0.3
        
        # Normalize context count
        count_factor = min(1.0, context_count / 3.0)
        
        # Combined confidence score
        confidence_score = (
            score_weight * avg_score + 
            count_weight * count_factor + 
            complexity_weight * complexity_factor
        )
        
        # More lenient thresholds
        high_threshold = max(0.4, CONFIDENCE_THRESHOLDS["HIGH"] * 0.8)
        medium_threshold = max(0.2, CONFIDENCE_THRESHOLDS["MEDIUM"] * 0.7)
        
        if confidence_score >= high_threshold and top_score >= 0.5:
            return "HIGH"
        elif confidence_score >= medium_threshold and top_score >= 0.2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _extract_confidence_level(self, response_text: str) -> Tuple[str, str]:
        """
        Extract confidence level from response text with enhanced pattern matching.
        
        This method searches for confidence indicators in generated responses
        using multiple pattern variations and extracts the confidence level
        while cleaning the response text.
        
        Args:
            response_text (str): Generated response text potentially containing confidence
            
        Returns:
            Tuple[str, str]: (confidence_level, cleaned_response_text)
            
        Confidence Patterns Searched:
            - "Confidence: HIGH/MEDIUM/LOW"
            - "**Confidence:** HIGH/MEDIUM/LOW"
            - "## Confidence: HIGH/MEDIUM/LOW"
            - Case-insensitive variations
            
        Example:
            >>> confidence, clean_text = pipeline._extract_confidence_level(
            ...     "Confidence: HIGH\n\nThe submit method accepts..."
            ... )
            >>> print(confidence)  # "HIGH"
            >>> print(clean_text)  # "The submit method accepts..."
        """
        confidence_level = "MEDIUM"
        cleaned_response = response_text
        
        # Multiple confidence pattern attempts
        confidence_patterns = [
            r"^Confidence:\s*(HIGH|MEDIUM|LOW)\s*\n",
            r"^confidence:\s*(HIGH|MEDIUM|LOW)\s*\n",
            r"^\*\*Confidence:\*\*\s*(HIGH|MEDIUM|LOW)\s*\n",
            r"^##\s*Confidence:\s*(HIGH|MEDIUM|LOW)\s*\n"
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                confidence_level = match.group(1).upper()
                cleaned_response = re.sub(pattern, "", response_text, 1, re.IGNORECASE).strip()
                break
        
        return confidence_level, cleaned_response
    
    def _detect_potential_hallucination(self, response: str, contexts: List[Dict], query: str) -> Dict[str, Any]:
        """
        Enhanced hallucination detection with reduced false positives and comprehensive validation.
        
        This method analyzes generated responses for potential hallucination indicators
        including external knowledge phrases, undocumented parameters/methods, and
        response-context misalignment. It provides detailed risk assessment with
        specific warnings and recommendations.
        
        Args:
            response (str): Generated response text to analyze
            contexts (List[Dict]): Source document contexts used for generation
            query (str): Original query for context-specific validation
            
        Returns:
            Dict[str, Any]: Hallucination analysis containing:
                - suspicious_phrases (List[str]): External knowledge indicators
                - undocumented_parameters (List[str]): Parameters not in source docs
                - undocumented_methods (List[str]): Methods not in source docs
                - risk_level (str): Overall risk assessment (LOW/MEDIUM/HIGH)
                - risk_score (int): Numerical risk score
                - warnings (List[str]): Specific warnings and issues
                
        Validation Categories:
            1. External Knowledge Phrases: "generally", "typically", "standard SAP"
            2. Parameter Validation: Check parameters against source documentation
            3. Method Validation: Verify method names exist in contexts
            4. Response-Context Alignment: Check response length vs context
            
        Risk Scoring:
            - Critical phrases: +2 points each
            - Undocumented parameters (>3): +3 points
            - Undocumented methods: +2 points each
            - Long response without context: +3 points
            
        Example:
            >>> hallucination_check = pipeline._detect_potential_hallucination(
            ...     "Generally, ABAP methods accept iv_param1, iv_param2...",
            ...     contexts,
            ...     "What parameters does submit accept?"
            ... )
            >>> print(hallucination_check["risk_level"])  # "HIGH"
        """
        hallucination_indicators = {
            "suspicious_phrases": [],
            "undocumented_parameters": [],
            "undocumented_methods": [],
            "risk_level": "LOW",
            "risk_score": 0,
            "warnings": []
        }
        
        # Safe string handling
        response_lower = str(response).lower() if response else ""
        
        # Build context text safely
        context_parts = []
        for ctx in contexts:
            if ctx:
                text = ctx.get("text") or ""
                code_snippet = ctx.get("code_snippet") or ""
                title = ctx.get("title") or ""
                
                text = str(text) if text else ""
                code_snippet = str(code_snippet) if code_snippet else ""
                title = str(title) if title else ""
                
                context_parts.append(f"{text} {code_snippet} {title}")
        
        context_text = " ".join(context_parts)
        context_lower = context_text.lower() if context_text else ""
        
        # 1. Check for obvious external knowledge phrases (only the most problematic ones)
        critical_phrases = ["generally speaking", "typically", "usually", "standard sap", "common practice"]
        for phrase in critical_phrases:
            if phrase in response_lower:
                hallucination_indicators["suspicious_phrases"].append(phrase)
                hallucination_indicators["risk_score"] += 2  # Reduced penalty
        
        # 2. Only validate parameters if it's a parameter-specific query
        if ("parameter" in query.lower() or "accept" in query.lower()) and response_lower and context_lower:
            try:
                response_params = set(re.findall(self.validation_patterns["parameter_patterns"], response_lower))
                context_params = set(re.findall(self.validation_patterns["parameter_patterns"], context_lower))
                undocumented_params = response_params - context_params
                
                if undocumented_params and len(undocumented_params) > 3:  # Only flag if many undocumented
                    hallucination_indicators["undocumented_parameters"] = list(undocumented_params)
                    hallucination_indicators["risk_score"] += 3  # Reduced penalty
            except Exception as e:
                logger.warning(f"Error in parameter validation: {e}")
        
        # 3. Only validate methods if many are mentioned
        if response_lower and context_text:
            try:
                response_methods = set(re.findall(self.validation_patterns["method_patterns"], response))
                if len(response_methods) > 3:
                    context_methods = set(re.findall(self.validation_patterns["method_patterns"], context_text))
                    undocumented_methods = response_methods - context_methods
                    
                    if undocumented_methods:
                        hallucination_indicators["undocumented_methods"] = list(undocumented_methods)
                        hallucination_indicators["risk_score"] += 2  # Reduced penalty
            except Exception as e:
                logger.warning(f"Error in method validation: {e}")
        
        # 4. Check for very long responses with no context (likely hallucination)
        if len(contexts) == 0 and len(response) > 1000:  # Increased threshold
            hallucination_indicators["risk_score"] += 3
            hallucination_indicators["warnings"].append("Long response without context")
        
        # 5. Determine risk level (more lenient thresholds)
        if hallucination_indicators["risk_score"] >= 10:  # Increased threshold
            hallucination_indicators["risk_level"] = "HIGH"
        elif hallucination_indicators["risk_score"] >= 5:  # Increased threshold
            hallucination_indicators["risk_level"] = "MEDIUM"
        else:
            hallucination_indicators["risk_level"] = "LOW"
        
        return hallucination_indicators
    
    def _filter_relevant_documents(self, context_results: List[Tuple[Any, float]]) -> List[Tuple[Any, float]]:
        """
        Enhanced document filtering with improved relevance detection and lenient thresholds.
        
        This method filters retrieved documents based on relevance scores while
        maintaining sufficient context for comprehensive responses. It applies
        minimal filtering to preserve useful documents while removing clearly
        irrelevant content.
        
        Args:
            context_results (List[Tuple[Any, float]]): Documents with relevance scores
            
        Returns:
            List[Tuple[Any, float]]: Filtered documents above relevance threshold
            
        Filtering Strategy:
            - Very lenient threshold (0.01) to preserve most documents
            - Reranked results are already sorted by relevance
            - Maximum 8 documents to prevent information overload
            - Detailed logging for debugging and optimization
            
        Example:
            >>> filtered = pipeline._filter_relevant_documents(reranked_results)
            >>> print(f"Kept {len(filtered)} documents")
        """
        if not context_results:
            logger.info("FILTER: No context results to filter")
            return []
        
        logger.info(f"FILTER: Starting with {len(context_results)} documents")
        
        # Take the reranked results as they are (reranker already sorted by relevance)
        # Apply minimal filtering to remove only clearly irrelevant documents
        filtered_results = []
        
        for i, (result, score) in enumerate(context_results):
            abs_score = abs(score)
            
            # Very lenient filtering - only remove extremely low scores
            if abs_score >= 0.01:  # Very low threshold
                filtered_results.append((result, score))
                if i < 5:  # Log first 5 for debugging
                    title = result.payload.get("title", "No title")
                    logger.info(f"FILTER: Kept {i+1}: '{title}' (score: {abs_score:.4f})")
            else:
                title = result.payload.get("title", "No title")
                logger.info(f"FILTER: Filtered out: '{title}' (score: {abs_score:.4f})")
        
        # Take up to 8 documents (increased from 5)
        max_docs = min(8, len(filtered_results))
        final_results = filtered_results[:max_docs]
        
        logger.info(f"FILTER: Kept {len(final_results)} documents from {len(context_results)} input")
        
        return final_results
    
    def _format_context_with_strict_boundaries(self, context_results: List[Tuple[Any, float]]) -> str:
        """
        Enhanced context formatting with improved structure preservation and content extraction.
        
        This method formats retrieved documents into a structured context string
        that preserves important information while maintaining clear boundaries
        between different sources. It processes both code snippets and documentation
        text with appropriate cleaning and organization.
        
        Args:
            context_results (List[Tuple[Any, float]]): Documents with relevance scores
            
        Returns:
            str: Formatted context string with clear document boundaries
            
        Formatting Structure:
            - Document numbering for clear identification
            - Separate sections for code and documentation
            - Clear separators between documents
            - Comprehensive content preservation with artifacts removal
            
        Example:
            >>> context = pipeline._format_context_with_strict_boundaries(documents)
            >>> print(context[:200])  # First 200 chars of formatted context
        """
        if not context_results:
            return "No relevant documentation found for this query."
        
        logger.info(f"CONTEXT FORMAT: Starting with {len(context_results)} documents")
        
        # Extract comprehensive content with better preservation
        content_parts = []
        
        # Include up to 6 documents (increased from 5)
        max_docs = min(6, len(context_results))
        
        for i, (result, score) in enumerate(context_results[:max_docs]):
            payload = result.payload
            
            # Safe extraction with proper null checking
            title = payload.get("title") or ""
            text = payload.get("text") or ""
            code_snippet = payload.get("code_snippet") or ""
            
            # Convert to string and strip safely
            title = str(title).strip() if title else ""
            text = str(text).strip() if text else ""
            code_snippet = str(code_snippet).strip() if code_snippet else ""
            
            logger.info(f"CONTEXT FORMAT: Processing document: {title if title else 'Untitled'}")
            
            # Add document header with clear numbering
            content_parts.append(f"Document {i+1}: {title}")
            
            # Add clean code if available
            if code_snippet:
                clean_code = self._extract_comprehensive_code(code_snippet)
                if clean_code:
                    content_parts.append(f"Code Implementation:\n{clean_code}")
            
            # Add clean documentation if available
            if text:
                clean_text = self._extract_comprehensive_text(text)
                if clean_text:
                    content_parts.append(f"Documentation:\n{clean_text}")
            
            # Add separator for clarity
            if i < max_docs - 1:
                content_parts.append("=" * 50)
        
        # Join with clear separators
        final_context = "\n\n".join(content_parts)
        
        logger.info(f"CONTEXT FORMAT: Created context - {len(final_context)} chars")
        
        return final_context
    
    def _extract_comprehensive_code(self, code_snippet: str) -> str:
        """
        Extract comprehensive code content with improved preservation and cleaning.
        
        This method processes code snippets to remove artifacts while preserving
        the essential structure and content. It handles various formatting issues
        and ensures clean, readable code presentation.
        
        Args:
            code_snippet (str): Raw code snippet with potential artifacts
            
        Returns:
            str: Cleaned code snippet with preserved structure
            
        Cleaning Process:
            - Remove end-of-snippet markers and artifacts
            - Filter out comment lines and empty lines
            - Preserve up to 35 lines of meaningful code
            - Maintain original indentation and structure
            
        Example:
            >>> clean_code = pipeline._extract_comprehensive_code(raw_code)
            >>> print(f"Cleaned code: {len(clean_code.split())} lines")
        """
        if not code_snippet:
            return ""
        
        # Convert to string safely
        code_snippet = str(code_snippet)
        
        # Remove artifacts but preserve structure
        cleaned = code_snippet.replace("##END OF CODE SNIPLET##", "")
        cleaned = cleaned.replace("##END OF CODE SNIPPET##", "")
        cleaned = cleaned.replace("#END OF CODE", "")
        cleaned = cleaned.replace("DOCUMNET", "")
        cleaned = cleaned.replace("============", "")
        
        # Get meaningful lines preserving original structure
        lines = []
        for line in cleaned.split('\n'):
            if line.strip() and not line.strip().startswith('*'):
                lines.append(line)
        
        # Take up to 35 lines (increased from 25)
        essential_lines = lines[:35]
        
        return '\n'.join(essential_lines) if essential_lines else ""
    
    def _extract_comprehensive_text(self, text: str) -> str:
        """
        Extract comprehensive text content with improved preservation and paragraph structure.
        
        This method processes documentation text to remove artifacts while
        preserving meaningful content and paragraph structure. It handles
        various formatting issues and ensures clean, readable text presentation.
        
        Args:
            text (str): Raw documentation text with potential artifacts
            
        Returns:
            str: Cleaned text with preserved paragraph structure
            
        Processing Steps:
            - Remove document artifacts and markers
            - Preserve paragraph boundaries and structure
            - Take up to 12 paragraphs for comprehensive coverage
            - Maintain readability while removing noise
            
        Example:
            >>> clean_text = pipeline._extract_comprehensive_text(raw_text)
            >>> print(f"Cleaned text: {len(clean_text.split())} words")
        """
        if not text:
            return ""
        
        # Convert to string safely
        text = str(text)
        
        # Remove artifacts
        cleaned = text.replace("DOCUMNET", "DOCUMENT")
        cleaned = cleaned.replace("============", "")
        cleaned = cleaned.replace("END ==============", "")
        
        # Preserve paragraph structure better
        paragraphs = []
        current_paragraph = []
        
        for line in cleaned.split('\n'):
            line = line.strip()
            if not line:
                if current_paragraph:
                    paragraphs.append(' '.join(current_paragraph))
                    current_paragraph = []
            else:
                current_paragraph.append(line)
        
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(' '.join(current_paragraph))
        
        # Take up to 12 paragraphs (increased from 10)
        result_paragraphs = paragraphs[:12]
        
        return '\n\n'.join(result_paragraphs) if result_paragraphs else ""
    
    def set_collection_type(self, collection_type: str) -> str:
        """
        Set document collection type for retrieval operations with validation.
        
        This method switches the active document collection for subsequent queries,
        allowing the pipeline to search different types of SAP ABAP documentation
        (e.g., classes vs reports).
        
        Args:
            collection_type (str): Collection type identifier ("classes", "reports", etc.)
            
        Returns:
            str: Actual collection name used by the retriever
            
        Example:
            >>> collection_name = pipeline.set_collection_type("classes")
            >>> print(f"Now using: {collection_name}")
        """
        if collection_type in COLLECTIONS:
            self.current_collection_type = collection_type
            collection_name = self.retriever.set_collection(collection_type)
            logger.info(f"Collection set to: {collection_type} ({collection_name})")
            return collection_name
        else:
            logger.warning(f"Unknown collection type: {collection_type}. Using default.")
            self.current_collection_type = DEFAULT_COLLECTION_TYPE
            collection_name = self.retriever.set_collection(DEFAULT_COLLECTION_TYPE)
            return collection_name
    
    def get_current_collection_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the current collection and pipeline state.
        
        This method returns detailed information about the active collection,
        enhancement settings, and pipeline configuration for monitoring and
        debugging purposes.
        
        Returns:
            Dict[str, Any]: Collection and pipeline information containing:
                - type (str): Collection type identifier
                - name (str): Actual collection name
                - description (str): Collection description
                - enhancement settings and generation parameters
                - timestamps and configuration details
                
        Example:
            >>> info = pipeline.get_current_collection_info()
            >>> print(f"Collection: {info['description']}")
            >>> print(f"Enhancement: {info['enhancement_mode']}")
        """
        collection_type = self.current_collection_type
        collection_config = COLLECTIONS[collection_type]
        collection_name = collection_config["name"]
        collection_description = collection_config["description"]
        
        return {
            "type": collection_type,
            "name": collection_name,
            "description": collection_description,
            "anti_hallucination_enabled": True,
            "query_enhancement_enabled": self.query_enhancement_enabled,
            "enhancement_mode": getattr(self.query_enhancer, 'enhancement_mode', 'unknown'),
            "max_tokens": self.max_tokens,
            "formatting_mode": self.formatting_mode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _analyze_query_type(self, query: str) -> Dict[str, Any]:
        """
        Enhanced query type analysis for appropriate prompt selection and processing strategy.
        
        This method performs comprehensive analysis of the input query to classify
        its type, complexity, and characteristics. This analysis guides prompt
        selection, retrieval strategies, and processing optimizations.
        
        Args:
            query (str): Input query to analyze
            
        Returns:
            Dict[str, Any]: Comprehensive query analysis containing:
                - Query type flags (method, interface, parameter, etc.)
                - Complexity indicators
                - Special pattern detection
                - Processing recommendations
                
        Query Types Detected:
            - Method queries: Function and procedure documentation
            - Interface queries: Interface implementation and structure
            - Parameter queries: Method parameters and argument analysis
            - Implementation queries: Usage examples and code patterns
            - Comparison queries: Feature and option comparisons
            - Error handling queries: Exception management and debugging
            - Situational queries: Complex scenario-based questions
            
        Example:
            >>> analysis = pipeline._analyze_query_type(
            ...     "What parameters does the submit method accept?"
            ... )
            >>> print(analysis["is_parameter_query"])  # True
            >>> print(analysis["is_method_query"])     # True
        """
        query_lower = query.lower()
        
        analysis = {
            "is_method_query": False,
            "is_interface_query": False,
            "is_structure_query": False,
            "is_parameter_query": False,
            "is_comparison_query": False,
            "is_situational_query": False,
            "is_complex_query": False,
            "is_implementation_query": False,
            "is_error_query": False,
            "query_length": len(query.split()),
            "contains_code_review": "code review" in query_lower,
            "contains_debugging": "debug" in query_lower or "debugging" in query_lower,
            "contains_specific_method": False
        }
        
        # Method queries
        method_keywords = ["method", "methods", "implement", "implementing", "function", "functions", "procedure", "initialize"]
        analysis["is_method_query"] = any(keyword in query_lower for keyword in method_keywords)
        
        # Interface queries
        interface_keywords = ["interface", "interfaces", "implement", "implements"]
        analysis["is_interface_query"] = any(keyword in query_lower for keyword in interface_keywords)
        
        # Structure queries
        structure_keywords = ["interface", "class", "method", "inherit", "implement", "extend"]
        analysis["is_structure_query"] = any(keyword in query_lower for keyword in structure_keywords)
        
        # Parameter queries
        parameter_keywords = ["parameter", "parameters", "accepts", "input", "output", "what does"]
        analysis["is_parameter_query"] = any(keyword in query_lower for keyword in parameter_keywords)
        
        # Comparison queries
        comparison_keywords = ["difference", "compare", "versus", "vs", "between", "contrast"]
        analysis["is_comparison_query"] = any(keyword in query_lower for keyword in comparison_keywords)
        
        # Implementation queries
        implementation_keywords = ["how to", "implement", "implementation", "usage", "use", "example"]
        analysis["is_implementation_query"] = any(keyword in query_lower for keyword in implementation_keywords)
        
        # Error handling queries
        error_keywords = ["error", "exception", "handling", "catch", "raise", "problem"]
        analysis["is_error_query"] = any(keyword in query_lower for keyword in error_keywords)
        
        # Situational queries
        situational_keywords = ["as part of", "i need to", "how do i", "what happens when", "scenario", "code review", "debugging", "understand what"]
        analysis["is_situational_query"] = any(keyword in query_lower for keyword in situational_keywords)
        
        # Check for specific method names
        specific_methods = ["escape_special_characters", "submit_success", "initialize"]
        analysis["contains_specific_method"] = any(method in query_lower for method in specific_methods)
        
        # Complex query detection
        analysis["is_complex_query"] = (
            analysis["query_length"] > 12 or
            analysis["is_situational_query"] or
            analysis["contains_code_review"] or
            analysis["is_comparison_query"] or
            (analysis["is_parameter_query"] and analysis["is_implementation_query"])
        )
        
        return analysis
    
    def _select_system_prompt(self, query_analysis: Dict[str, Any]) -> str:
        """
        Select the most appropriate system prompt based on enhanced query analysis.
        
        This method chooses the optimal system prompt for response generation
        based on the query type and characteristics. Different prompts provide
        specialized instructions for different types of SAP ABAP documentation
        queries.
        
        Args:
            query_analysis (Dict[str, Any]): Query analysis results from _analyze_query_type
            
        Returns:
            str: Selected system prompt optimized for the query type
            
        Prompt Selection Logic:
            1. Parameter queries ? Enhanced method prompt (strictest validation)
            2. Method queries ? Method-specific prompt
            3. Interface queries ? Interface-specific prompt
            4. Structure queries ? Structure-specific prompt
            5. Implementation queries ? Enhanced response prompt
            6. Error queries ? Enhanced method prompt
            7. Situational queries ? Enhanced response prompt
            8. Default ? Enhanced response prompt
            
        Example:
            >>> prompt = pipeline._select_system_prompt(query_analysis)
            >>> print(f"Selected prompt: {prompt[:100]}...")
        """
        # Parameter queries get the strictest validation
        if query_analysis["is_parameter_query"]:
            logger.info("Using enhanced method prompt for parameter query")
            return ENHANCED_METHOD_SYSTEM_PROMPT
        
        # Method queries with specific methods
        elif query_analysis["is_method_query"] or query_analysis["contains_specific_method"]:
            logger.info("Using method-specific prompt")
            return ENHANCED_METHOD_SYSTEM_PROMPT
        
        # Interface queries
        elif query_analysis["is_interface_query"]:
            logger.info("Using interface-specific prompt")
            return INTERFACE_SYSTEM_PROMPT
        
        # Structure queries
        elif query_analysis["is_structure_query"]:
            logger.info("Using structure-specific prompt")
            return STRUCTURE_SYSTEM_PROMPT
        
        # Implementation queries
        elif query_analysis["is_implementation_query"]:
            logger.info("Using enhanced response prompt for implementation query")
            return ENHANCED_RESPONSE_PROMPT
        
        # Error handling queries
        elif query_analysis["is_error_query"]:
            logger.info("Using enhanced method prompt for error query")
            return ENHANCED_METHOD_SYSTEM_PROMPT
        
        # Situational queries
        elif query_analysis["is_situational_query"]:
            logger.info("Using enhanced response prompt for situational query")
            return ENHANCED_RESPONSE_PROMPT
        
        # Default to enhanced prompt for better responses
        else:
            logger.info("Using enhanced response prompt as default")
            return ENHANCED_RESPONSE_PROMPT
    
    async def process_query(self, 
                            query: str, 
                            conversation_history: Optional[List[Dict[str, str]]] = None,
                            use_hybrid: bool = USE_HYBRID_DEFAULT, 
                            use_reranker: bool = USE_RERANKER_DEFAULT, 
                            top_k: int = DEFAULT_TOP_K,
                            collection_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Complete Enhanced query processing with comprehensive pipeline orchestration.
        
        This method orchestrates the complete RAG workflow from query enhancement
        through response generation, with comprehensive monitoring, validation,
        and quality controls. It provides detailed logging and returns extensive
        metadata about the processing pipeline.
        
        Args:
            query (str): User input query to process
            conversation_history (Optional[List[Dict[str, str]]]): Previous conversation
                messages for context-aware processing
            use_hybrid (bool): Whether to use hybrid retrieval (dense + sparse)
            use_reranker (bool): Whether to apply cross-encoder reranking
            top_k (int): Number of top documents to return in final results
            collection_type (Optional[str]): Specific collection to search
            
        Returns:
            Dict[str, Any]: Comprehensive processing results containing:
                - Query information (original, enhanced, analysis)
                - Retrieval metrics (time, document counts, scores)
                - Generation details (response, confidence, system prompt)
                - Validation results (hallucination check, risk assessment)
                - Performance metrics (timing, throughput, resource usage)
                - Quality indicators (confidence levels, source relevance)
                
        Processing Pipeline:
            1. Collection setup and configuration
            2. Query enhancement with intelligent context addition
            3. Query type analysis and complexity assessment
            4. Adaptive retrieval with query-specific limits
            5. Cross-encoder reranking for relevance refinement
            6. Document filtering and quality validation
            7. Context formatting with structure preservation
            8. System prompt selection based on query analysis
            9. Response generation with appropriate parameters
            10. Anti-hallucination validation and risk assessment
            11. Confidence scoring and quality metrics
            12. Comprehensive result compilation and logging
            
        Example:
            >>> result = await pipeline.process_query(
            ...     "What parameters does the submit method accept?",
            ...     use_hybrid=True,
            ...     use_reranker=True,
            ...     top_k=5
            ... )
            >>> 
            >>> print(f"Enhanced Query: {result['enhanced_query']}")
            >>> print(f"Confidence: {result['confidence_level']}")
            >>> print(f"Sources: {result['context_count']}")
            >>> print(f"Processing Time: {result['total_processing_time']:.2f}s")
        """
        start_time = time.time()
        
        # Set collection if specified
        if collection_type:
            self.set_collection_type(collection_type)
        
        collection_info = self.get_current_collection_info()
        logger.info(f"ENHANCED: Processing query with query enhancement: '{query[:100]}...'")
        logger.info(f"Collection: {collection_info['name']}")
        logger.info(f"Parameters: hybrid={use_hybrid}, reranker={use_reranker}, top_k={top_k}")
        logger.info(f"Enhancement mode: {collection_info['enhancement_mode']}")
        
        # Initialize comprehensive result dictionary
        result_dict = {
            "query": query,
            "original_query": query,
            "enhanced_query": query,
            "query_length": len(query),
            "query_word_count": len(query.split()),
            "collection_info": collection_info,
            "is_follow_up": conversation_history is not None and len(conversation_history) > 0,
            "query_enhancement_used": False,
            "anti_hallucination_enabled": True,
            "processing_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "processing_start_time": start_time
        }
        
        try:
            # Enhanced query enhancement if enabled
            enhanced_query = query
            if self.query_enhancement_enabled and hasattr(self, 'query_enhancer'):
                logger.info("ENHANCED: Applying intelligent query enhancement...")
                enhancement_result = await self.query_enhancer.enhance_query(query)
                
                enhanced_query = enhancement_result.get("enhanced_query", query)
                result_dict.update({
                    "original_query": enhancement_result.get("original_query", query),
                    "enhanced_query": enhanced_query,
                    "query_enhancement_used": enhancement_result.get("is_enhanced", False),
                    "enhancement_mode": enhancement_result.get("enhancement_mode", "unknown"),
                    "is_out_of_context": enhancement_result.get("is_out_of_context", False),
                    "query_analysis": enhancement_result.get("query_analysis", {}),
                    "enhancement_details": enhancement_result.get("enhancement_details", {})
                })
                
                # Handle out of context queries
                if enhancement_result.get("is_out_of_context", False):
                    logger.info("Query determined to be out of context")
                    result_dict.update({
                        "response": OUT_OF_SCOPE_PROMPT,
                        "confidence_level": "LOW",
                        "context_count": 0,
                        "contexts": [],
                        "has_relevant_results": False,
                        "out_of_scope": True
                    })
                    return result_dict
                
                logger.info(f"ENHANCED: Query enhancement result - Enhanced: {enhancement_result.get('is_enhanced', False)}")
                if enhancement_result.get('is_enhanced', False):
                    logger.info(f"  Original: '{query}'")
                    logger.info(f"  Enhanced: '{enhanced_query}'")
            else:
                logger.info("ENHANCED: Query enhancement disabled")
            
            # Enhanced query type analysis for prompt selection
            query_analysis = self._analyze_query_type(enhanced_query)
            result_dict.update(query_analysis)
            
            logger.info(f"ENHANCED: Query analysis: {query_analysis}")
            
            # Determine retrieval limits based on query complexity and type
            base_limit = 25  # Increased base limit
            
            if query_analysis["is_parameter_query"] or query_analysis["contains_specific_method"]:
                retrieval_limit = max(35, top_k * 7)  # More context for specific queries
                logger.info(f"ENHANCED: Parameter/specific method query - retrieval limit: {retrieval_limit}")
            elif query_analysis["is_situational_query"] or query_analysis["is_complex_query"]:
                retrieval_limit = max(45, top_k * 9)  # Even more for complex scenarios
                logger.info(f"ENHANCED: Situational/complex query - retrieval limit: {retrieval_limit}")
            elif query_analysis["is_comparison_query"]:
                retrieval_limit = max(30, top_k * 6)  # Multiple items to compare
                logger.info(f"ENHANCED: Comparison query - retrieval limit: {retrieval_limit}")
            elif query_analysis["is_implementation_query"]:
                retrieval_limit = max(30, top_k * 6)  # Implementation examples
                logger.info(f"ENHANCED: Implementation query - retrieval limit: {retrieval_limit}")
            else:
                retrieval_limit = max(base_limit, top_k * 5)
                logger.info(f"ENHANCED: Standard query - retrieval limit: {retrieval_limit}")
            
            result_dict["retrieval_limit_used"] = retrieval_limit
            
            # Use enhanced query for retrieval
            logger.info(f"ENHANCED: Retrieving documents for query: '{enhanced_query}'")
            
            retrieval_start = time.time()
            original_results = self.retriever.retrieve(
                query=enhanced_query,  # Use enhanced query
                limit=retrieval_limit,
                use_hybrid=use_hybrid
            )
            retrieval_time = time.time() - retrieval_start
            
            result_dict["retrieval_time"] = retrieval_time
            result_dict["documents_retrieved"] = len(original_results)
            
            logger.info(f"ENHANCED: Retrieved {len(original_results)} documents in {retrieval_time:.2f}s")
            
            # Log retrieved documents for debugging
            for i, result in enumerate(original_results[:5]):
                title = result.payload.get("title", "No title")
                score = getattr(result, 'score', 0.0)
                logger.info(f"ENHANCED: Retrieved {i+1}: '{title}' (score: {score:.4f})")
            
            # Enhanced reranking if requested
            rerank_start = time.time()
            if use_reranker and self.retriever.has_reranker and len(original_results) > 0:
                context_results = self.retriever.rerank_results(
                    query=enhanced_query,  # Use enhanced query for reranking too
                    results=original_results, 
                    top_k=top_k * 5  # Get more for filtering
                )
                result_dict["reranking_used"] = True
                logger.info(f"ENHANCED: Reranking applied, got {len(context_results)} results")
            else:
                context_results = [(result, getattr(result, 'score', 0.0)) for result in original_results[:top_k * 4]]
                result_dict["reranking_used"] = False
                logger.info(f"ENHANCED: No reranking, using {len(context_results)} results")
            
            rerank_time = time.time() - rerank_start
            result_dict["rerank_time"] = rerank_time
            
            # Log reranked results
            for i, (result, score) in enumerate(context_results[:5]):
                title = result.payload.get("title", "No title")
                logger.info(f"ENHANCED: After rerank/selection {i+1}: '{title}' (score: {score:.4f})")
            
            # Enhanced filtering for relevance
            filtering_start = time.time()
            filtered_context_results = self._filter_relevant_documents(context_results)
            filtering_time = time.time() - filtering_start
            
            result_dict["filtering_time"] = filtering_time
            result_dict["documents_after_filtering"] = len(filtered_context_results)
            result_dict["documents_filtered_out"] = len(context_results) - len(filtered_context_results)
            
            # Check if we have relevant results
            has_relevant_results = len(filtered_context_results) > 0
            
            if has_relevant_results:
                # Calculate enhanced confidence level
                confidence_level = self._calculate_confidence_level(filtered_context_results, enhanced_query)
                
                # Create enhanced contexts for response generation
                contexts = []
                for i, (result, score) in enumerate(filtered_context_results):
                    payload = result.payload
                    
                    # Safe extraction with proper null handling
                    title = payload.get("title") or ""
                    filename = payload.get("filename") or ""
                    text = payload.get("text") or ""
                    code_snippet = payload.get("code_snippet") or ""
                    
                    # Convert to strings safely
                    title = str(title) if title else ""
                    filename = str(filename) if filename else ""
                    text = str(text) if text else ""
                    code_snippet = str(code_snippet) if code_snippet else ""
                    
                    context = {
                        "id": i+1,
                        "title": title,
                        "filename": filename,
                        "text": text,
                        "code_snippet": code_snippet,
                        "score": score,
                        "abs_score": abs(score),
                        "relevant": abs(score) >= MIN_DOCUMENT_SCORE,
                        "content_length": len(text),
                        "has_code": bool(code_snippet.strip()),
                        "source_type": "code" if code_snippet.strip() else "documentation"
                    }
                    contexts.append(context)
                
                # Format context with enhanced boundaries
                formatted_context = self._format_context_with_strict_boundaries(filtered_context_results)
                
                # Select most appropriate system prompt
                system_prompt = self._select_system_prompt(query_analysis)
                
                # Generate response with enhanced monitoring
                generation_start = time.time()
                
                # Set generation parameters if not already set
                if hasattr(self.generator, 'set_generation_config'):
                    self.generator.set_generation_config(
                        max_new_tokens=self.max_tokens,
                        formatting_mode=self.formatting_mode
                    )
                
                if result_dict["is_follow_up"]:
                    response = await self.generator.generate_with_history(
                        enhanced_query, formatted_context, conversation_history, system_prompt=system_prompt
                    )
                else:
                    response = await self.generator.generate(
                        enhanced_query, formatted_context, system_prompt=system_prompt
                    )
                
                generation_time = time.time() - generation_start
                result_dict["generation_time"] = generation_time
                
                logger.info(f"ENHANCED: Response generated in {generation_time:.2f}s")
                
                # Extract confidence level from response
                extracted_confidence, cleaned_response = self._extract_confidence_level(response)
                
                # Use calculated confidence if not properly extracted
                final_confidence = extracted_confidence if extracted_confidence in ["HIGH", "MEDIUM", "LOW"] else confidence_level
                
                # Enhanced hallucination detection
                hallucination_start = time.time()
                hallucination_check = self._detect_potential_hallucination(cleaned_response, contexts, enhanced_query)
                hallucination_time = time.time() - hallucination_start
                
                result_dict["hallucination_check_time"] = hallucination_time
                
                # Adjust confidence based on hallucination risk (more lenient)
                original_confidence = final_confidence
                if hallucination_check["risk_level"] == "HIGH":
                    final_confidence = "LOW"
                    logger.warning(f"High hallucination risk detected - confidence reduced from {original_confidence} to LOW")
                elif hallucination_check["risk_level"] == "MEDIUM" and final_confidence == "HIGH":
                    final_confidence = "MEDIUM"
                    logger.warning(f"Medium hallucination risk detected - confidence reduced from HIGH to MEDIUM")
                
                # Update comprehensive results
                result_dict.update({
                    "response": cleaned_response,
                    "full_response": response,
                    "confidence_level": final_confidence,
                    "original_confidence": original_confidence,
                    "calculated_confidence": confidence_level,
                    "extracted_confidence": extracted_confidence,
                    "context_count": len(filtered_context_results),
                    "contexts": contexts,
                    "has_relevant_results": True,
                    "hallucination_check": hallucination_check,
                    "filtered_low_relevance_docs": len(context_results) - len(filtered_context_results),
                    "system_prompt_used": system_prompt[:100] + "..." if len(system_prompt) > 100 else system_prompt,
                    "response_word_count": len(cleaned_response.split()),
                    "response_character_count": len(cleaned_response)
                })
                
            else:
                # No relevant results found
                logger.warning("ENHANCED: No relevant results found - using out of scope response")
                
                if result_dict["is_follow_up"]:
                    response = OUT_OF_SCOPE_CONVERSATION_PROMPT
                else:
                    response = OUT_OF_SCOPE_PROMPT
                
                result_dict.update({
                    "response": response,
                    "full_response": response,
                    "confidence_level": "LOW",
                    "context_count": 0,
                    "contexts": [],
                    "has_relevant_results": False,
                    "hallucination_check": {
                        "risk_level": "LOW", 
                        "risk_score": 0,
                        "warnings": ["No relevant documentation found"],
                        "suspicious_phrases": [],
                        "undocumented_parameters": [],
                        "undocumented_methods": []
                    },
                    "out_of_scope": True
                })
            
            # Calculate total processing time
            total_time = time.time() - start_time
            result_dict["total_processing_time"] = total_time
            
            # Log comprehensive processing summary
            logger.info(f"ENHANCED: Query processing completed in {total_time:.2f}s")
            logger.info(f"  Enhancement used: {result_dict.get('query_enhancement_used', False)}")
            logger.info(f"  Confidence: {result_dict.get('confidence_level', 'UNKNOWN')}")
            logger.info(f"  Contexts: {result_dict.get('context_count', 0)}")
            logger.info(f"  Risk: {result_dict.get('hallucination_check', {}).get('risk_level', 'UNKNOWN')}")
            
            return result_dict
            
        except Exception as e:
            logger.error(f"ENHANCED: Error processing query: {e}", exc_info=True)
            
            # Return comprehensive error information
            error_time = time.time() - start_time
            result_dict.update({
                "error": str(e),
                "error_type": type(e).__name__,
                "response": f"Error processing query: {str(e)}",
                "confidence_level": "LOW",
                "context_count": 0,
                "contexts": [],
                "has_relevant_results": False,
                "total_processing_time": error_time,
                "hallucination_check": {
                    "risk_level": "LOW", 
                    "risk_score": 0,
                    "warnings": [f"Processing error: {str(e)}"],
                    "suspicious_phrases": [],
                    "undocumented_parameters": [],
                    "undocumented_methods": []
                }
            })
            return result_dict


# Backward compatibility alias
RAGPipeline = EnhancedRAGPipeline


def create_enhanced_pipeline():
    """
    Factory function to create an enhanced RAG pipeline instance.
    
    This function provides a convenient way to create and initialize
    an Enhanced RAG Pipeline with default configurations.
    
    Returns:
        EnhancedRAGPipeline: Configured pipeline instance ready for use
        
    Example:
        >>> pipeline = create_enhanced_pipeline()
        >>> result = await pipeline.process_query("What is ABAP class?")
    """
    return EnhancedRAGPipeline()


# Export important classes and functions
__all__ = [
    'EnhancedRAGPipeline',
    'RAGPipeline',  # Alias for backward compatibility
    'create_enhanced_pipeline'
]


if __name__ == "__main__":
    # Test the enhanced pipeline if run directly
    async def test_enhanced_pipeline():
        """
        Test the enhanced pipeline with comprehensive validation.
        
        This function performs a complete test of the Enhanced RAG Pipeline
        including initialization, query processing, and result validation.
        """
        logger.info("Testing Enhanced RAG Pipeline...")
        
        try:
            pipeline = EnhancedRAGPipeline()
            print("? Enhanced Pipeline initialized successfully")
            
            # Quick test query
            test_query = "What is the purpose of the escape_special_characters method?"
            result = await pipeline.process_query(test_query)
            
            print(f"Test Query: {test_query}")
            print(f"Enhanced Query: {result.get('enhanced_query', 'N/A')}")
            print(f"Enhancement Used: {result.get('query_enhancement_used', False)}")
            print(f"Response: {result.get('response', 'No response')[:200]}...")
            print(f"Confidence: {result.get('confidence_level', 'Unknown')}")
            print(f"Context Count: {result.get('context_count', 0)}")
            
            logger.info("? Enhanced Pipeline test completed successfully")
        
        except Exception as e:
            logger.error(f"? Enhanced Pipeline test failed: {e}")
    
    # Run test
    import asyncio
    asyncio.run(test_enhanced_pipeline())