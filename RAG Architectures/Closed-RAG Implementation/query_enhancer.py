#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Query Enhancement Module for RAG Pipeline.

This module provides sophisticated query enhancement capabilities for the RAG pipeline
to improve retrieval quality and response accuracy. The module implements two main
enhancement strategies:

1. Query Rewriting: Reformulates user queries to be more specific, detailed, and
   optimized for document retrieval in SAP ABAP documentation contexts.

2. Sub-query Decomposition: Breaks down complex, multi-faceted queries into simpler,
   focused sub-queries that can be processed independently and then combined.

The module also includes context validation to filter out queries that are not
relevant to SAP ABAP documentation, ensuring the system focuses on its intended
domain of expertise.

Features:
    - Intelligent query rewriting for improved retrieval accuracy
    - Complex query decomposition into manageable sub-queries
    - Context validation for SAP ABAP relevance
    - Robust error handling with fallback to original queries
    - Comprehensive logging for debugging and monitoring

Classes:
    QueryEnhancer: Main class coordinating all query enhancement operations

Dependencies:
    - LLM generator component for query transformation
    - asyncio for asynchronous operation
    - Regular expressions for pattern matching and extraction

Author: SAP ABAP RAG Team
Version: 1.0.0
Date: 2025
License: MIT
"""

# Standard library imports
import asyncio
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QueryEnhancer:
    """
    Component that enhances queries to improve RAG retrieval and response quality.
    
    This class provides comprehensive query enhancement capabilities including
    rewriting for clarity, decomposition for complex queries, and context validation
    for domain relevance. It integrates with the RAG generator component to leverage
    LLM capabilities for intelligent query transformation.
    
    The enhancement process follows these principles:
    - Preserve technical terms and specific identifiers exactly
    - Focus on SAP ABAP documentation domain
    - Maintain query intent while improving clarity
    - Provide graceful fallbacks for enhancement failures
    
    Attributes:
        generator: The RAG generator component providing LLM access
        
    Examples:
        >>> enhancer = QueryEnhancer(generator)
        >>> result = await enhancer.enhance_query("How do ABAP classes work?")
        >>> print(f"Enhanced: {result['enhanced_query']}")
        >>> if result['is_decomposed']:
        ...     for sub_query in result['sub_queries']:
        ...         print(f"Sub-query: {sub_query}")
    """
    
    def __init__(self, generator) -> None:
        """
        Initialize the query enhancer with a generator component.
        
        Args:
            generator: The RAG generator component that provides access to the LLM
                      for query transformation operations. Must have a 
                      _generate_with_client method for LLM interactions.
                      
        Note:
            The generator is expected to be already initialized and ready for
            LLM generation tasks. The enhancer will use it for both query
            rewriting and decomposition operations.
        """
        logger.info("Initializing Query Enhancer")
        self.generator = generator
        
    async def rewrite_query(self, original_query: str) -> str:
        """
        Rewrite the original query to be more specific and detailed.
        
        This method uses LLM capabilities to reformulate user queries for better
        retrieval performance in SAP ABAP documentation. The rewriting process:
        - Removes irrelevant personal information or anecdotes
        - Makes queries more specific and focused
        - Preserves technical terms and identifiers exactly
        - Filters out completely off-topic queries
        
        Args:
            original_query: The user's original query string. Should be non-empty
                           text representing an information need.
                           
        Returns:
            Rewritten query string optimized for retrieval, or "OUT_OF_CONTEXT"
            if the query is not relevant to SAP ABAP documentation, or the
            original query if rewriting fails.
            
        Examples:
            >>> enhancer = QueryEnhancer(generator)
            >>> rewritten = await enhancer.rewrite_query(
            ...     "I'm working on a project and need to know about ABAP classes"
            ... )
            >>> print(rewritten)  # "ABAP class structure and implementation details"
            
            >>> rewritten = await enhancer.rewrite_query("What's my name?")
            >>> print(rewritten)  # "OUT_OF_CONTEXT"
            
        Note:
            The rewriting preserves exact class names, method names, and interfaces
            mentioned in the original query to maintain technical accuracy.
        """
        logger.info(f"Rewriting query: '{original_query}'")
        
        # Create system prompt for query rewriting
        system_prompt = """You are an AI assistant tasked with reformulating user queries to improve retrieval in a RAG system.
Given the original query, rewrite it to be more specific, detailed, and likely to retrieve relevant information.
If the query contains irrelevant information like personal anecdotes, remove those parts.
If the query is completely out of context for SAP ABAP documentation (like personal questions or non-technical questions), 
just return the text "OUT_OF_CONTEXT" without any other text.

Important rules:
1. Focus only on SAP ABAP code documentation in your rewrite
2. Keep the rewritten query concise (1-2 sentences)
3. Only rewrite technical queries related to SAP ABAP code
4. If the query mentions specific class names, method names, or interfaces, preserve them exactly
5. If the query asks about implementation details, focus on those in the rewrite
6. Do not add any explanations or prefixes - just provide the rewritten query
7. Return only the rewritten query and nothing else
8. If the query is completely unrelated to SAP ABAP (like "What is my name?" or historical questions), 
   just return the exact text "OUT_OF_CONTEXT" without any additional text
"""
        
        # Prepare the message with the original query
        messages = [
            {"role": "user", "content": f"Original query: {original_query}\n\nRewritten query:"}
        ]
        
        # Generate rewritten query
        try:
            response = await self.generator._generate_with_client(messages, system_prompt)
            
            # Check if the response indicates the query is out of context
            if response and response.strip().upper() == "OUT_OF_CONTEXT":
                logger.info("Query identified as out of context")
                return "OUT_OF_CONTEXT"
                
            # Clean up the response
            rewritten_query = response.strip()
            logger.info(f"Rewritten query: '{rewritten_query}'")
            return rewritten_query
        except Exception as e:
            logger.error(f"Error rewriting query: {e}")
            # Fall back to original query
            return original_query

    async def decompose_query(self, original_query: str) -> List[str]:
        """
        Decompose a complex query into simpler sub-queries.
        
        This method breaks down complex, multi-faceted queries into 2-4 simpler
        sub-queries that can be processed independently. Each sub-query focuses
        on a single aspect of the original query, enabling more targeted retrieval
        and better overall coverage of the user's information need.
        
        Args:
            original_query: The user's original complex query string. Should contain
                           multiple aspects or questions that can be separated.
                           
        Returns:
            List of simpler sub-query strings, each focusing on one aspect of the
            original query. Returns ["OUT_OF_CONTEXT"] if the query is not relevant
            to SAP ABAP, or [original_query] if decomposition fails or the query
            is already simple enough.
            
        Examples:
            >>> enhancer = QueryEnhancer(generator)
            >>> sub_queries = await enhancer.decompose_query(
            ...     "How do ABAP classes work and what are the main methods?"
            ... )
            >>> for i, sub_query in enumerate(sub_queries, 1):
            ...     print(f"{i}. {sub_query}")
            # 1. How do ABAP classes work and what is their structure?
            # 2. What are the main methods in ABAP classes?
            
        Note:
            The decomposition preserves exact technical terms, class names, and
            method names from the original query. If the original query is already
            simple, it returns the original query as a single-item list.
        """
        logger.info(f"Decomposing query: '{original_query}'")
        
        # Create system prompt for sub-query decomposition
        system_prompt = """You are an AI assistant tasked with breaking down complex queries into simpler sub-queries for a RAG system.
Given the original query about SAP ABAP code documentation, decompose it into 2-4 simpler sub-queries that, when answered together, would provide a comprehensive response to the original query.

Important rules:
1. Only return the numbered list of sub-queries, nothing else
2. Each sub-query should be focused on a single aspect of the original query
3. Ensure each sub-query is clear and specific
4. If the original query is already simple, just return the text "ALREADY_SIMPLE" without any other text
5. If the query is completely unrelated to SAP ABAP (like personal questions), just return the text "OUT_OF_CONTEXT" without any other text
6. Format the response with one sub-query per line, each preceded by a number and period (e.g., "1. First sub-query")
7. Preserve exact class names, method names, and interfaces in the sub-queries

Example:
Original query: "What are the two sources from which transaction parameters can be retrieved in the CHECK_SIGNATURES_COMPATIBLE method, and what exception is caught during the error handling process?"
Sub-queries:
1. What are the two sources from which transaction parameters can be retrieved in the CHECK_SIGNATURES_COMPATIBLE method?
2. What exception is caught during the error handling process in the CHECK_SIGNATURES_COMPATIBLE method?
"""
        
        # Prepare the message with the original query
        messages = [
            {"role": "user", "content": f"Original query: {original_query}\n\nSub-queries:"}
        ]
        
        # Generate sub-queries
        try:
            response = await self.generator._generate_with_client(messages, system_prompt)
            
            # Check if the response indicates the query is already simple or out of context
            if response and (response.strip().upper() == "ALREADY_SIMPLE" or response.strip().upper() == "OUT_OF_CONTEXT"):
                logger.info(f"Query identified as {response.strip().upper()}")
                if response.strip().upper() == "OUT_OF_CONTEXT":
                    return ["OUT_OF_CONTEXT"]
                else:
                    # Return the original query as the only sub-query
                    return [original_query]
                    
            # Extract sub-queries using regex to handle various formats
            sub_queries = []
            pattern = r'^\s*\d+\.\s*(.+)$'
            for line in response.split('\n'):
                match = re.match(pattern, line.strip())
                if match:
                    sub_query = match.group(1).strip()
                    if sub_query:
                        sub_queries.append(sub_query)
            
            # If no sub-queries were found but there's a response,
            # try an alternative approach - look for each line that's not "Sub-queries:"
            if not sub_queries and response:
                sub_queries = [q.strip() for q in response.split('\n') 
                              if q.strip() and not q.strip().lower().startswith('sub-queries:')]
                
            logger.info(f"Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")
            
            # If decomposition failed, return the original query as a single sub-query
            if not sub_queries:
                logger.warning("Query decomposition failed, using original query")
                return [original_query]
                
            return sub_queries
        except Exception as e:
            logger.error(f"Error decomposing query: {e}")
            # Fall back to original query
            return [original_query]
            
    async def is_query_complex(self, query: str) -> bool:
        """
        Determine if a query is complex and would benefit from decomposition.
        
        This method uses heuristic analysis to identify queries that contain
        multiple distinct parts or aspects that could be better addressed by
        breaking them into simpler sub-queries.
        
        Args:
            query: The user's query string to analyze for complexity
            
        Returns:
            True if the query is complex and should be decomposed, False if it's
            simple enough to be processed as-is
            
        Examples:
            >>> enhancer = QueryEnhancer(generator)
            >>> is_complex = await enhancer.is_query_complex(
            ...     "How do ABAP classes work and what are the main methods?"
            ... )
            >>> print(is_complex)  # True
            
            >>> is_complex = await enhancer.is_query_complex("What is an ABAP class?")
            >>> print(is_complex)  # False
            
        Note:
            The complexity detection uses multiple heuristics including:
            - Presence of conjunctions (and, or, also, etc.)
            - Multiple question marks
            - Query length
            - Multiple distinct conceptual parts
        """
        # Simple heuristics for detecting complex queries
        if "," in query and "and" in query.lower():
            return True
            
        if "?" in query and query.count("?") > 1:
            return True
            
        if len(query.split()) > 15:  # Long queries might be complex
            return True
            
        conjunction_words = ["and", "or", "as well as", "along with", "also", "additionally"]
        if any(word in query.lower() for word in conjunction_words):
            # Check if the query has multiple distinct parts
            parts = re.split(r'\s+and\s+|\s+or\s+|\s+as well as\s+|\s+along with\s+|\s+also\s+|\s+additionally\s+', query.lower())
            if len(parts) > 1:
                return True
                
        return False
    
    async def is_out_of_context(self, query: str) -> bool:
        """
        Determine if a query is completely out of context for SAP ABAP documentation.
        
        This method performs domain validation to identify queries that are not
        relevant to SAP ABAP documentation and should not be processed by the
        RAG system. It uses keyword analysis and pattern matching to detect
        personal questions, general knowledge queries, and other off-topic content.
        
        Args:
            query: The user's query string to validate for domain relevance
            
        Returns:
            True if the query is out of context and should be rejected, False if
            it appears to be relevant to SAP ABAP documentation
            
        Examples:
            >>> enhancer = QueryEnhancer(generator)
            >>> is_out = await enhancer.is_out_of_context("What is my name?")
            >>> print(is_out)  # True
            
            >>> is_out = await enhancer.is_out_of_context("How do ABAP classes work?")
            >>> print(is_out)  # False
            
        Note:
            The method uses a combination of SAP ABAP keyword detection and
            personal question pattern matching to make the determination.
            It errs on the side of caution, preferring to process questionable
            queries rather than reject potentially valid ones.
        """
        # Keywords that might indicate SAP ABAP related queries
        sap_keywords = ["sap", "abap", "class", "method", "interface", "function", "module", 
                        "program", "report", "transaction", "table", "data element", "structure",
                        "domain", "bapi", "rfc", "view", "cl_", "if_", "alv", "badi", "enhancement",
                        "extension", "exit", "enhancement point", "bdc", "idoc", "message",
                        "exception", "parameter", "returning", "importing", "exporting", "tables",
                        "changing"]
                        
        # Query is very short (likely not a technical question)
        if len(query.split()) < 3:
            logger.info(f"Query '{query}' is very short, checking for context")
            
        # Check if query contains any SAP ABAP related keywords
        query_lower = query.lower()
        has_sap_keyword = any(keyword.lower() in query_lower for keyword in sap_keywords)
        
        # Check for personal or general knowledge questions
        personal_patterns = [
            r'\b(?:my|your|his|her|their)\s+name\b',
            r'\bwhat\s+(?:is|are)\s+(?:my|your)\b',
            r'\bwhen\s+(?:was|is|are|did)\s+(?:the)?\s*world\s+war\b',
            r'\bwho\s+(?:is|was|are)\s+(?:the)?\s*president\b',
            r'\bwhat\s+(?:is|was)\s+(?:the)?\s*capital\b',
            r'\bhow\s+(?:old|tall|young)\b',
            r'\bwhere\s+(?:is|are|was|were)\s+(?:the)?\s*(?:located|situated)\b'
        ]
        
        is_personal_question = any(re.search(pattern, query_lower) for pattern in personal_patterns)
        
        # If the query doesn't contain SAP keywords and might be a personal question,
        # it's likely out of context
        if is_personal_question or not has_sap_keyword:
            logger.info(f"Query might be out of context: '{query}'")
            return True
            
        return False
        
    async def enhance_query(self, original_query: str) -> Dict[str, Any]:
        """
        Apply all query enhancement techniques to improve retrieval.
        
        This is the main entry point for query enhancement that orchestrates
        all enhancement techniques including context validation, query rewriting,
        complexity analysis, and decomposition. It provides a comprehensive
        enhancement result with all relevant metadata.
        
        Args:
            original_query: The user's original query string to be enhanced
            
        Returns:
            Dictionary containing comprehensive enhancement information:
            - original_query (str): The user's original query
            - enhanced_query (str): The rewritten/improved query
            - is_out_of_context (bool): Whether query is off-topic
            - is_enhanced (bool): Whether query was actually rewritten
            - is_decomposed (bool): Whether query was broken into sub-queries
            - sub_queries (List[str]): List of sub-queries if decomposed
            
        Examples:
            >>> enhancer = QueryEnhancer(generator)
            >>> result = await enhancer.enhance_query(
            ...     "How do ABAP classes work and what are their methods?"
            ... )
            >>> print(f"Original: {result['original_query']}")
            >>> print(f"Enhanced: {result['enhanced_query']}")
            >>> print(f"Decomposed: {result['is_decomposed']}")
            >>> if result['is_decomposed']:
            ...     for sub_query in result['sub_queries']:
            ...         print(f"  - {sub_query}")
            
        Note:
            The enhancement process is designed to be robust with multiple
            fallback strategies. Even if individual enhancement steps fail,
            the method will return a valid result structure with the original
            query preserved.
        """
        logger.info(f"Enhancing query: '{original_query}'")
        
        # Check if the query is out of context
        is_out_of_context = await self.is_out_of_context(original_query)
        
        if is_out_of_context:
            logger.info(f"Query appears to be out of context: '{original_query}'")
            return {
                "original_query": original_query,
                "enhanced_query": original_query,
                "is_out_of_context": True,
                "is_enhanced": False,
                "sub_queries": []
            }
            
        # First, try to rewrite the query for better retrieval
        rewritten_query = await self.rewrite_query(original_query)
        
        # Check if the query was identified as out of context by the LLM
        if rewritten_query == "OUT_OF_CONTEXT":
            logger.info(f"Query identified as out of context by LLM: '{original_query}'")
            return {
                "original_query": original_query,
                "enhanced_query": original_query,
                "is_out_of_context": True,
                "is_enhanced": False,
                "sub_queries": []
            }
            
        # If the query is complex, try to decompose it
        is_complex = await self.is_query_complex(original_query)
        
        if is_complex:
            logger.info(f"Query identified as complex: '{original_query}'")
            # Use the rewritten query as input for decomposition
            sub_queries = await self.decompose_query(rewritten_query)
            
            # Check if any sub-query was identified as out of context
            if len(sub_queries) == 1 and sub_queries[0] == "OUT_OF_CONTEXT":
                logger.info(f"Query identified as out of context during decomposition: '{original_query}'")
                return {
                    "original_query": original_query,
                    "enhanced_query": original_query,
                    "is_out_of_context": True,
                    "is_enhanced": False,
                    "sub_queries": []
                }
                
            return {
                "original_query": original_query,
                "enhanced_query": rewritten_query,
                "is_out_of_context": False,
                "is_enhanced": True,
                "is_decomposed": True,
                "sub_queries": sub_queries
            }
        else:
            logger.info(f"Query is simple, using rewritten query: '{rewritten_query}'")
            return {
                "original_query": original_query,
                "enhanced_query": rewritten_query,
                "is_out_of_context": False,
                "is_enhanced": original_query != rewritten_query,
                "is_decomposed": False,
                "sub_queries": []
            }