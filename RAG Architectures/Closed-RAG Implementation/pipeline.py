#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main RAG Pipeline Module.

This module provides the central RAG (Retrieval-Augmented Generation) pipeline that
orchestrates all components for processing user queries about SAP ABAP documentation.
The pipeline combines retrieval, generation, and query enhancement with sophisticated
conversation context support, confidence level assessment, and specialized handling
for different query types.

Key Features:
    - Unified RAG pipeline coordinating retrieval, generation, and enhancement
    - Query enhancement with rewriting and decomposition capabilities
    - Conversation context support for multi-turn interactions
    - Confidence level assessment for response reliability
    - Specialized handling for interface, method, and structure queries
    - Multi-collection support for different document types
    - Comprehensive error handling and fallback mechanisms
    - Method implementation extraction and interface analysis
    - Sub-query processing for complex questions

Architecture:
    The pipeline follows a modular architecture where each component has specific
    responsibilities:
    - RAGRetriever: Document retrieval with hybrid search capabilities
    - RAGGenerator: Response generation using LLM models
    - QueryEnhancer: Query improvement through rewriting and decomposition
    - RAGPipeline: Central coordinator managing the complete workflow

Classes:
    RAGPipeline: Main pipeline class orchestrating all RAG operations

Dependencies:
    - retriever: Document retrieval component
    - generator: Response generation component
    - query_enhancer: Query enhancement component
    - config: Configuration settings and system prompts

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

# Third-party imports
import nest_asyncio

# Apply nest_asyncio to allow asyncio to work with Streamlit
nest_asyncio.apply()

# Local imports
from config import (
    COLLECTIONS,
    DEFAULT_COLLECTION_TYPE,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TOP_K,
    ENHANCED_METHOD_SYSTEM_PROMPT,
    INTERFACE_SYSTEM_PROMPT,
    LOG_FILE,
    LOG_LEVEL,
    STRUCTURE_SYSTEM_PROMPT,
    USE_HYBRID_DEFAULT,
    USE_RERANKER_DEFAULT,
)
from generator import RAGGenerator
from query_enhancer import QueryEnhancer
from retriever import RAGRetriever

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Main RAG pipeline that combines retrieval and generation components with conversation support and query enhancement.
    
    This class serves as the central coordinator for all RAG operations, orchestrating
    the interaction between retrieval, generation, and query enhancement components.
    It provides sophisticated query processing with support for different query types,
    conversation context, and specialized handling for SAP ABAP documentation.
    
    The pipeline implements several advanced features:
    - Query enhancement through rewriting and decomposition
    - Specialized prompts for different query types (interface, method, structure)
    - Method implementation extraction for interface queries
    - Conversation context management for multi-turn interactions
    - Confidence level assessment for response reliability
    - Comprehensive error handling with graceful degradation
    
    Attributes:
        retriever (RAGRetriever): Document retrieval component
        generator (RAGGenerator): Response generation component
        query_enhancer (QueryEnhancer): Query enhancement component
        current_collection_type (str): Currently active collection type
        
    Examples:
        >>> pipeline = RAGPipeline()
        >>> result = await pipeline.process_query(
        ...     "How do ABAP classes implement interfaces?",
        ...     use_hybrid=True,
        ...     top_k=5
        ... )
        >>> print(f"Response: {result['response']}")
        >>> print(f"Confidence: {result['confidence_level']}")
    """
    
    def __init__(self) -> None:
        """
        Initialize the RAG pipeline with its components.
        
        Sets up all required components for the RAG pipeline including retriever,
        generator, and query enhancer. Also establishes the default collection type
        for document retrieval.
        
        Note:
            The initialization process creates instances of all required components
            and configures them for optimal performance. The query enhancer receives
            the generator instance to enable LLM-powered query transformation.
        """
        logger.info("Initializing RAG Pipeline")
        self.retriever = RAGRetriever()
        self.generator = RAGGenerator()
        self.query_enhancer = QueryEnhancer(self.generator)  # Initialize the query enhancer
        self.current_collection_type = DEFAULT_COLLECTION_TYPE
        logger.info(f"RAG Pipeline initialized with default collection type: {self.current_collection_type}")
        
    def _extract_confidence_level(self, response_text: str) -> Tuple[str, str]:
        """
        Extract confidence level from the response text.
        
        This method parses the generated response to extract confidence level
        indicators and provides a cleaned version of the response without
        the confidence declaration.
        
        Args:
            response_text: Generated response from Claude containing potential
                          confidence level indicators
                          
        Returns:
            Tuple containing:
            - confidence_level: Extracted confidence (HIGH, MEDIUM, LOW, or UNKNOWN)
            - cleaned_response: Response text with confidence declaration removed
            
        Examples:
            >>> pipeline = RAGPipeline()
            >>> confidence, clean_text = pipeline._extract_confidence_level(
            ...     "Confidence: HIGH\\n\\nThis is the response content."
            ... )
            >>> print(confidence)  # "HIGH"
            >>> print(clean_text)  # "This is the response content."
            
        Note:
            The method looks for various confidence declaration patterns and removes
            accuracy metrics as specified in requirements. It preserves other
            evaluation metrics for display in the UI.
        """
        confidence_level = "UNKNOWN"
        cleaned_response = response_text
        
        # Look for confidence level at the beginning of the response
        confidence_patterns = [
            r"^Confidence:\s*(HIGH|MEDIUM|LOW)\s*\n",
            r"^confidence:\s*(HIGH|MEDIUM|LOW)\s*\n",
            r"^Confidence level:\s*(HIGH|MEDIUM|LOW)\s*\n",
            r"^confidence level:\s*(HIGH|MEDIUM|LOW)\s*\n"
        ]
        
        for pattern in confidence_patterns:
            match = re.search(pattern, response_text, re.IGNORECASE)
            if match:
                confidence_level = match.group(1).upper()
                # Remove the confidence declaration from the response
                cleaned_response = re.sub(pattern, "", response_text, 1, re.IGNORECASE)
                break
                
        # Only remove the Accuracy metric as requested
        accuracy_pattern = r'Accuracy:\s*\d+%\s*\([^)]+\)'
        cleaned_response = re.sub(accuracy_pattern, "", cleaned_response)
        
        # Clean up any extra whitespace or newlines from the removal
        cleaned_response = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_response)
        cleaned_response = cleaned_response.strip()
        
        return confidence_level, cleaned_response
        
    def _extract_method_implementations(self, contexts: List[Dict], target_interface: Optional[str] = None) -> List[Dict]:
        """
        Enhanced method to extract ALL method implementations from contexts with special handling for interface methods.
        
        This method performs comprehensive analysis of retrieved contexts to identify
        and extract method implementations, particularly focusing on interface methods
        with the standard SAP ABAP interface~method naming pattern.
        
        Args:
            contexts: List of context documents from retrieval results
            target_interface: Optional target interface name to filter methods
                             (e.g., "/LTB/IF_JOB_DISPATCHER" or "IF_EXAMPLE")
                
        Returns:
            List of detected method implementations, each containing:
            - method_name: Full method name (interface~method format)
            - context: Relevant text context explaining the method
            - doc_id: Source document identifier
            - doc_title: Source document title
            - doc_score: Relevance score from retrieval
            
        Examples:
            >>> pipeline = RAGPipeline()
            >>> contexts = [{"text": "## IF_EXAMPLE~METHOD1\\nThis method...", ...}]
            >>> methods = pipeline._extract_method_implementations(
            ...     contexts, target_interface="IF_EXAMPLE"
            ... )
            >>> for method in methods:
            ...     print(f"Method: {method['method_name']}")
            
        Note:
            The method uses multiple regex patterns to capture various documentation
            formats and provides comprehensive context extraction around each method.
            It handles both exact interface matching and general method detection.
        """
        method_implementations = []
        known_methods = set()  # Track unique methods to avoid duplicates
        
        # First pass: scan all documents to find any method implementations
        # This helps build a comprehensive list of methods even if patterns are missed
        all_possible_methods = set()
        
        # Common interface method patterns with more variations
        patterns = [
            r'(\b\S+~\S+\b)',                         # Interface method with tilde (general)
            r'method\s+(\w+)',                        # Method keyword followed by name
            r'METHOD\s+(\w+)',                        # Uppercase METHOD keyword
            r'(\b\S+)~(\w+)',                         # Interface~method pattern 
            r'(?:implements|implements method)\s+(\S+)', # Implements keyword
            r'(?:#{1,4}\s*|\*\s+)(\S+~\S+)',         # Header or list item with tilde pattern
            r'(?:public|protected|private)\s+method\s+(\w+)', # Method with visibility
            r'([\w/]+)~(\w+)',                        # Namespace~method pattern
        ]
        
        if target_interface:
            # Escape any special regex characters in the interface name
            escaped_interface = re.escape(target_interface.lower())
            
            # Special patterns specifically for this interface
            interface_patterns = [
                fr'({escaped_interface}~\w+)',               # Exact interface~method
                fr'{escaped_interface}\s*~\s*(\w+)',         # Interface ~ method (with spaces)
                fr'method\s+{escaped_interface}~(\w+)',      # METHOD interface~method
                fr'implements\s+{escaped_interface}~(\w+)',  # IMPLEMENTS interface~method
                fr'(?:#{1,4}\s*){escaped_interface}~(\w+)',  # Header with interface~method
                fr'METHOD\s+{escaped_interface}~(\w+)',      # METHOD interface~method (uppercase)
                fr'call\s+method\s+{escaped_interface}~(\w+)', # CALL METHOD pattern
                fr'call\s+function\s+{escaped_interface}~(\w+)', # CALL FUNCTION pattern
            ]
            patterns = interface_patterns + patterns  # Prioritize interface-specific patterns
        
        # First scan: comprehensive search for all possible method names
        for ctx in contexts:
            text = ctx.get('text', '').lower()  # Convert to lowercase for case-insensitive matching
            
            # Special case: Check for interface name in text first
            if target_interface and target_interface.lower() in text:
                # Aggressive search for all methods that could belong to this interface
                # Look for the exact pattern interface~method or header ## interface~method
                for line in text.split('\n'):
                    if target_interface.lower() in line and '~' in line:
                        # This line potentially contains a method reference
                        for pattern in patterns:
                            matches = re.finditer(pattern, line, re.IGNORECASE)
                            for match in matches:
                                method_name = match.group(1) if len(match.groups()) >= 1 else None
                                
                                if method_name:
                                    # If it's just a method part without interface, combine them
                                    if target_interface and '~' not in method_name:
                                        method_name = f"{target_interface}~{method_name}"
                                    
                                    # Add to possible methods
                                    all_possible_methods.add(method_name.lower())
        
        # Additional special pattern matching for section headers
        for ctx in contexts:
            text = ctx.get('text', '')
            
            # Look for headers that might contain method names
            header_pattern = r'#{1,4}\s+([^#\n]+)'
            header_matches = re.finditer(header_pattern, text)
            for match in header_matches:
                header_text = match.group(1).strip()
                if '~' in header_text:
                    # This looks like a method header
                    method_name = header_text
                    
                    # If target interface is specified, only include matching methods
                    if target_interface and not method_name.lower().startswith(target_interface.lower()):
                        continue
                    
                    # Add to possible methods
                    all_possible_methods.add(method_name.lower())
        
        # Second pass: Extract comprehensive content for each identified method
        for method_name_lower in all_possible_methods:
            # Re-use existing implementation details if we've already found this method
            if method_name_lower in known_methods:
                continue
            
            known_methods.add(method_name_lower)
            
            # Get proper case for method name for display
            method_name = None
            for ctx in contexts:
                text = ctx.get('text', '')
                # Try to find this method with proper case
                matches = re.finditer(re.escape(method_name_lower), text.lower())
                for match in matches:
                    start_pos = match.start()
                    end_pos = match.end()
                    # Extract the original case version
                    method_name = text[start_pos:end_pos]
                    break
                if method_name:
                    break
            
            # If we couldn't find proper case, use the lowercase version
            if not method_name:
                method_name = method_name_lower
            
            # Now extract comprehensive info for this method
            best_doc_id = None
            best_doc_title = None
            best_doc_score = 0
            best_context = None
            
            for idx, ctx in enumerate(contexts):
                doc_id = ctx.get('id', idx + 1)
                text = ctx.get('text', '')
                title = ctx.get('title', 'Untitled')
                score = abs(ctx.get('score', 0))  # Use absolute score for consistency
                
                if method_name_lower not in text.lower():
                    continue
                    
                # Find the best section explaining this method
                text_lower = text.lower()
                method_pos = text_lower.find(method_name_lower)
                
                if method_pos >= 0:
                    # Extract a generous context window around the method
                    start_pos = max(0, text.rfind('\n', 0, method_pos - 200) if method_pos > 200 else 0)
                    
                    # Find the end of this section - look for next header or 500 chars after
                    next_header_pos = text.find('##', method_pos + len(method_name))
                    if next_header_pos < 0 or next_header_pos > method_pos + 1000:
                        end_pos = min(len(text), method_pos + 500)
                    else:
                        end_pos = next_header_pos
                    
                    context_text = text[start_pos:end_pos].strip()
                    
                    # Use this context if it's better (higher score or first occurrence)
                    if best_context is None or score > best_doc_score:
                        best_doc_id = doc_id
                        best_doc_title = title
                        best_doc_score = score
                        best_context = context_text
            
            # Only add if we found some context
            if best_context:
                method_implementations.append({
                    'method_name': method_name,
                    'context': best_context,
                    'doc_id': best_doc_id,
                    'doc_title': best_doc_title,
                    'doc_score': best_doc_score
                })
        
        # Special case for interface methods: if we found any methods for this interface,
        # make sure we have the complete set even if detailed context is missing
        if target_interface and method_implementations:
            # Standard methods for common interfaces - add if not already found
            common_methods = {
                '/ltb/if_job_dispatcher': ['job_open', 'job_close', 'submit'],
                # Add other common interfaces as needed
            }
            
            if target_interface.lower() in common_methods:
                # Check for missing standard methods
                found_method_names = [m['method_name'].lower() for m in method_implementations]
                
                for method_suffix in common_methods[target_interface.lower()]:
                    expected_method = f"{target_interface}~{method_suffix}".lower()
                    
                    if not any(expected_method in method_name for method_name in found_method_names):
                        # Try to find this method in the context
                        for idx, ctx in enumerate(contexts):
                            doc_id = ctx.get('id', idx + 1)
                            text = ctx.get('text', '')
                            title = ctx.get('title', 'Untitled')
                            score = abs(ctx.get('score', 0))
                            
                            # Look for any mention of this method
                            if method_suffix.lower() in text.lower() and target_interface.lower() in text.lower():
                                # Add with minimal context
                                method_implementations.append({
                                    'method_name': f"{target_interface}~{method_suffix}",
                                    'context': f"Method {target_interface}~{method_suffix} mentioned in document.",
                                    'doc_id': doc_id,
                                    'doc_title': title,
                                    'doc_score': score
                                })
                                break
        
        return method_implementations

    def _format_context_document_references(self, context_results: List[Tuple[Any, float]]) -> str:
        """
        Format document references for consistent numbering between LLM and UI.
        
        This method creates properly formatted context strings with consistent
        document numbering that matches what users see in the UI, ensuring
        that document references in generated responses align with the UI display.
        
        Args:
            context_results: Results from retriever containing (result, score) tuples
                           where result contains payload with document information
                           
        Returns:
            Formatted context string with consistent document numbering and
            comprehensive information including titles, sources, scores, and content
            
        Note:
            Uses 1-based document numbering to match UI display and converts
            scores to absolute values for consistent representation across
            different retrieval methods.
        """
        context_parts = []
        
        # Process each document with explicit numbering
        for i, (result, score) in enumerate(context_results):
            # Use 1-based document numbering to match UI
            doc_number = i + 1
            
            # Extract fields from payload
            payload = result.payload
            text = payload.get("text", "")
            title = payload.get("title", "")
            code_snippet = payload.get("code_snippet", "")
            filename = payload.get("filename", "")
            
            # Important: Use the absolute score value to avoid negative scores in references
            # This matches what users see in the UI
            abs_score = abs(score)
            
            # Format context with explicit document number
            context_part = f"[Document {doc_number}: {title} (Score: {abs_score:.4f})]\n"
            if filename:
                context_part += f"Source: {filename}\n"
            if code_snippet:
                context_part += f"Code: ```abap\n{code_snippet}\n```\n"
            if text:
                context_part += f"Content:\n{text}\n"
            
            context_part += f"\n"
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def set_collection_type(self, collection_type: str) -> str:
        """
        Set the collection type to use for retrievals.
        
        Changes the active collection for all subsequent retrieval operations,
        updating both the pipeline's collection type and the retriever's
        active collection.
        
        Args:
            collection_type: Type of collection to use ('classes' or 'reports').
                           Must be a valid key in the COLLECTIONS configuration.
                           
        Returns:
            The actual collection name that will be used for retrievals
            
        Examples:
            >>> pipeline = RAGPipeline()
            >>> collection_name = pipeline.set_collection_type("classes")
            >>> print(f"Now using collection: {collection_name}")
            
        Note:
            If an invalid collection type is provided, the method falls back
            to the default collection type and logs a warning.
        """
        if collection_type in COLLECTIONS:
            self.current_collection_type = collection_type
            collection_name = self.retriever.set_collection(collection_type)
            logger.info(f"Pipeline collection type set to: {collection_type} (collection: {collection_name})")
            return collection_name
        else:
            logger.warning(f"Unknown collection type: {collection_type}, using default")
            self.current_collection_type = DEFAULT_COLLECTION_TYPE
            collection_name = self.retriever.set_collection(DEFAULT_COLLECTION_TYPE)
            return collection_name
    
    def get_current_collection_info(self) -> Dict[str, str]:
        """
        Get information about the current collection.
        
        Returns comprehensive information about the currently active collection
        including its type, name, and description for display and logging purposes.
        
        Returns:
            Dictionary containing:
            - type: Collection type identifier ('classes' or 'reports')
            - name: Actual collection name in the vector database
            - description: Human-readable description of the collection
            
        Examples:
            >>> pipeline = RAGPipeline()
            >>> info = pipeline.get_current_collection_info()
            >>> print(f"Using {info['type']}: {info['description']}")
        """
        collection_type = self.current_collection_type
        collection_name = COLLECTIONS[collection_type]["name"]
        collection_description = COLLECTIONS[collection_type]["description"]
        
        return {
            "type": collection_type,
            "name": collection_name,
            "description": collection_description
        }
    
    def _is_interface_query(self, query: str) -> bool:
        """
        Determine if a query is specifically asking about interfaces.
        
        Analyzes the query text to identify whether the user is asking
        about interfaces, interface implementations, or interface-related topics.
        
        Args:
            query: User query string to analyze
            
        Returns:
            True if the query appears to be about interfaces, False otherwise
            
        Examples:
            >>> pipeline = RAGPipeline()
            >>> pipeline._is_interface_query("What interfaces does this class implement?")
            True
            >>> pipeline._is_interface_query("How do I create a variable?")
            False
        """
        interface_keywords = [
            "interface", "interfaces", "implement", "implements", 
            "what interface", "which interface", "all interface"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in interface_keywords)
    
    def _is_method_query(self, query: str) -> bool:
        """
        Determine if a query is specifically asking about methods.
        
        Analyzes the query text to identify whether the user is asking
        about methods, method implementations, or method-related functionality.
        
        Args:
            query: User query string to analyze
            
        Returns:
            True if the query appears to be about methods, False otherwise
            
        Examples:
            >>> pipeline = RAGPipeline()
            >>> pipeline._is_method_query("What methods are available in this interface?")
            True
            >>> pipeline._is_method_query("How do I declare a variable?")
            False
        """
        method_keywords = [
            "method", "methods", "implement", "implementing", "implemented",
            "what method", "which method", "all method", "list method",
            "function", "functions", "procedure", "procedures"
        ]
        
        query_lower = query.lower()
        
        # Check for specific method query patterns
        if "method" in query_lower and "interface" in query_lower:
            return True
            
        # Check for any method keywords
        return any(keyword in query_lower for keyword in method_keywords)
    
    def _is_structure_related_query(self, query: str) -> bool:
        """
        Determine if a query is related to code structure.
        
        Analyzes the query to identify whether it's asking about structural
        elements of ABAP code such as classes, interfaces, inheritance, etc.
        
        Args:
            query: User query string to analyze
            
        Returns:
            True if the query is about code structure, False otherwise
            
        Examples:
            >>> pipeline = RAGPipeline()
            >>> pipeline._is_structure_related_query("How do classes inherit from interfaces?")
            True
            >>> pipeline._is_structure_related_query("What is the syntax for loops?")
            False
        """
        structure_keywords = [
            "interface", "interfaces", "class", "classes", "method", "methods",
            "inherit", "inherits", "inheritance", "implement", "implements",
            "extend", "extends", "attribute", "attributes", "property", "properties"
        ]
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in structure_keywords)
    
    def _extract_tables_from_contexts(self, contexts: List[Dict]) -> Tuple[List[Dict], bool]:
        """
        Extract tables from context documents.
        
        Identifies and extracts tabular data from retrieved context documents
        using various table format patterns including Markdown tables, ASCII
        tables, and structured lists.
        
        Args:
            contexts: List of context documents from retrieval results
            
        Returns:
            Tuple containing:
            - extracted_tables: List of table dictionaries with text, doc_id, title, score
            - has_tables: Boolean indicating whether any tables were found
            
        Note:
            Supports multiple table formats including Markdown-style tables with
            pipe separators, ASCII tables with box drawing, and structured lists
            that represent tabular data.
        """
        extracted_tables = []
        has_tables = False
        
        # Regex patterns for different table formats
        table_patterns = [
            # Markdown style tables with | separators
            r'(\|[^\n]+\|\n\|[-:| ]+\|\n(?:\|[^\n]+\|\n)+)',
            # Simple tables with | but without header separator
            r'(\|[^\n]+\|\n(?:\|[^\n]+\|\n)+)',
            # ASCII-style tables with +---+ format
            r'(\+[-+]+\+\n(?:\|[^\n]+\|\n)+\+[-+]+\+)',
            # Lists that might be formatted as tables
            r'((?:^ *- [^\n]+\n){2,})'
        ]
        
        for ctx in contexts:
            text = ctx.get('text', '')
            doc_id = ctx.get('id', '')  # Use document ID as provided in the context
            title = ctx.get('title', 'Untitled')
            score = ctx.get('score', 0)
            
            # Try each pattern to find tables
            for pattern in table_patterns:
                tables = re.finditer(pattern, text, re.MULTILINE)
                for table_match in tables:
                    table_text = table_match.group(1).strip()
                    if table_text and len(table_text.split('\n')) > 1:  # Ensure it's actually a multi-line table
                        extracted_tables.append({
                            "text": table_text,
                            "doc_id": doc_id,
                            "title": title,
                            "score": score
                        })
                        has_tables = True
        
        return extracted_tables, has_tables
    
    def _extract_method_sections(self, contexts: List[Dict]) -> Tuple[List[Dict], bool]:
        """
        Extract method implementation sections from context documents.
        
        Identifies and extracts sections of documentation that specifically
        describe method implementations, particularly focusing on interface
        method patterns with the tilde (~) notation.
        
        Args:
            contexts: List of context documents from retrieval results
            
        Returns:
            Tuple containing:
            - extracted_sections: List of method section dictionaries
            - has_method_sections: Boolean indicating whether any sections were found
            
        Note:
            Looks for various patterns including section headers with method names,
            method implementation descriptions, and bulleted lists of methods.
        """
        extracted_sections = []
        has_method_sections = False
        
        # Patterns to identify method implementations
        method_patterns = [
            # Headers followed by content
            r'(#{1,4}\s*[^\n#]+?~[^\n#]+?(?:\n+(?:(?!#{1,4}\s)[^\n]+\n+)*))',
            # Method name as heading with implementation details
            r'([^\n]+?~[^\n]+?\s*:\s*\n+(?:[^\n]+\n+)*)',
            # Section with method implementation
            r'(- [^\n]+?~[^\n]+?:?\s*\n+(?:\s*- [^\n]+\n+)*)'
        ]
        
        for ctx in contexts:
            text = ctx.get('text', '')
            doc_id = ctx.get('id', '')
            title = ctx.get('title', 'Untitled')
            score = ctx.get('score', 0)
            
            # Try each pattern to find method implementations
            for pattern in method_patterns:
                sections = re.finditer(pattern, text, re.MULTILINE)
                for section_match in sections:
                    section_text = section_match.group(1).strip()
                    # Check if this actually looks like a method section (contains tilde character)
                    if section_text and '~' in section_text:
                        extracted_sections.append({
                            "text": section_text,
                            "doc_id": doc_id,
                            "title": title,
                            "score": score
                        })
                        has_method_sections = True
        
        return extracted_sections, has_method_sections
    
    def _contains_interface_info(self, table_text: str) -> bool:
        """
        Check if a table contains interface information.
        
        Analyzes table text to determine whether it contains information
        about interfaces, interface implementations, or interface-related data.
        
        Args:
            table_text: Extracted table text to analyze
            
        Returns:
            True if the table contains interface information, False otherwise
        """
        interface_keywords = ["interface", "interfaces", "implements", "implement", "if_"]
        table_lower = table_text.lower()
        
        return any(keyword in table_lower for keyword in interface_keywords)
    
    def _contains_method_info(self, section_text: str) -> bool:
        """
        Check if a section contains method implementation information.
        
        Analyzes section text to determine whether it contains information
        about method implementations, particularly interface methods.
        
        Args:
            section_text: Extracted text section to analyze
            
        Returns:
            True if the section contains method implementation info, False otherwise
        """
        method_keywords = ["~", "method", "methods", "implements", "implementation"]
        section_lower = section_text.lower()
        
        # Check for the tilde character which indicates interface~method pattern
        if "~" in section_text:
            return True
            
        return any(keyword in section_lower for keyword in method_keywords)
    
    def _create_tailored_system_prompt(self, query: str, contexts: List[Dict]) -> str:
        """
        Create a tailored system prompt based on query and content.
        
        Analyzes the query and retrieved contexts to create a specialized system
        prompt that optimizes the LLM's response for the specific type of question
        being asked (method, interface, structure, or general queries).
        
        Args:
            query: User query string to analyze
            contexts: Retrieved context documents for additional analysis
            
        Returns:
            Tailored system prompt optimized for the query type and available content
            
        Note:
            The method creates specialized prompts for different query types:
            - Method queries: Focus on specific method implementations and parameters
            - Interface queries: Emphasize interface listings and implementations
            - Structure queries: Highlight code organization and relationships
            - General queries: Use default prompt with formatting guidance
        """
        # Check if this is a method-specific query
        is_method_query = self._is_method_query(query)
        
        if is_method_query:
            logger.info("Query identified as method-specific")
            
            # Extract interface name if present
            interface_name = None
            interface_pattern = r'(/\w+/if_\w+|if_\w+)'
            interface_match = re.search(interface_pattern, query, re.IGNORECASE)
            if interface_match:
                interface_name = interface_match.group(0)
                logger.info(f"Method query references interface: {interface_name}")
                
                # Check what the user is actually asking about
                purpose_patterns = ["purpose", "what does", "what is", "why is", "goal of", "objective of"]
                return_patterns = ["return", "output", "provide", "give back", "result"]
                parameter_patterns = ["parameter", "argument", "input", "take", "accept"]
                
                query_lower = query.lower()
                asks_about_purpose = any(pattern in query_lower for pattern in purpose_patterns)
                asks_about_return = any(pattern in query_lower for pattern in return_patterns)
                asks_about_parameters = any(pattern in query_lower for pattern in parameter_patterns)
                
                # Build a focused prompt based on what the user is asking
                method_appendix = f"""
    SPECIFIC QUERY ANALYSIS:
    - User is asking about method: {interface_name}~{query.split('~')[1].split()[0] if '~' in query else 'method'}
    - Focus areas detected:
    {'- Purpose/functionality of the method' if asks_about_purpose else ''}
    {'- Return values/output of the method' if asks_about_return else ''}
    {'- Parameters/inputs of the method' if asks_about_parameters else ''}

    RESPONSE REQUIREMENTS:
    1. Answer ONLY what was asked - {'purpose' if asks_about_purpose else ''} {'return values' if asks_about_return else ''} {'parameters' if asks_about_parameters else ''}
    2. Begin with a direct, simple sentence answering the question
    3. Use a conversational tone as if continuing a dialogue
    4. Don't use headings, section titles or formal structure
    5. After answering the specific question(s), stop - do not add additional information

    CRITICAL: Focus exclusively on information from documents that contain the method {interface_name}~{query.split('~')[1].split()[0] if '~' in query else 'method'}.
    """
                # Add specifically tailored instructions for this interface
                system_prompt = ENHANCED_METHOD_SYSTEM_PROMPT + method_appendix
                
                # Extract method sections from contexts
                method_sections, has_method_sections = self._extract_method_sections(contexts)
                
                # If we found method sections, add guidance
                if has_method_sections:
                    method_appendix = f"""
    Found {len(method_sections)} sections containing relevant method information.
    Focus only on these sections to answer the specific question asked.
    """
                    system_prompt += method_appendix
                
                return system_prompt
        
        # Check if this is an interface-specific query
        is_interface_query = self._is_interface_query(query)
        if is_interface_query:
            logger.info("Query identified as interface-specific")
            system_prompt = INTERFACE_SYSTEM_PROMPT
            
            # Extract tables from contexts to emphasize in the prompt
            tables, has_tables = self._extract_tables_from_contexts(contexts)
            
            # Filter for tables that contain interface information
            interface_tables = []
            for table in tables:
                if self._contains_interface_info(table["text"]):
                    interface_tables.append(table)
            
            # If we found interface tables, add specific guidance
            if interface_tables:
                table_appendix = f"""
    FORMATTING INSTRUCTIONS:
    1. Begin with a direct answer in conversational style
    2. Don't use section headings or formal report style
    3. Present information in a natural dialogue format
    4. End with a concise summary if needed
    """
                system_prompt += table_appendix
            
            return system_prompt
        
        # If not interface-specific or method-specific, check if it's structure-related
        is_structure_query = self._is_structure_related_query(query)
        if is_structure_query:
            logger.info("Query identified as structure-related")
            system_prompt = STRUCTURE_SYSTEM_PROMPT
            
            # Extract tables from contexts
            tables, has_tables = self._extract_tables_from_contexts(contexts)
            
            # If we found tables, add guidance
            if has_tables:
                table_appendix = f"""
    FORMATTING INSTRUCTIONS:
    1. Begin with a direct answer in conversational style
    2. Don't use section headings or formal report style
    3. Present information in a natural dialogue format
    4. End with a concise summary if needed
    """
                system_prompt += table_appendix
            
            return system_prompt
        
        # Default prompt for other queries with improved format guidance
        return DEFAULT_SYSTEM_PROMPT + """
    FORMATTING REQUIREMENTS:
    1. Begin with a direct answer to the question in a conversational tone
    2. Do not use section headings or formal report structure
    3. Maintain a natural dialogue flow throughout your response
    4. Focus exclusively on information directly addressing the question
    5. Present information as if continuing a conversation, not as a report
    """
    
    async def _process_sub_queries(self, 
                                  sub_queries: List[str], 
                                  conversation_history: Optional[List[Dict[str, str]]] = None,
                                  use_hybrid: bool = USE_HYBRID_DEFAULT, 
                                  use_reranker: bool = USE_RERANKER_DEFAULT, 
                                  top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        """
        Process a list of sub-queries and combine the results.
        
        This method handles the processing of decomposed queries by executing
        each sub-query independently and then combining the results into a
        comprehensive response that addresses the original complex query.
        
        Args:
            sub_queries: List of decomposed sub-queries to process
            conversation_history: Optional list of previous conversation messages
            use_hybrid: Whether to use hybrid retrieval for sub-queries
            use_reranker: Whether to apply reranking to sub-query results
            top_k: Number of top contexts to use for each sub-query
            
        Returns:
            Dictionary containing combined response and comprehensive metadata including:
            - response: Combined response addressing all sub-queries
            - full_response: Response with confidence level prefix
            - confidence_level: Overall confidence based on sub-query confidences
            - contexts: Combined contexts from all sub-queries
            - sub_query_results: Individual results for each sub-query
            - is_decomposed: True to indicate this is a decomposed query result
            
        Note:
            Uses a reduced top_k per sub-query to avoid context overload while
            ensuring comprehensive coverage. Combines results using LLM to
            maintain coherence and avoid redundancy.
        """
        logger.info(f"Processing {len(sub_queries)} sub-queries")
        
        combined_contexts = []
        all_detected_methods = []
        all_sub_responses = []
        
        for i, sub_query in enumerate(sub_queries):
            logger.info(f"Processing sub-query {i+1}/{len(sub_queries)}: '{sub_query}'")
            
            # Process each sub-query with reduced top_k to avoid context overload
            sub_top_k = max(2, top_k // len(sub_queries))
            
            # Retrieve relevant documents for this sub-query
            results = self.retriever.retrieve(
                query=sub_query, 
                limit=max(10, sub_top_k * 3),
                use_hybrid=use_hybrid
            )
            
            # Rerank results if requested
            if use_reranker and self.retriever.has_reranker:
                context_results = self.retriever.rerank_results(
                    query=sub_query, 
                    results=results, 
                    top_k=sub_top_k
                )
            else:
                # Just take the top_k from the original results
                context_results = [(result, result.score) for result in results[:sub_top_k]]
                
            # Check if results are relevant
            has_relevant_results = any(score > 0 for _, score in context_results)
            
            if has_relevant_results:
                # Create contexts list for this sub-query
                sub_contexts = []
                for j, (result, score) in enumerate(context_results):
                    # Use the actual document position as the ID to match what user sees in UI
                    # For sub-queries, we'll add a prefix to identify which sub-query it came from
                    context = {
                        "id": f"{i+1}.{j+1}",  # Format: "sub_query_num.doc_num"
                        "title": result.payload.get("title", ""),
                        "filename": result.payload.get("filename", ""),
                        "text": result.payload.get("text", ""),
                        "code_snippet": result.payload.get("code_snippet", ""),
                        "score": score,
                        "sub_query": sub_query  # Track which sub-query this context belongs to
                    }
                    sub_contexts.append(context)
                    
                    # Also add to the combined contexts list with unified numbering
                    combined_context = context.copy()
                    combined_context["id"] = len(combined_contexts) + 1  # Sequential numbering
                    combined_contexts.append(combined_context)
                
                # Extract method implementations if needed
                if self._is_method_query(sub_query):
                    # Try to extract interface name if present
                    interface_name = None
                    interface_pattern = r'(/\w+/if_\w+|if_\w+)'
                    interface_match = re.search(interface_pattern, sub_query, re.IGNORECASE)
                    if interface_match:
                        interface_name = interface_match.group(0)
                        detected_methods = self._extract_method_implementations(sub_contexts, target_interface=interface_name)
                        all_detected_methods.extend(detected_methods)
                
                # Create a tailored system prompt for this sub-query
                tailored_system_prompt = self._create_tailored_system_prompt(sub_query, sub_contexts)
                
                # Format the context with consistent document numbering
                formatted_context = self._format_context_document_references(context_results)
                
                # Generate response for this sub-query
                sub_response = None
                if conversation_history and len(conversation_history) > 0:
                    # Include conversation history when generating response
                    sub_response = await self.generator.generate_with_history(
                        sub_query, formatted_context, conversation_history, system_prompt=tailored_system_prompt
                    )
                else:
                    # Standard response without conversation history
                    sub_response = await self.generator.generate(
                        sub_query, formatted_context, system_prompt=tailored_system_prompt
                    )
                
                # Extract confidence level from response (if present)
                confidence_level, cleaned_response = self._extract_confidence_level(sub_response)
                
                # Add the sub-response
                all_sub_responses.append({
                    "sub_query": sub_query,
                    "response": cleaned_response,
                    "confidence_level": confidence_level,
                    "contexts": sub_contexts
                })
            else:
                # If no relevant results for this sub-query, generate an "not enough information" response
                logger.warning(f"No relevant results found for sub-query: '{sub_query}'")
                all_sub_responses.append({
                    "sub_query": sub_query,
                    "response": f"I couldn't find enough information to answer the question: '{sub_query}'.",
                    "confidence_level": "LOW",
                    "contexts": []
                })
        
        # Combine sub-responses into a final response
        # For combining, we'll generate a new prompt asking Claude to combine the answers
        combine_system_prompt = """You are an expert assistant for SAP ABAP code documentation.
        You need to combine multiple partial answers into a cohesive, comprehensive response.
        Maintain a logical flow and avoid redundancy.
        Start with a brief overview that answers the main query directly.
        Then provide details from each sub-answer, organizing information logically.
        Include all relevant details from the sub-answers, but avoid repeating the same information.
        End with a concise summary that ties everything together.
        """
        
        combine_content = f"Original query: {sub_queries[0]}\n\n"  # Use first sub-query as representative
        combine_content += "I need to combine the following sub-answers into a comprehensive response:\n\n"
        
        for i, sub_resp in enumerate(all_sub_responses):
            combine_content += f"Sub-query {i+1}: {sub_resp['sub_query']}\n"
            combine_content += f"Answer {i+1}: {sub_resp['response']}\n\n"
        
        combine_messages = [
            {"role": "user", "content": combine_content}
        ]
        
        # Generate combined response
        combined_response = await self.generator._generate_with_client(combine_messages, combine_system_prompt)
        
        # Determine overall confidence level based on sub-response confidence levels
        confidence_levels = [resp["confidence_level"] for resp in all_sub_responses]
        overall_confidence = "LOW"
        
        if all(level == "HIGH" for level in confidence_levels):
            overall_confidence = "HIGH"
        elif any(level == "HIGH" for level in confidence_levels) and not any(level == "LOW" for level in confidence_levels):
            overall_confidence = "MEDIUM"
        elif all(level == "MEDIUM" for level in confidence_levels):
            overall_confidence = "MEDIUM"
        
        # Add confidence level prefix to the combined response
        final_response = f"Confidence: {overall_confidence}\n\n{combined_response}"
        
        # Create the final result dictionary
        result_dict = {
            "response": combined_response,
            "full_response": final_response,
            "confidence_level": overall_confidence,
            "context_count": len(combined_contexts),
            "contexts": combined_contexts,
            "detected_methods": all_detected_methods,
            "is_follow_up": conversation_history is not None and len(conversation_history) > 0,
            "has_relevant_results": len(combined_contexts) > 0,
            "sub_query_results": all_sub_responses,
            "is_decomposed": True
        }
        
        return result_dict
    
    async def process_query(self, 
                            query: str, 
                            conversation_history: Optional[List[Dict[str, str]]] = None,
                            use_hybrid: bool = USE_HYBRID_DEFAULT, 
                            use_reranker: bool = USE_RERANKER_DEFAULT, 
                            top_k: int = DEFAULT_TOP_K,
                            collection_type: Optional[str] = None,
                            enable_query_enhancement: bool = True) -> Dict[str, Any]:
        """
        Process a user query through the complete RAG pipeline.
        
        This is the main entry point for the RAG pipeline that orchestrates the
        complete query processing workflow including query enhancement, retrieval,
        generation, and specialized handling for different query types.
        
        Args:
            query: User's question or information request
            conversation_history: Optional list of previous conversation messages
                                 for context-aware responses
            use_hybrid: Whether to use hybrid retrieval (dense + sparse vectors)
            use_reranker: Whether to apply cross-encoder reranking to results
            top_k: Number of top context documents to use for generation
            collection_type: Optional collection type to use for this specific query
            enable_query_enhancement: Whether to apply query rewriting and decomposition
            
        Returns:
            Comprehensive dictionary containing:
            - response: Generated response text
            - full_response: Response with confidence prefix
            - confidence_level: Assessed confidence (HIGH, MEDIUM, LOW)
            - contexts: Retrieved document contexts with metadata
            - collection_info: Information about the collection used
            - query enhancement metadata (if enabled)
            - method/interface analysis results (if applicable)
            - conversation metadata
            
        Raises:
            Exception: Re-raises any exceptions from component failures with
                      error information included in the result dictionary
                      
        Examples:
            >>> pipeline = RAGPipeline()
            >>> result = await pipeline.process_query(
            ...     "How do ABAP classes implement interfaces?",
            ...     use_hybrid=True,
            ...     top_k=5
            ... )
            >>> print(f"Response: {result['response']}")
            >>> print(f"Confidence: {result['confidence_level']}")
            >>> for ctx in result['contexts']:
            ...     print(f"Document: {ctx['title']}")
            
        Note:
            The method implements sophisticated query analysis to determine the
            appropriate retrieval strategy, system prompts, and response formatting
            based on the query type (method, interface, structure, or general).
        """
        # Set collection for this query if specified
        if collection_type:
            self.set_collection_type(collection_type)
            
        collection_info = self.get_current_collection_info()
        logger.info(f"Processing query: '{query}' using collection {collection_info['name']} ({collection_info['type']})")
        logger.info(f"Parameters: use_hybrid={use_hybrid}, use_reranker={use_reranker}, top_k={top_k}")
        logger.info(f"Conversation history provided: {conversation_history is not None}")
        logger.info(f"Query enhancement enabled: {enable_query_enhancement}")
        
        # Initialize result dictionary with basic information
        result_dict = {
            "query": query,
            "collection_info": collection_info
        }
        
        # Check if this is a follow-up question
        is_follow_up = False
        if conversation_history and len(conversation_history) > 0:
            is_follow_up = True
            logger.info("Processing as a follow-up question")
        
        try:
            # Use the enable_query_enhancement parameter instead of hardcoded value
            
            # Step 1: Apply query enhancement (rewriting and decomposition) if enabled
            if enable_query_enhancement:
                enhanced_query_info = await self.query_enhancer.enhance_query(query)
                enhanced_query = enhanced_query_info["enhanced_query"]
                is_enhanced = enhanced_query_info["is_enhanced"]
                is_out_of_context = enhanced_query_info["is_out_of_context"]
                is_decomposed = enhanced_query_info.get("is_decomposed", False)
                sub_queries = enhanced_query_info.get("sub_queries", [])
                
                # Add query enhancement info to result dict
                result_dict.update({
                    "original_query": query,
                    "enhanced_query": enhanced_query,
                    "is_enhanced": is_enhanced,
                    "is_out_of_context": is_out_of_context,
                    "is_decomposed": is_decomposed,
                    "sub_queries": sub_queries
                })
                
                # Log query enhancement results
                if is_enhanced:
                    logger.info(f"Enhanced query: '{enhanced_query}'")
                
                if is_out_of_context:
                    logger.info(f"Query identified as out of context: '{query}'")
                    # Generate an out-of-scope response
                    if is_follow_up:
                        response = await self.generator.generate_out_of_scope_with_history(query, conversation_history)
                    else:
                        response = await self.generator.generate_out_of_scope(query)
                    
                    # Extract confidence level from response (if present)
                    confidence_level, cleaned_response = self._extract_confidence_level(response)
                    
                    # Update result dict with response and return
                    result_dict.update({
                        "response": cleaned_response,
                        "full_response": response,
                        "confidence_level": confidence_level,
                        "context_count": 0,
                        "contexts": [],
                        "is_follow_up": is_follow_up,
                        "has_relevant_results": False
                    })
                    
                    return result_dict
                
                # If query was decomposed, process sub-queries
                if is_decomposed and sub_queries:
                    logger.info(f"Processing decomposed query with {len(sub_queries)} sub-queries: {sub_queries}")
                    
                    # Process all sub-queries and combine results
                    sub_query_results = await self._process_sub_queries(
                        sub_queries=sub_queries,
                        conversation_history=conversation_history,
                        use_hybrid=use_hybrid,
                        use_reranker=use_reranker,
                        top_k=top_k
                    )
                    
                    # Update result dict with combined results
                    result_dict.update(sub_query_results)
                    return result_dict
                
                # Continue with normal query processing using the enhanced query
                processing_query = enhanced_query
            else:
                # If query enhancement is disabled, use the original query
                logger.info("Query enhancement is disabled, using original query")
                processing_query = query
            
            # Check if this is a method-specific query
            is_method_query = self._is_method_query(processing_query)
            interface_reference = None
            if is_method_query:
                # Try to extract interface name if present
                interface_pattern = r'/\w+/if_\w+|if_\w+'
                interface_match = re.search(interface_pattern, processing_query, re.IGNORECASE)
                if interface_match:
                    interface_reference = interface_match.group(0)
                    # Ensure the interface name is preserved with exact casing and formatting
                    logger.info(f"Method query references interface: {interface_reference}")
            
            # Check query type to determine retrieval limits
            is_interface_query = self._is_interface_query(processing_query)
            is_class_query = False
            class_reference = None
            
            # Try to extract class name if present
            class_pattern = r'/\w+/cl_\w+|cl_\w+'
            class_match = re.search(class_pattern, processing_query, re.IGNORECASE)
            if class_match:
                class_reference = class_match.group(0)
                is_class_query = True
                logger.info(f"Query references class: {class_reference}")
            
            # Update result dict with query type information
            result_dict.update({
                "is_method_query": is_method_query,
                "is_interface_query": is_interface_query,
                "is_class_query": is_class_query,
                "interface_reference": interface_reference,
                "class_reference": class_reference
            })
            
            # Set retrieval limits based on query type
            if is_method_query:
                logger.info("Query identified as method-related")
                # For method queries, retrieve many documents to find all implementations
                retrieval_limit = max(30, top_k * 3)
                logger.info(f"Increased retrieval limit to {retrieval_limit} for method query")
            elif is_interface_query:
                logger.info("Query identified as interface-related")
                # For interface queries, retrieve more documents
                retrieval_limit = max(20, top_k * 2)
                logger.info(f"Increased retrieval limit to {retrieval_limit} for interface query")
            elif self._is_structure_related_query(processing_query):
                logger.info("Query identified as structure-related (classes, methods, etc.)")
                # For structure queries, retrieve more documents
                retrieval_limit = max(15, int(top_k * 1.5))
                logger.info(f"Increased retrieval limit to {retrieval_limit} for structure query")
            else:
                retrieval_limit = max(10, top_k * 2)
            
            # Step 2: Retrieve relevant documents
            results = self.retriever.retrieve(
                query=processing_query, 
                limit=retrieval_limit,
                use_hybrid=use_hybrid
            )
            
            # Step 3: Optionally rerank the results
            if use_reranker and self.retriever.has_reranker:
                context_results = self.retriever.rerank_results(
                    query=processing_query, 
                    results=results, 
                    top_k=top_k
                )
            else:
                # Just take the top_k from the original results
                context_results = [(result, result.score) for result in results[:top_k]]
            
            # Step 4: Check if results are relevant (have positive scores)
            has_relevant_results = any(score > 0 for _, score in context_results)
            
            if has_relevant_results:
                # Step 5: Create appropriate contexts list for later use with consistent numbering
                contexts = []
                for i, (result, score) in enumerate(context_results):
                    # Use the actual document position (i+1) as the ID to match what user sees in UI
                    context = {
                        "id": i+1,  # 1-based indexing to match UI display
                        "title": result.payload.get("title", ""),
                        "filename": result.payload.get("filename", ""),
                        "text": result.payload.get("text", ""),
                        "code_snippet": result.payload.get("code_snippet", ""),
                        "score": score
                    }
                    contexts.append(context)
                
                # For method queries, extract method implementations with target interface filter
                detected_methods = []
                if is_method_query and interface_reference:
                    # Pass the exact interface name to filter methods specifically for this interface
                    detected_methods = self._extract_method_implementations(contexts, target_interface=interface_reference)
                    logger.info(f"Detected {len(detected_methods)} method implementations for interface {interface_reference}")
                
                # Create a tailored system prompt based on the query and retrieved contexts
                tailored_system_prompt = self._create_tailored_system_prompt(processing_query, contexts)
                
                # If we have specific interface_reference, enhance the prompt to be very explicit
                if interface_reference:
                    tailored_system_prompt += f"\n\nIMPORTANT: This query is specifically about the interface '{interface_reference}'. Only return methods that exactly match this interface name with the pattern '{interface_reference}~method_name'. DO NOT return methods from any other interface, even if they are similar. Verify each method name carefully before including it."
                
                # Format the context with consistent document numbering
                # This ensures document references in LLM output match what's shown in the UI
                formatted_context = self._format_context_document_references(context_results)
                
                # For method queries, extract method implementations with target interface filter
                if is_method_query and interface_reference and detected_methods:
                    # Group methods by name to deduplicate while preserving document references
                    grouped_methods = {}
                    for method in detected_methods:
                        method_name = method['method_name']
                        if method_name not in grouped_methods:
                            grouped_methods[method_name] = []
                        grouped_methods[method_name].append(method)
                    
                    method_summary = f"\n===== DETECTED METHODS FROM {interface_reference} =====\n"
                    method_summary += f"The following {len(grouped_methods)} methods were found for interface '{interface_reference}':\n\n"
                    
                    for method_name, method_refs in grouped_methods.items():
                        # Use absolute score values to avoid confusion with negative scores
                        method_summary += f"## {method_name}\n"
                        method_summary += "Found in documents:\n"
                        for ref in method_refs:
                            doc_id = ref.get('doc_id', '')
                            doc_title = ref.get('doc_title', '')
                            doc_score = abs(ref.get('doc_score', 0))  # Use absolute score
                            
                            method_summary += f"- [Document {doc_id}: {doc_title} (Score: {doc_score:.4f})]\n"
                        
                        # Include a snippet of context for each method
                        if method_refs:
                            method_summary += "\nContext snippet:\n"
                            method_summary += method_refs[0].get('context', '').strip() + "\n\n"
                    
                    method_summary += "\nIMPORTANT: Please include ALL of these methods in your answer, not just one or two.\n"
                    method_summary += "===== END OF DETECTED METHODS =====\n\n"
                    
                    # Add the method summary at the beginning of the context
                    formatted_context = method_summary + formatted_context
                
                # Step 6: Generate the response
                if is_follow_up:
                    # Include conversation history when generating response
                    response = await self.generator.generate_with_history(
                        processing_query, formatted_context, conversation_history, system_prompt=tailored_system_prompt
                    )
                else:
                    # Standard response without conversation history
                    response = await self.generator.generate(
                        processing_query, formatted_context, system_prompt=tailored_system_prompt
                    )
                    
                # Extract confidence level from response (if present)
                confidence_level, cleaned_response = self._extract_confidence_level(response)
            else:
                # If no relevant results, generate a "not in scope" response
                if is_follow_up:
                    response = await self.generator.generate_out_of_scope_with_history(query, conversation_history)
                else:
                    response = await self.generator.generate_out_of_scope(query)
                contexts = []
                detected_methods = []
                
                # Extract confidence level from response (if present)
                confidence_level, cleaned_response = self._extract_confidence_level(response)
            
            # Prepare return data with complete context information
            result_dict.update({
                "response": cleaned_response,  # Response without the confidence prefix
                "full_response": response,     # Original full response
                "confidence_level": confidence_level,
                "context_count": len(context_results) if has_relevant_results else 0,
                "contexts": contexts,
                "detected_methods": detected_methods if 'detected_methods' in locals() else [],
                "is_follow_up": is_follow_up,
                "has_relevant_results": has_relevant_results
            })
            
            logger.info(f"Query processed successfully with {len(context_results) if has_relevant_results else 0} contexts")
            return result_dict
            
        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            result_dict.update({
                "error": str(e),
                "response": f"I encountered an error while processing your query: {str(e)}",
                "confidence_level": "UNKNOWN",
                "context_count": 0,
                "contexts": [],
                "is_follow_up": is_follow_up
            })
            return result_dict
    
    def save_results(self, results: Dict[str, Any], filename: str = "rag_result.json") -> None:
        """
        Save query results to a JSON file.
        
        Persists the complete results from a query processing operation to a JSON
        file for later analysis, debugging, or record-keeping purposes.
        
        Args:
            results: Results dictionary from process_query containing response,
                    contexts, metadata, and other processing information
            filename: Output file name for saving the results
            
        Note:
            Removes non-serializable fields like 'original_contexts' before saving
            to ensure the JSON file can be properly written and read later.
            
        Examples:
            >>> pipeline = RAGPipeline()
            >>> result = await pipeline.process_query("How do ABAP classes work?")
            >>> pipeline.save_results(result, "query_result.json")
        """
        try:
            # Create a copy without original_contexts (can't be serialized)
            save_results = results.copy()
            if "original_contexts" in save_results:
                del save_results["original_contexts"]
                
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(save_results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving results to {filename}: {e}")
            
    def save_conversation(self, conversation: List[Dict[str, Any]], filename: str = "rag_conversations.json") -> None:
        """
        Save a complete conversation to a JSON file.
        
        Persists a conversation session containing multiple message exchanges
        to a JSON file with metadata about the collection used and timestamps.
        
        Args:
            conversation: List of conversation messages with metadata including
                         user queries, assistant responses, and context information
            filename: Output file name for saving the conversation
            
        Note:
            Appends the conversation to existing conversations in the file if it
            exists, or creates a new file. Each conversation gets a unique ID
            and timestamp for identification.
            
        Examples:
            >>> pipeline = RAGPipeline()
            >>> conversation = [
            ...     {"role": "user", "content": "How do ABAP classes work?"},
            ...     {"role": "assistant", "content": "ABAP classes...", "contexts": [...]}
            ... ]
            >>> pipeline.save_conversation(conversation, "my_conversations.json")
        """
        import os
        import time

        try:
            # Load existing conversations if file exists
            existing_conversations = []
            try:
                if os.path.exists(filename):
                    with open(filename, "r", encoding="utf-8") as f:
                        existing_conversations = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing conversations: {e}")
            
            # Add this conversation
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            conversation_data = {
                "id": f"conv_{len(existing_conversations) + 1}",
                "timestamp": timestamp,
                "collection_type": self.current_collection_type,
                "collection_name": COLLECTIONS[self.current_collection_type]["name"],
                "messages": conversation
            }
            
            existing_conversations.append(conversation_data)
                
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(existing_conversations, f, indent=2, ensure_ascii=False)
            logger.info(f"Conversation saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving conversation to {filename}: {e}")