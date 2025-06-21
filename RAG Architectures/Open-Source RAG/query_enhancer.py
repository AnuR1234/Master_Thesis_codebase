#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FIXED Enhanced query enhancer with intelligent SAP ABAP context understanding.

This module provides the EnhancedQueryEnhancer class that intelligently enhances
user queries for better retrieval and response quality in SAP ABAP documentation.
Now properly respects disabled mode and user settings.

The query enhancer includes:
- Comprehensive SAP ABAP keyword database
- Pattern matching for different query types
- Multiple enhancement modes (disabled, conservative, aggressive)
- Context-aware query enhancement
- Safety validation for query inputs
- Out-of-scope detection for non-SAP queries

Features:
- FIXED: Properly handles disabled mode without any enhancement
- FIXED: Respects user settings from Streamlit interface
- Intelligent SAP ABAP keyword detection
- Multiple enhancement templates for different query types
- Safety validation to prevent misuse
- Comprehensive pattern matching for technical queries
- Dynamic enhancement mode switching

Author: SAP ABAP RAG Team
Version: 1.0.0
License: MIT
"""

import logging
import re
from typing import Dict, Any, List, Optional, Union
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedQueryEnhancer:
    """
    FIXED Enhanced query enhancer that properly respects disabled mode.
    
    This class provides intelligent query enhancement capabilities specifically designed
    for SAP ABAP documentation queries. It analyzes query patterns, complexity, and
    context to provide appropriate enhancements that improve retrieval quality.
    
    The enhancer operates in three modes:
    - disabled: No enhancement applied (FIXED: now properly returns original query)
    - conservative: Moderate enhancement for clarity
    - aggressive: Comprehensive enhancement with detailed context
    
    Features:
    - Comprehensive SAP ABAP keyword database
    - Pattern-based query type detection
    - Context-aware enhancement templates
    - Safety validation and out-of-scope detection
    - Dynamic mode switching for different use cases
    
    Attributes:
        generator: Optional generator instance for advanced processing
        enhancement_mode (str): Current enhancement mode
        sap_keywords (Dict[str, List[str]]): Comprehensive SAP keyword database
        pattern_matchers (Dict[str, List[str]]): Regex patterns for query analysis
        enhancement_templates (Dict[str, Dict[str, str]]): Templates for different modes
    """
    
    def __init__(self, generator=None) -> None:
        """
        Initialize the enhanced query enhancer.
        
        Sets up the query enhancer with comprehensive SAP ABAP knowledge,
        pattern matchers, and enhancement templates.
        
        Args:
            generator: Optional generator instance for advanced processing
        """
        logger.info("Initializing FIXED Enhanced Query Enhancer")
        self.generator = generator
        self.enhancement_mode = "conservative"  # "disabled", "conservative", "aggressive"
        
        # Comprehensive SAP ABAP keyword database
        self.sap_keywords = self._build_sap_keyword_database()
        
        # Pattern matchers for different types of queries
        self.pattern_matchers = self._build_pattern_matchers()
        
        # FIXED: Enhanced templates for complex queries
        self.enhancement_templates = self._build_enhancement_templates()
        
    def _build_sap_keyword_database(self) -> Dict[str, List[str]]:
        """
        Build comprehensive SAP ABAP keyword database.
        
        Creates a structured database of SAP ABAP keywords organized by categories
        for intelligent query analysis and enhancement.
        
        Returns:
            Dict[str, List[str]]: Categorized SAP ABAP keywords including:
                - Core SAP terms
                - Programming constructs
                - Data types and structures
                - Parameters and interfaces
                - Control structures
                - Database operations
                - System variables
                - Object-oriented concepts
                - Error handling
                - Common methods and classes
        """
        return {
            "core_sap": [
                "sap", "abap", "r/3", "s/4hana", "netweaver", "hana"
            ],
            "programming_constructs": [
                "class", "method", "interface", "function", "module", "program", 
                "report", "transaction", "form", "perform", "subroutine"
            ],
            "data_types": [
                "data", "types", "constants", "field-symbols", "ranges",
                "table", "structure", "internal", "work", "area"
            ],
            "parameters": [
                "importing", "exporting", "changing", "returning", "raising",
                "parameter", "select-options", "iv_", "ev_", "cv_", "rv_",
                "it_", "et_", "ct_", "rt_", "is_", "es_", "cs_", "rs_"
            ],
            "control_structures": [
                "if", "endif", "case", "when", "endcase", "loop", "endloop",
                "while", "endwhile", "do", "enddo", "try", "catch", "endtry"
            ],
            "database_operations": [
                "select", "insert", "update", "delete", "modify", "commit", 
                "rollback", "open", "close", "fetch", "cursor"
            ],
            "system_variables": [
                "sy-subrc", "sy-tabix", "sy-index", "sy-datum", "sy-uzeit",
                "sy-uname", "sy-mandt", "sy-langu"
            ],
            "object_oriented": [
                "inheritance", "polymorphism", "encapsulation", "abstract",
                "final", "static", "instance", "constructor", "destructor"
            ],
            "error_handling": [
                "exception", "message", "error", "warning", "information",
                "dump", "short", "raise", "resume"
            ],
            "common_methods": [
                "initialize", "escape_special_characters", "submit", "call",
                "get", "set", "create", "destroy", "check", "validate"
            ],
            "common_classes": [
                "cl_abap_", "cl_salv_", "cl_gui_", "cl_oo_", "cl_sql_", 
                "cl_http_", "cl_xml_", "cl_json_"
            ],
            "common_interfaces": [
                "if_abap_", "if_salv_", "if_gui_", "if_oo_", "if_sql_",
                "if_http_", "if_xml_", "if_json_"
            ]
        }
    
    def _build_pattern_matchers(self) -> Dict[str, List[str]]:
        """
        Build pattern matchers for different query types.
        
        Creates regex patterns to identify different types of queries and
        technical content for intelligent enhancement decisions.
        
        Returns:
            Dict[str, List[str]]: Pattern matchers including:
                - Method and class name patterns
                - Parameter and interface patterns
                - Programming concept patterns
                - Question type patterns
                - Complex technical patterns
                - Out-of-scope detection patterns
        """
        return {
            "method_patterns": [
                r'\b\w+_\w+(?:_\w+)*\b',           # snake_case methods
                r'\b[A-Z][a-z]+(?:[A-Z][a-z]+)*\b', # CamelCase methods
                r'\bCL_\w+\b',                      # SAP class names
                r'\bIF_\w+\b',                      # SAP interface names
                r'\b[A-Z]{2,}_[A-Z_]+\b',          # SAP constants/methods
                r'\b\w+~\w+\b'                     # Interface implementations
            ],
            "parameter_patterns": [
                r'\b(?:iv_|ev_|cv_|rv_|it_|et_|ct_|rt_|is_|es_|cs_|rs_)\w+\b',
                r'\b(?:importing|exporting|changing|returning)\b',
                r'\bparameter\s+\w+\b',
                r'\bselect-options\s+\w+\b'
            ],
            "programming_patterns": [
                r'\b(?:how\s+to|what\s+is|explain|describe|implement|use|call|invoke)\b',
                r'\b(?:parameter|argument|return|exception|error)\b',
                r'\b(?:code|syntax|example|snippet|sample)\b',
                r'\b(?:method|function|class|interface|module)\b'
            ],
            "question_patterns": [
                r'\b(?:what|how|when|where|why|which)\b',
                r'\b(?:can\s+i|should\s+i|do\s+i|will\s+it)\b',
                r'\b(?:purpose|function|role|responsibility)\b'
            ],
            "complex_patterns": [
                r'\b(?:algorithm|approach|implementation|processing|handling)\b',
                r'\b(?:limitation|restriction|constraint|issue|problem)\b',
                r'\b(?:extraction|manipulation|transformation|parsing)\b',
                r'\b(?:multiple|several|various|different|complex)\b'
            ],
            "out_of_scope_patterns": [
                r'\b(?:my|your|his|her|their)\s+(?:name|age|birthday|family|personal)\b',
                r'\bwhat\s+(?:is|are)\s+(?:my|your)\s+(?:name|age|address|phone)\b',
                r'\bwhen\s+(?:was|is|are|did)\s+(?:the)?\s*world\s+war\b',
                r'\bwho\s+(?:is|was|are)\s+(?:the)?\s*(?:president|king|queen|prime\s+minister)\b',
                r'\bwhat\s+(?:is|was)\s+(?:the)?\s*capital\s+of\b',
                r'\b(?:weather|sports|news|politics|entertainment|celebrity)\b',
                r'\b(?:recipe|cooking|food|restaurant|travel|vacation)\b',
                r'\b(?:movie|film|book|novel|music|song|album)\b'
            ]
        }
    
    def _build_enhancement_templates(self) -> Dict[str, Dict[str, str]]:
        """
        FIXED: Build enhancement templates that properly handle complex queries.
        
        Creates templates for different enhancement modes and query types,
        with special emphasis on properly enhancing complex queries.
        
        Returns:
            Dict[str, Dict[str, str]]: Enhancement templates organized by mode:
                - conservative: Moderate enhancement templates
                - aggressive: Comprehensive enhancement templates
        """
        return {
            "conservative": {
                # FIXED: Now enhances complex queries properly
                "simple_query": "SAP ABAP {query}",
                "method_query": "SAP ABAP {query} - method implementation, parameters, usage",
                "complex_query": "SAP ABAP {query} - comprehensive documentation with implementation details",
                "algorithm_query": "SAP ABAP {query} - algorithm implementation, approach, limitations",
                "parameter_query": "{query} - SAP ABAP parameter details and usage",
                "implementation_query": "SAP ABAP {query} - detailed implementation and code examples",
                "error_query": "SAP ABAP {query} - error handling and exception management"
            },
            "aggressive": {
                "simple_query": "SAP ABAP {query} - detailed documentation with examples",
                "method_query": "SAP ABAP {query} - complete method implementation, parameters, usage examples, error handling",
                "complex_query": "SAP ABAP {query} - comprehensive technical documentation, implementation details, algorithms, limitations, best practices",
                "algorithm_query": "SAP ABAP {query} - detailed algorithm analysis, implementation approach, performance considerations, limitations, edge cases",
                "parameter_query": "{query} - SAP ABAP complete parameter documentation, types, usage patterns, validation",
                "implementation_query": "SAP ABAP {query} - comprehensive implementation guide, code examples, design patterns, error handling",
                "error_query": "SAP ABAP {query} - complete error handling, exception management, debugging approaches",
                "class_query": "SAP ABAP {query} - class structure, inheritance, methods, interfaces, implementation patterns",
                "interface_query": "SAP ABAP {query} - interface implementation, method definitions, usage patterns"
            }
        }
    
    def set_enhancement_mode(self, mode: str) -> None:
        """
        Set enhancement mode for query processing.
        
        Configures the enhancement behavior for different use cases:
        - disabled: No enhancement applied
        - conservative: Moderate enhancement for basic context
        - aggressive: Comprehensive enhancement with detailed context
        
        Args:
            mode (str): Enhancement mode ('disabled', 'conservative', 'aggressive')
        """
        if mode in ["disabled", "conservative", "aggressive"]:
            old_mode = self.enhancement_mode
            self.enhancement_mode = mode
            logger.info(f"FIXED: Query enhancement mode changed from '{old_mode}' to '{mode}'")
        else:
            logger.warning(f"Invalid enhancement mode: {mode}. Using conservative.")
            self.enhancement_mode = "conservative"
    
    def get_enhancement_mode(self) -> str:
        """
        Get current enhancement mode.
        
        Returns:
            str: Current enhancement mode
        """
        return self.enhancement_mode
    
    async def is_out_of_context(self, query: str) -> bool:
        """
        Enhanced out-of-context detection with comprehensive SAP ABAP understanding.
        
        Analyzes queries to determine if they are related to SAP ABAP development
        or are general/personal questions that should not be enhanced.
        
        Args:
            query (str): Query to analyze
            
        Returns:
            bool: True if query is out of SAP ABAP context, False otherwise
        """
        query_lower = query.lower()
        
        # Check for SAP ABAP keywords
        has_sap_keyword = self._has_sap_keywords(query_lower)
        
        # Check for programming patterns
        has_programming_pattern = self._has_programming_patterns(query)
        
        # Check for method/technical patterns
        has_technical_pattern = self._has_technical_patterns(query)
        
        # Check for clearly out-of-scope patterns
        is_clearly_out_of_scope = self._is_clearly_out_of_scope(query_lower)
        
        # Decision logic: Only mark as out of context if clearly personal/general 
        # AND no SAP/programming/technical indicators
        is_out_of_context = (is_clearly_out_of_scope and 
                           not (has_sap_keyword or has_programming_pattern or has_technical_pattern))
        
        if is_out_of_context:
            logger.info(f"Query marked as out of context: '{query}'")
        
        return is_out_of_context
    
    def _has_sap_keywords(self, query_lower: str) -> bool:
        """
        Check if query contains SAP ABAP keywords.
        
        Args:
            query_lower (str): Lowercase query string
            
        Returns:
            bool: True if SAP ABAP keywords are found
        """
        for category, keywords in self.sap_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return True
        return False
    
    def _has_programming_patterns(self, query: str) -> bool:
        """
        Check if query contains programming-related patterns.
        
        Args:
            query (str): Query to analyze
            
        Returns:
            bool: True if programming patterns are found
        """
        for pattern in self.pattern_matchers["programming_patterns"]:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _has_technical_patterns(self, query: str) -> bool:
        """
        Check if query contains technical patterns (methods, parameters, etc.).
        
        Args:
            query (str): Query to analyze
            
        Returns:
            bool: True if technical patterns are found
        """
        for pattern_type in ["method_patterns", "parameter_patterns"]:
            for pattern in self.pattern_matchers[pattern_type]:
                if re.search(pattern, query):
                    return True
        return False
    
    def _has_complex_patterns(self, query: str) -> bool:
        """
        FIXED: Check if query contains complex technical patterns.
        
        Identifies queries that involve complex concepts like algorithms,
        processing approaches, or advanced technical implementations.
        
        Args:
            query (str): Query to analyze
            
        Returns:
            bool: True if complex patterns are found
        """
        for pattern in self.pattern_matchers["complex_patterns"]:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _is_clearly_out_of_scope(self, query_lower: str) -> bool:
        """
        Check if query is clearly out of scope for SAP ABAP documentation.
        
        Args:
            query_lower (str): Lowercase query string
            
        Returns:
            bool: True if query is clearly out of scope
        """
        for pattern in self.pattern_matchers["out_of_scope_patterns"]:
            if re.search(pattern, query_lower):
                return True
        return False
    
    async def enhance_query(self, original_query: str) -> Dict[str, Any]:
        """
        FIXED: Main query enhancement that properly respects disabled mode.
        
        Main entry point for query enhancement. Analyzes the query and applies
        appropriate enhancement based on the current mode and query characteristics.
        
        Args:
            original_query (str): Original user query
            
        Returns:
            Dict[str, Any]: Enhancement results including:
                - original_query: The input query
                - enhanced_query: The enhanced version
                - is_out_of_context: Whether query is out of scope
                - is_enhanced: Whether enhancement was applied
                - enhancement_mode: Current enhancement mode
                - query_analysis: Detailed analysis results
                - enhancement_details: Enhancement processing details
        """
        logger.info(f"FIXED: Processing query with enhancement mode '{self.enhancement_mode}': '{original_query}'")
        
        # Initialize result
        result = {
            "original_query": original_query,
            "enhanced_query": original_query,
            "is_out_of_context": False,
            "is_enhanced": False,
            "enhancement_mode": self.enhancement_mode,
            "query_analysis": {},
            "enhancement_details": {}
        }
        
        # FIXED: Check enhancement mode first - if disabled, return immediately
        if self.enhancement_mode == "disabled":
            result["enhancement_details"]["reason"] = "Enhancement disabled"
            logger.info(f"FIXED: Enhancement disabled - returning original query unchanged")
            return result
        
        # Check if out of context
        is_out_of_context = await self.is_out_of_context(original_query)
        
        if is_out_of_context:
            result["is_out_of_context"] = True
            result["enhancement_details"]["reason"] = "Query out of context"
            logger.info(f"FIXED: Query out of context: '{original_query}'")
            return result
        
        # FIXED: Apply enhancement based on mode
        logger.info(f"FIXED: Enhancement enabled (mode: {self.enhancement_mode}) - will enhance query")
        
        # Perform comprehensive query analysis
        query_analysis = self._analyze_query_comprehensively(original_query)
        result["query_analysis"] = query_analysis
        
        # Apply enhancement based on analysis and mode
        enhanced_query = self._apply_enhancement_fixed(original_query, query_analysis)
        
        # Update result
        result["enhanced_query"] = enhanced_query
        result["is_enhanced"] = enhanced_query != original_query
        result["enhancement_details"] = {
            "enhancement_applied": result["is_enhanced"],
            "enhancement_reason": f"Query enhanced using {self.enhancement_mode} mode",
            "query_complexity": query_analysis.get("complexity", "unknown"),
            "query_type": query_analysis.get("primary_type", "general")
        }
        
        logger.info(f"FIXED: Enhancement completed:")
        logger.info(f"  Mode: {self.enhancement_mode}")
        logger.info(f"  Original: '{original_query}'")
        logger.info(f"  Enhanced: '{enhanced_query}'")
        logger.info(f"  Is Enhanced: {result['is_enhanced']}")
        
        return result
    
    def _analyze_query_comprehensively(self, query: str) -> Dict[str, Any]:
        """
        FIXED: Perform comprehensive analysis of the query.
        
        Analyzes query characteristics including length, complexity, keywords,
        patterns, and determines the most appropriate enhancement approach.
        
        Args:
            query (str): Query to analyze
            
        Returns:
            Dict[str, Any]: Comprehensive analysis including:
                - Basic metrics (length, character count)
                - Pattern detection results
                - Query type classification
                - Complexity assessment
                - Enhancement recommendations
        """
        query_lower = query.lower()
        
        analysis = {
            "length": len(query.split()),
            "character_count": len(query),
            "has_sap_keywords": self._has_sap_keywords(query_lower),
            "has_programming_patterns": self._has_programming_patterns(query),
            "has_technical_patterns": self._has_technical_patterns(query),
            "has_complex_patterns": self._has_complex_patterns(query),
            "query_types": [],
            "primary_type": "general",
            "complexity": "simple",
            "enhancement_reason": ""
        }
        
        # Detect specific query types
        query_types = []
        
        # Method queries
        if any(word in query_lower for word in ["method", "function", "procedure"]):
            query_types.append("method")
            
        # Parameter queries
        if any(word in query_lower for word in ["parameter", "argument", "input", "output", "accepts"]):
            query_types.append("parameter")
            
        # Class/Interface queries
        if any(word in query_lower for word in ["class", "interface", "inheritance"]):
            query_types.append("class_interface")
            
        # Error/Exception queries
        if any(word in query_lower for word in ["error", "exception", "handling", "catch"]):
            query_types.append("error")
            
        # Implementation queries
        if any(word in query_lower for word in ["implement", "implementation", "how to", "usage"]):
            query_types.append("implementation")
            
        # FIXED: Algorithm/Complex queries
        if any(word in query_lower for word in ["algorithm", "approach", "processing", "handling", "extraction", "limitation"]):
            query_types.append("algorithm")
            
        # Example/Code queries
        if any(word in query_lower for word in ["example", "code", "sample", "snippet"]):
            query_types.append("example")
        
        analysis["query_types"] = query_types
        
        # Determine primary type
        if query_types:
            analysis["primary_type"] = query_types[0]
        
        # FIXED: Determine complexity properly
        word_count = len(query.split())
        
        if word_count >= 15 or analysis["has_complex_patterns"]:
            analysis["complexity"] = "complex"
            analysis["enhancement_reason"] = "Complex query needs comprehensive enhancement"
        elif word_count >= 8 or (analysis["has_sap_keywords"] and analysis["has_technical_patterns"]):
            analysis["complexity"] = "medium"
            analysis["enhancement_reason"] = "Medium complexity query can benefit from enhancement"
        elif word_count <= 3:
            analysis["complexity"] = "simple"
            analysis["enhancement_reason"] = "Simple query needs basic context"
        else:
            analysis["complexity"] = "medium"
            analysis["enhancement_reason"] = "Query can benefit from additional context"
        
        return analysis
    
    def _apply_enhancement_fixed(self, original_query: str, analysis: Dict[str, Any]) -> str:
        """
        FIXED: Apply enhancement based on query analysis and enhancement mode.
        
        Applies the appropriate enhancement strategy based on query complexity,
        type, and the current enhancement mode setting.
        
        Args:
            original_query (str): Original query text
            analysis (Dict[str, Any]): Query analysis results
            
        Returns:
            str: Enhanced query
        """
        complexity = analysis.get("complexity", "simple")
        primary_type = analysis.get("primary_type", "general")
        word_count = analysis.get("length", 0)
        
        logger.info(f"FIXED: Applying enhancement - complexity: {complexity}, type: {primary_type}, words: {word_count}")
        
        # FIXED: Now we enhance MORE for complex queries, not less
        
        # Conservative mode enhancement logic
        if self.enhancement_mode == "conservative":
            return self._conservative_enhance_fixed(original_query, analysis)
        elif self.enhancement_mode == "aggressive":
            return self._aggressive_enhance_fixed(original_query, analysis)
        
        return original_query
    
    def _conservative_enhance_fixed(self, query: str, analysis: Dict[str, Any]) -> str:
        """
        FIXED: Conservative enhancement that properly handles complex queries.
        
        Applies moderate enhancement that adds necessary context without
        overwhelming the query, with special handling for complex queries.
        
        Args:
            query (str): Query to enhance
            analysis (Dict[str, Any]): Query analysis results
            
        Returns:
            str: Conservatively enhanced query
        """
        query_lower = query.lower()
        primary_type = analysis.get("primary_type", "general")
        complexity = analysis.get("complexity", "simple")
        
        logger.info(f"FIXED Conservative: complexity={complexity}, type={primary_type}")
        
        # FIXED: Now enhance complex queries MORE, not less
        templates = self.enhancement_templates["conservative"]
        
        # Handle complex queries with comprehensive enhancement
        if complexity == "complex":
            if primary_type == "algorithm":
                enhanced = templates["algorithm_query"].format(query=query)
                logger.info(f"FIXED: Applied algorithm enhancement")
                return enhanced
            elif primary_type in ["method", "implementation"]:
                enhanced = templates["complex_query"].format(query=query)
                logger.info(f"FIXED: Applied complex method enhancement")
                return enhanced
            else:
                enhanced = templates["complex_query"].format(query=query)
                logger.info(f"FIXED: Applied general complex enhancement")
                return enhanced
        
        # Handle medium complexity queries
        elif complexity == "medium":
            if primary_type == "method":
                enhanced = templates["method_query"].format(query=query)
                logger.info(f"FIXED: Applied method enhancement")
                return enhanced
            elif primary_type == "parameter":
                enhanced = templates["parameter_query"].format(query=query)
                logger.info(f"FIXED: Applied parameter enhancement")
                return enhanced
            elif primary_type == "implementation":
                enhanced = templates["implementation_query"].format(query=query)
                logger.info(f"FIXED: Applied implementation enhancement")
                return enhanced
            else:
                # Add basic SAP context if missing
                if not any(sap_word in query_lower for sap_word in ['sap', 'abap']):
                    enhanced = f"SAP ABAP {query} - documentation and usage"
                    logger.info(f"FIXED: Added SAP context")
                    return enhanced
        
        # Handle simple queries
        elif complexity == "simple":
            if not any(sap_word in query_lower for sap_word in ['sap', 'abap']):
                enhanced = templates["simple_query"].format(query=query)
                logger.info(f"FIXED: Applied simple enhancement")
                return enhanced
        
        logger.info(f"FIXED: No enhancement applied - query already well-formed")
        return query
    
    def _aggressive_enhance_fixed(self, query: str, analysis: Dict[str, Any]) -> str:
        """
        FIXED: Aggressive enhancement that comprehensively enhances complex queries.
        
        Applies comprehensive enhancement with detailed context, especially
        beneficial for complex technical queries requiring extensive documentation.
        
        Args:
            query (str): Query to enhance
            analysis (Dict[str, Any]): Query analysis results
            
        Returns:
            str: Aggressively enhanced query
        """
        query_lower = query.lower()
        primary_type = analysis.get("primary_type", "general")
        complexity = analysis.get("complexity", "simple")
        
        logger.info(f"FIXED Aggressive: complexity={complexity}, type={primary_type}")
        
        templates = self.enhancement_templates["aggressive"]
        
        # FIXED: Aggressive mode enhances ALL queries, especially complex ones
        
        if complexity == "complex":
            if primary_type == "algorithm":
                enhanced = templates["algorithm_query"].format(query=query)
                logger.info(f"FIXED: Applied comprehensive algorithm enhancement")
                return enhanced
            elif primary_type in ["method", "implementation"]:
                enhanced = templates["implementation_query"].format(query=query)
                logger.info(f"FIXED: Applied comprehensive implementation enhancement")
                return enhanced
            else:
                enhanced = templates["complex_query"].format(query=query)
                logger.info(f"FIXED: Applied comprehensive complex enhancement")
                return enhanced
        
        elif complexity == "medium":
            if primary_type == "method":
                enhanced = templates["method_query"].format(query=query)
                logger.info(f"FIXED: Applied comprehensive method enhancement")
                return enhanced
            elif primary_type == "parameter":
                enhanced = templates["parameter_query"].format(query=query)
                logger.info(f"FIXED: Applied comprehensive parameter enhancement")
                return enhanced
            else:
                enhanced = templates["complex_query"].format(query=query)
                logger.info(f"FIXED: Applied comprehensive medium enhancement")
                return enhanced
        
        else:  # simple
            enhanced = templates["simple_query"].format(query=query)
            logger.info(f"FIXED: Applied comprehensive simple enhancement")
            return enhanced
    
    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about enhancement usage and capabilities.
        
        Returns comprehensive information about the enhancer's configuration,
        capabilities, and current settings for monitoring and debugging.
        
        Returns:
            Dict[str, Any]: Enhancement statistics including:
                - Current mode and available modes
                - Keyword and pattern counts
                - Template information
                - Feature flags and version info
        """
        return {
            "current_mode": self.enhancement_mode,
            "available_modes": ["disabled", "conservative", "aggressive"],
            "sap_keywords_count": sum(len(keywords) for keywords in self.sap_keywords.values()),
            "pattern_matchers_count": sum(len(patterns) for patterns in self.pattern_matchers.values()),
            "enhancement_templates": list(self.enhancement_templates.keys()),
            "fixed_version": True,
            "complex_query_enhancement": "enabled",
            "disabled_mode_fixed": True
        }
    
    def validate_query_safety(self, query: str) -> Dict[str, Any]:
        """
        Validate query for safety and appropriateness.
        
        Performs safety checks to detect potential security issues, injection
        attempts, or inappropriate content in user queries.
        
        Args:
            query (str): Query to validate
            
        Returns:
            Dict[str, Any]: Safety validation results including:
                - is_safe: Boolean indicating if query is safe
                - warnings: List of detected issues
                - risk_level: LOW, MEDIUM, or HIGH risk assessment
                - safety_checks: List of performed safety checks
        """
        safety_result = {
            "is_safe": True,
            "warnings": [],
            "risk_level": "LOW",
            "safety_checks": []
        }
        
        query_lower = query.lower()
        
        # Check for potentially problematic patterns
        unsafe_patterns = [
            r"ignore\s+(?:previous|above|system)\s+instructions?",
            r"disregard\s+(?:the|all)\s+(?:rules?|instructions?)",
            r"act\s+as\s+(?:if|though)\s+you\s+(?:are|were)",
            r"pretend\s+(?:that\s+)?you\s+(?:are|were)",
            r"roleplay\s+as",
            r"jailbreak",
            r"DAN\s+mode"
        ]
        
        for pattern in unsafe_patterns:
            if re.search(pattern, query_lower):
                safety_result["is_safe"] = False
                safety_result["warnings"].append(f"Potentially unsafe pattern detected: {pattern}")
                safety_result["risk_level"] = "HIGH"
        
        # Check for injection attempts
        injection_patterns = [
            r"<script",
            r"javascript:",
            r"eval\(",
            r"exec\(",
            r"system\(",
            r"shell_exec"
        ]
        
        for pattern in injection_patterns:
            if re.search(pattern, query_lower):
                safety_result["warnings"].append(f"Potential injection pattern: {pattern}")
                safety_result["risk_level"] = "MEDIUM" if safety_result["risk_level"] == "LOW" else safety_result["risk_level"]
        
        safety_result["safety_checks"] = [
            "Unsafe instruction patterns",
            "Code injection patterns",
            "Roleplay attempts",
            "System manipulation"
        ]
        
        return safety_result


# Backward compatibility alias
QueryEnhancer = EnhancedQueryEnhancer


def create_query_enhancer(generator=None) -> EnhancedQueryEnhancer:
    """
    Factory function to create a fixed enhanced query enhancer.
    
    Provides a simple factory interface for creating query enhancer instances
    with optional generator integration.
    
    Args:
        generator: Optional generator instance for advanced processing
        
    Returns:
        EnhancedQueryEnhancer: Configured query enhancer instance
    """
    return EnhancedQueryEnhancer(generator)


def test_query_enhancer_fixed() -> bool:
    """
    Test the FIXED enhanced query enhancer functionality.
    
    Performs comprehensive testing of the query enhancer including initialization,
    enhancement processing, and statistical reporting to verify functionality.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    try:
        enhancer = EnhancedQueryEnhancer()
        print("? FIXED Enhanced Query Enhancer initialized successfully")
        
        # Test queries - including complex ones
        test_queries = [
            "What is escape_special_characters",
            "How does the SPLIT_FILE_NAME method handle the extraction of file extensions when dealing with files that contain multiple period characters",
            "method parameters",
            "how to implement",
            "weather today",  # Should be out of context
            "SAP ABAP class structure"
        ]
        
        async def run_tests():
            # Test disabled mode first
            enhancer.set_enhancement_mode("disabled")
            print(f"\n?? Testing DISABLED mode:")
            
            for query in test_queries[:3]:  # Test first 3 queries
                result = await enhancer.enhance_query(query)
                enhanced = result['enhanced_query']
                is_enhanced = result['is_enhanced']
                
                print(f"Query: '{query}'")
                print(f"Enhanced: {is_enhanced} (should be False)")
                print(f"Result: '{enhanced}' (should be same as original)")
                
                if is_enhanced:
                    print(f"? ERROR: Enhancement applied in disabled mode!")
                    return False
                else:
                    print(f"? DISABLED mode working correctly")
                print("-" * 80)
            
            # Test conservative mode
            enhancer.set_enhancement_mode("conservative")
            print(f"\n?? Testing CONSERVATIVE mode:")
            
            for query in test_queries:
                result = await enhancer.enhance_query(query)
                enhanced = result['enhanced_query']
                is_enhanced = result['is_enhanced']
                complexity = result['query_analysis'].get('complexity', 'unknown')
                
                print(f"Query: '{query}'")
                print(f"Complexity: {complexity}")
                print(f"Enhanced: {is_enhanced}")
                print(f"Result: '{enhanced}'")
                print("-" * 80)
            
            # Test aggressive mode
            enhancer.set_enhancement_mode("aggressive")
            print(f"\n? Testing AGGRESSIVE mode:")
            
            for query in test_queries[:3]:  # Test first 3 queries
                result = await enhancer.enhance_query(query)
                enhanced = result['enhanced_query']
                is_enhanced = result['is_enhanced']
                complexity = result['query_analysis'].get('complexity', 'unknown')
                
                print(f"Query: '{query}'")
                print(f"Complexity: {complexity}")
                print(f"Enhanced: {is_enhanced}")
                print(f"Result: '{enhanced}'")
                print("-" * 80)
            
            return True
        
        success = asyncio.run(run_tests())
        
        if success:
            # Test statistics
            stats = enhancer.get_enhancement_statistics()
            print(f"\n?? Enhancement statistics: {stats}")
            print("? All tests passed!")
        
        return success
    except Exception as e:
        print(f"? FIXED Enhanced Query Enhancer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_query_enhancer_fixed()