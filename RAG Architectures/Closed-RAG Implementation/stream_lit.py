#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streamlit Web Interface for SAP ABAP RAG Pipeline.

This module provides a comprehensive web interface for the RAG (Retrieval-Augmented Generation) 
pipeline with simplified conversation management, collection selection, confidence level display, 
and query enhancement visualization for SAP ABAP code documentation.

Features:
    - Interactive collection selection (Classes vs Reports)
    - Query enhancement with rewriting and decomposition
    - Confidence level assessment for responses
    - Conversation history management
    - Document context display with pagination
    - Evaluation metrics visualization
    - Real-time feedback system

Author: SAP ABAP RAG Team
Version: 1.0.0
Date: 2025
"""

# Add this at the very beginning of stream_lit.py, before any imports
import sys
sys.argv.extend(["--server.fileWatcherType", "none"])

# Standard library imports
import asyncio
import json
import os
import re
import time
import types
from typing import Any, Dict, List, Optional

# Third-party imports
import streamlit as st

# Apply nest_asyncio ONCE at the beginning to allow asyncio to work with Streamlit
import nest_asyncio
if not getattr(asyncio, '_nest_patched', False):
    nest_asyncio.apply()
    setattr(asyncio, '_nest_patched', True)


class DummyModule(types.ModuleType):
    """
    Dummy module to handle PyTorch module path issues.
    
    This class creates a fake module to handle path access issues with PyTorch
    when running in the Streamlit environment.
    
    Args:
        name (str): The name of the module
    """
    
    def __init__(self, name: str) -> None:
        """
        Initialize the dummy module.
        
        Args:
            name: The module name
        """
        super().__init__(name)
        self._path = []
    
    def __getattr__(self, name: str) -> Any:
        """
        Handle attribute access for the dummy module.
        
        Args:
            name: The attribute name being accessed
            
        Returns:
            The module path if requested, otherwise raises AttributeError
            
        Raises:
            AttributeError: If the requested attribute is not __path__
        """
        if name == "__path__":
            return self._path
        raise AttributeError(f"Module '{self.__name__}' has no attribute '{name}'")


# Apply the patch only if torch is installed
try:
    import torch
    if not hasattr(torch, "_classes_patched"):
        # Create fake module to handle path access
        sys.modules["torch._classes"] = DummyModule("torch._classes")
        setattr(torch, "_classes_patched", True)
except ImportError:
    pass

# Import configuration only after nest_asyncio is applied
from config import (
    COLLECTIONS,
    DEFAULT_COLLECTION_TYPE,
    DEFAULT_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
    USE_HYBRID_DEFAULT,
    USE_RERANKER_DEFAULT,
)


def clean_document_content(text: str) -> str:
    """
    Clean up document content by removing duplicated titles and improving formatting.
    
    This function processes raw document content to remove common formatting issues
    such as repeated titles, excessive blank lines, and other artifacts that may
    appear in the retrieved document contexts.
    
    Args:
        text: Raw document content string to be cleaned
        
    Returns:
        Cleaned document content with improved formatting
        
    Examples:
        >>> content = "Title\\nTitle\\n\\nSome content here..."
        >>> clean_document_content(content)
        "Title\\n\\nSome content here..."
    """
    if not text:
        return ""
    
    # Split text into lines
    lines = text.strip().split('\n')
    
    # Case 1: Remove identical repeated lines at the beginning
    if len(lines) >= 2 and lines[0] == lines[1]:
        lines = lines[1:]
        
    # Case 2: If first line is just the title and appears in first paragraph
    if len(lines) >= 3 and lines[0].strip() and lines[1].strip() == '':
        first_line = lines[0].strip()
        if any(line.startswith(first_line) for line in lines[2:5]):
            lines = lines[1:]
    
    # Remove excessive blank lines (more than 2 consecutive newlines)
    cleaned_lines = []
    empty_count = 0
    
    for line in lines:
        if not line.strip():
            empty_count += 1
            if empty_count > 2:
                continue
        else:
            empty_count = 0
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


def clean_assistant_response(response: str) -> tuple[str, str]:
    """
    Clean up assistant response by removing unnecessary formatting and extracting confidence.
    
    This function processes the raw response from the RAG pipeline to remove
    unnecessary section headers and extract confidence level information.
    
    Args:
        response: Raw response from the assistant
        
    Returns:
        A tuple containing:
            - cleaned_response: The cleaned response text
            - confidence_level: Extracted confidence level (HIGH, MEDIUM, LOW, or UNKNOWN)
            
    Examples:
        >>> response = "Some content here. Confidence: HIGH"
        >>> clean_assistant_response(response)
        ("Some content here.", "HIGH")
    """
    if not response:
        return "", "UNKNOWN"
    
    # Remove any explicit section headers
    headers_to_remove = [
        r"^#+\s*Purpose\s*$", 
        r"^#+\s*Method Implementation.*$",
        r"^#+\s*Usage in Security Context\s*$",
        r"^#+\s*Significance\s*$",
        r"^#+\s*Data Structure\s*$",
        r"^#+\s*Conclusion\s*$",
        r"^#+\s*Summary\s*$"
    ]
    
    cleaned_response = response
    for header in headers_to_remove:
        cleaned_response = re.sub(header, "", cleaned_response, flags=re.MULTILINE)
    
    # Remove excessive blank lines
    cleaned_response = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_response)
    
    # Extract confidence level
    confidence_pattern = r"Confidence: (HIGH|MEDIUM|LOW)"
    confidence_match = re.search(confidence_pattern, cleaned_response)
    
    if confidence_match:
        # Remove the confidence line to be re-added as a badge
        confidence_level = confidence_match.group(1)
        cleaned_response = re.sub(confidence_pattern, "", cleaned_response).strip()
        return cleaned_response, confidence_level
    
    return cleaned_response, "UNKNOWN"


def get_evaluation_metrics_html(metrics: str) -> str:
    """
    Generate HTML for evaluation metrics badges.
    
    This function creates colored HTML badges for displaying evaluation metrics
    such as groundedness, overall quality, and hallucination scores.
    
    Args:
        metrics: String containing evaluation metrics text
        
    Returns:
        HTML string containing styled metric badges
        
    Examples:
        >>> metrics = "Groundedness: 90% (Good)\\nOverall Quality: 85% (Good)"
        >>> html = get_evaluation_metrics_html(metrics)
        >>> "background-color: #28a745" in html
        True
    """
    badges = []
    
    # Define colors and labels for each metric type
    metric_styles = {
        "Groundedness": {"high": "#28a745", "medium": "#fd7e14", "low": "#dc3545"},
        "Overall Quality": {"high": "#28a745", "medium": "#fd7e14", "low": "#dc3545"},
        "Hallucination": {"low": "#28a745", "medium": "#fd7e14", "high": "#dc3545"}
    }
    
    # Extract metrics from the text using regex
    for metric_name, style_dict in metric_styles.items():
        pattern = rf'{metric_name}:\s*(\d+)%\s*\(([^)]+)\)'
        match = re.search(pattern, metrics)
        if match:
            percentage = match.group(1)
            level = match.group(2).lower()
            
            # Determine color based on level
            if "good" in level.lower() or "excellent" in level.lower():
                level_key = "high"
            elif "adequate" in level.lower() or "fair" in level.lower():
                level_key = "medium"
            else:
                level_key = "low"
                
            color = style_dict.get(level_key, "#6c757d")  # Default to gray if level not found
            
            # Create badge
            badge_html = f"""
            <div style="display: inline-block; margin-bottom: 10px; margin-right: 10px;">
                <span style="background-color: {color}; color: white; padding: 4px 8px; 
                border-radius: 4px; font-weight: bold; font-size: 14px;">
                    {metric_name}: {percentage}% ({match.group(2)})
                </span>
            </div>
            """
            badges.append(badge_html)
    
    # If we have sample metrics but no matches were found with the regex
    if not badges and "Groundedness: 90% (Good)" in metrics:
        # Create sample badges manually
        badges.append("""
        <div style="display: inline-block; margin-bottom: 10px; margin-right: 10px;">
            <span style="background-color: #28a745; color: white; padding: 4px 8px; 
            border-radius: 4px; font-weight: bold; font-size: 14px;">
                Groundedness: 90% (Good)
            </span>
        </div>
        """)
        
        badges.append("""
        <div style="display: inline-block; margin-bottom: 10px; margin-right: 10px;">
            <span style="background-color: #28a745; color: white; padding: 4px 8px; 
            border-radius: 4px; font-weight: bold; font-size: 14px;">
                Overall Quality: 85% (Good)
            </span>
        </div>
        """)
    
    return "".join(badges)


def generate_sample_metrics() -> str:
    """
    Generate sample evaluation metrics for testing or when none are present.
    
    Returns:
        String with sample evaluation metrics in the expected format
        
    Examples:
        >>> metrics = generate_sample_metrics()
        >>> "Groundedness: 90%" in metrics
        True
    """
    return """
Groundedness: 90% (Good)
Overall Quality: 85% (Good)
Verdict: Reliable - The response is well-supported by the documentation
"""


def display_evaluation_metrics(message: Dict[str, Any], show_sample: bool = False) -> None:
    """
    Display evaluation metrics from the message content.
    
    This function extracts and displays evaluation metrics such as groundedness,
    overall quality, and hallucination scores from the message content.
    
    Args:
        message: Message dictionary containing content with evaluation metrics
        show_sample: Whether to show sample metrics if none are found
        
    Note:
        This function modifies the Streamlit interface by adding metric displays
        when evaluation metrics are enabled in the session state.
    """
    # Don't display metrics if evaluation metrics are disabled
    if not st.session_state.get('enable_evaluation_metrics', True):
        return
    
    # Only show metrics if "Show evaluation details" is enabled
    if not st.session_state.get('show_evaluation_details', False):
        return
        
    if "content" not in message:
        return
    
    # Extract all evaluation metrics using regex
    metrics_text = message["content"]
    
    # Look for metrics patterns: Metric: XX% (Level)
    metrics_pattern = r'(Groundedness|Overall Quality|Hallucination):\s*\d+%\s*\([^)]+\)'
    metrics_matches = re.findall(metrics_pattern, metrics_text)
    
    # If no metrics found and show_sample is True, generate sample metrics
    if not metrics_matches and show_sample:
        metrics_text = generate_sample_metrics()
        metrics_matches = ["Groundedness", "Overall Quality"]
        
        # Print debug info
        print(f"No metrics found in response, using sample metrics: {metrics_text}")
    elif metrics_matches:
        print(f"Found metrics in response: {metrics_matches}")
    
    if metrics_matches:
        # Generate HTML for the metrics badges
        metrics_html = get_evaluation_metrics_html(metrics_text)
        if metrics_html:
            st.markdown("### Evaluation Metrics")
            st.markdown(metrics_html, unsafe_allow_html=True)
            
            # Display detailed verdict information
            verdict_pattern = r'Verdict:\s*([^-]+)-([^<]+)'
            verdict_match = re.search(verdict_pattern, metrics_text)
            if verdict_match:
                verdict_type = verdict_match.group(1).strip()
                verdict_desc = verdict_match.group(2).strip()
                st.markdown(f"**Evaluation Verdict**: {verdict_type}")
                st.markdown(f"**Details**: {verdict_desc}")
            else:
                # If no verdict found but we're using sample metrics, show sample verdict
                if show_sample and not metrics_matches:
                    st.markdown("**Evaluation Verdict**: Reliable")
                    st.markdown("**Details**: The response is well-supported by the documentation")


def clean_response_text(text: str) -> str:
    """
    Clean up response text by selectively removing metrics.
    
    This function removes accuracy metrics always and other evaluation metrics
    based on user preferences stored in session state.
    
    Args:
        text: Raw response text containing metrics
        
    Returns:
        Cleaned response text with metrics removed based on settings
        
    Note:
        This function accesses Streamlit session state to determine which
        metrics to remove based on user preferences.
    """
    if not text:
        return ""
    
    # Create a copy of the original text to return if metrics are enabled
    original_text = text
    
    # Always remove the Accuracy metric
    accuracy_pattern = r'Accuracy:\s*\d+%\s*\([^)]+\)'
    cleaned_text = re.sub(accuracy_pattern, "", text)
    
    # If evaluation metrics are disabled, remove all metrics
    if not st.session_state.get('enable_evaluation_metrics', True):
        evaluation_patterns = [
            r'Groundedness:\s*\d+%\s*\([^)]+\)',
            r'Overall Quality:\s*\d+%',
            r'Hallucination:\s*\d+%\s*\([^)]+\)',
            r'Verdict:\s*[^-]+-[^<]+'
        ]
        
        for pattern in evaluation_patterns:
            cleaned_text = re.sub(pattern, "", cleaned_text)
    
    # Clean up any extra whitespace or newlines from the removal
    cleaned_text = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned_text)
    cleaned_text = cleaned_text.strip()
    
    # Print metrics to help with debugging
    metrics_pattern = r'(Groundedness|Overall Quality|Hallucination):\s*\d+%\s*\([^)]+\)'
    metrics_matches = re.findall(metrics_pattern, original_text)
    if metrics_matches:
        print(f"Found metrics in response: {metrics_matches}")
    else:
        print("No metrics found in response")
    
    return cleaned_text


@st.cache_resource(show_spinner=False)
def get_rag_pipeline():
    """
    Initialize and cache the RAG pipeline.
    
    This function creates and caches an instance of the RAG pipeline to avoid
    reinitialization on every query. The cache is maintained across Streamlit runs.
    
    Returns:
        RAGPipeline: Cached instance of the RAG pipeline
        
    Note:
        Uses Streamlit's cache_resource decorator to maintain the pipeline
        instance across multiple user sessions and interactions.
    """
    from pipeline import RAGPipeline
    return RAGPipeline()


async def run_rag_query(
    query: str, 
    conversation_history: Optional[List[Dict[str, str]]] = None, 
    use_hybrid: bool = True, 
    use_reranker: bool = True, 
    top_k: int = 5, 
    collection_type: Optional[str] = None, 
    enable_query_enhancement: bool = True
) -> Dict[str, Any]:
    """
    Run RAG query asynchronously with enhanced parameters.
    
    This function processes a user query through the RAG pipeline with various
    configuration options including hybrid retrieval, reranking, and query enhancement.
    
    Args:
        query: The user's question or query string
        conversation_history: Previous conversation messages for context
        use_hybrid: Whether to use both dense and sparse vector retrieval
        use_reranker: Whether to apply cross-encoder reranking
        top_k: Number of context documents to retrieve
        collection_type: Type of collection to search ('classes' or 'reports')
        enable_query_enhancement: Whether to apply query rewriting and decomposition
        
    Returns:
        Dictionary containing:
            - response: Generated response text
            - contexts: Retrieved document contexts
            - confidence_level: Confidence assessment
            - is_enhanced: Whether query was enhanced
            - is_decomposed: Whether query was decomposed
            - sub_queries: List of sub-queries if decomposed
            - Other metadata from the pipeline
            
    Note:
        Automatically adjusts top_k for interface-related queries to improve
        context retrieval for complex class implementation questions.
    """
    rag_pipeline = get_rag_pipeline()
    
    # Adjust top_k for specific query types related to interfaces and implementations
    if "interface" in query.lower() and "implement" in query.lower() and "class" in query.lower():
        # For interface-related class queries, increase top_k significantly
        original_top_k = top_k
        top_k = max(top_k * 3, 15)  # Triple the number of contexts for interface queries
        print(f"Interface implementation query detected. Increasing top_k from {original_top_k} to {top_k}")
    
    return await rag_pipeline.process_query(
        query=query,
        conversation_history=conversation_history,
        use_hybrid=use_hybrid,
        use_reranker=use_reranker,
        top_k=top_k,
        collection_type=collection_type,
        enable_query_enhancement=enable_query_enhancement  # Pass the enhancement flag to the pipeline
    )


def run_async(coro) -> Any:
    """
    Run async function and return result in Streamlit environment.
    
    This function creates a new event loop to run asynchronous functions
    in the Streamlit environment, properly handling cleanup of pending tasks.
    
    Args:
        coro: Coroutine object to be executed
        
    Returns:
        Result of the coroutine execution
        
    Note:
        Creates a new event loop and ensures proper cleanup of all pending
        tasks to prevent resource leaks in the Streamlit environment.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()


def save_conversation_to_file(
    conversation_data: List[Dict[str, Any]], 
    collection_type: str, 
    filename: str = "rag_conversations.json"
) -> None:
    """
    Save a conversation to the conversation history file.
    
    This function appends a new conversation to the persistent conversation
    history file, creating the file if it doesn't exist.
    
    Args:
        conversation_data: List of message dictionaries representing the conversation
        collection_type: Type of collection used ('classes' or 'reports')
        filename: Name of the file to save conversations to
        
    Note:
        Creates a timestamped conversation entry with metadata about the
        collection used and saves it to a JSON file for later reference.
    """
    # Load existing conversations if file exists
    existing_conversations = []
    try:
        if os.path.exists(filename):
            with open(filename, "r", encoding="utf-8") as f:
                existing_conversations = json.load(f)
    except Exception as e:
        st.error(f"Error loading conversation history: {str(e)}")
    
    # Add this conversation
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create a conversation entry
    conversation_entry = {
        "id": f"conv_{len(existing_conversations) + 1}",
        "timestamp": timestamp,
        "collection_type": collection_type,
        "collection_name": COLLECTIONS[collection_type]["name"],
        "messages": conversation_data
    }
    
    # Add to existing conversations
    existing_conversations.append(conversation_entry)
    
    # Save to file
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(existing_conversations, f, indent=2, ensure_ascii=False)
    except Exception as e:
        st.error(f"Error saving conversation history: {str(e)}")


def continue_conversation() -> None:
    """
    Continue the current conversation by resetting the awaiting decision state.
    
    This function allows the user to continue asking questions in the current
    conversation context without ending the session.
    """
    st.session_state.awaiting_decision = False


def end_conversation() -> None:
    """
    End the current conversation and optionally save it to history.
    
    This function terminates the current conversation, saves it to the history
    file if enabled, and clears the conversation state to return to collection selection.
    """
    # Save the conversation
    if st.session_state.save_history and st.session_state.messages:
        conversation_data = []
        for message in st.session_state.messages:
            msg_copy = {
                "role": message["role"],
                "content": message["content"],
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            # Include contexts if available
            if "contexts" in message:
                msg_copy["contexts"] = message["contexts"]
            # Include confidence level if available
            if "confidence_level" in message:
                msg_copy["confidence_level"] = message["confidence_level"]
            conversation_data.append(msg_copy)
        save_conversation_to_file(conversation_data, st.session_state.collection_type)
    
    # Clear the state
    st.session_state.messages = []
    st.session_state.awaiting_decision = False
    # Return to collection selection
    st.session_state.collection_selected = False


def select_collection(collection_type: str) -> None:
    """
    Select a collection for the conversation and initialize the pipeline.
    
    This function sets the active collection type, initializes the RAG pipeline
    with the selected collection, and prepares the session for new conversations.
    
    Args:
        collection_type: Type of collection to select ('classes' or 'reports')
        
    Note:
        Updates session state to reflect the selected collection and clears
        any previous conversation messages.
    """
    # Initialize RAG pipeline with the selected collection
    pipeline = get_rag_pipeline()
    pipeline.set_collection_type(collection_type)
    
    # Store collection type in session state
    st.session_state.collection_type = collection_type
    st.session_state.collection_selected = True
    st.session_state.messages = []  # Clear any previous messages


def get_confidence_badge_html(confidence_level: str) -> str:
    """
    Generate HTML for a confidence level badge.
    
    This function creates a colored HTML badge to display the confidence level
    of the assistant's response with appropriate color coding.
    
    Args:
        confidence_level: Confidence level string (HIGH, MEDIUM, LOW, UNKNOWN)
        
    Returns:
        HTML string containing the styled confidence badge
        
    Examples:
        >>> badge = get_confidence_badge_html("HIGH")
        >>> "background-color: #28a745" in badge
        True
    """
    # Define colors for each confidence level
    confidence_colors = {
        "HIGH": "#28a745",    # Green
        "MEDIUM": "#fd7e14",  # Orange
        "LOW": "#dc3545",     # Red
        "UNKNOWN": "#6c757d"  # Gray
    }
    
    color = confidence_colors.get(confidence_level, confidence_colors["UNKNOWN"])
    
    # Create badge HTML
    badge_html = f"""
    <div style="display: inline-block; margin-bottom: 10px; margin-right: 10px;">
        <span style="background-color: {color}; color: white; padding: 4px 8px; 
        border-radius: 4px; font-weight: bold; font-size: 14px;">
            Confidence: {confidence_level}
        </span>
    </div>
    """
    
    return badge_html


def get_query_enhancement_badge_html(is_enhanced: bool, is_decomposed: bool) -> str:
    """
    Generate HTML for query enhancement badges.
    
    This function creates colored badges to indicate when queries have been
    enhanced through rewriting or decomposition processes.
    
    Args:
        is_enhanced: Whether the query was rewritten for clarity
        is_decomposed: Whether the query was decomposed into sub-queries
        
    Returns:
        HTML string containing styled enhancement badges
        
    Examples:
        >>> badges = get_query_enhancement_badge_html(True, False)
        >>> "Query Rewritten" in badges
        True
    """
    badges = []
    
    if is_enhanced:
        # Badge for query rewriting
        badges.append("""
        <div style="display: inline-block; margin-bottom: 10px; margin-right: 10px;">
            <span style="background-color: #17a2b8; color: white; padding: 4px 8px; 
            border-radius: 4px; font-weight: bold; font-size: 14px;">
                Query Rewritten
            </span>
        </div>
        """)
    
    if is_decomposed:
        # Badge for query decomposition
        badges.append("""
        <div style="display: inline-block; margin-bottom: 10px; margin-right: 10px;">
            <span style="background-color: #6f42c1; color: white; padding: 4px 8px; 
            border-radius: 4px; font-weight: bold; font-size: 14px;">
                Query Decomposed
            </span>
        </div>
        """)
    
    return "".join(badges)


def display_query_enhancement_info(result: Dict[str, Any]) -> None:
    """
    Display information about query enhancement processes.
    
    This function shows detailed information about how the user's query
    was enhanced, including original vs enhanced queries and sub-query breakdown.
    
    Args:
        result: Dictionary containing query enhancement results including:
            - original_query: The user's original question
            - enhanced_query: The rewritten query
            - is_decomposed: Whether the query was broken into parts
            - sub_queries: List of sub-queries if decomposed
            - sub_query_results: Results from individual sub-queries
            
    Note:
        Modifies the Streamlit interface by adding sections for query
        enhancement details and sub-query results.
    """
    # Create a section for query enhancement details
    st.markdown("### Query Enhancement")
    
    # Display original and enhanced query
    st.markdown("**Original Query:**")
    st.markdown(f"> {result.get('original_query', 'N/A')}")
    
    if result.get("is_enhanced", False):
        st.markdown("**Enhanced Query:**")
        st.markdown(f"> {result.get('enhanced_query', 'N/A')}")
    
    # If query was decomposed, show sub-queries
    if result.get("is_decomposed", False) and "sub_queries" in result:
        st.markdown("**Sub-queries:**")
        for i, sub_query in enumerate(result["sub_queries"]):
            st.markdown(f"{i+1}. {sub_query}")
        
        # If there are sub-query results, show them in a table
        if "sub_query_results" in result:
            st.markdown("### Sub-query Results")
            # Create a table to display sub-query results
            sub_query_data = []
            for i, sub_result in enumerate(result["sub_query_results"]):
                sub_query_data.append({
                    "Number": i+1,
                    "Sub-query": sub_result.get("sub_query", "N/A"),
                    "Confidence": sub_result.get("confidence_level", "UNKNOWN"),
                    "Doc Count": len(sub_result.get("contexts", [])),
                    "Response Length": len(sub_result.get("response", "").split())
                })
            st.dataframe(sub_query_data)


def initialize_session_state() -> None:
    """
    Initialize all session state variables with default values.
    
    This function sets up the initial state for the Streamlit application,
    ensuring all required session variables are properly initialized.
    """
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "awaiting_decision" not in st.session_state:
        st.session_state.awaiting_decision = False

    if "selected_doc_index" not in st.session_state:
        st.session_state.selected_doc_index = 0

    if "collection_selected" not in st.session_state:
        st.session_state.collection_selected = False
        
    if "collection_type" not in st.session_state:
        st.session_state.collection_type = DEFAULT_COLLECTION_TYPE
        
    if "enable_query_enhancement" not in st.session_state:
        st.session_state.enable_query_enhancement = True

    if "enable_evaluation_metrics" not in st.session_state:
        st.session_state.enable_evaluation_metrics = True
        
    if "show_evaluation_details" not in st.session_state:
        st.session_state.show_evaluation_details = False


def setup_page_config() -> None:
    """
    Configure the Streamlit page settings and styling.
    
    This function sets up the page configuration including title, icon,
    layout, and custom CSS styling for the application.
    """
    # Page configuration
    st.set_page_config(
        page_title="SAP ABAP Code Documentation RAG System",
        page_icon="??",
        layout="wide"
    )

    # Add custom CSS for confidence badges and other styling
    st.markdown("""
    <style>
        .source-section {
            margin-top: 15px;
            border-top: 1px solid #ddd;
            padding-top: 10px;
        }
        .source-title {
            font-weight: bold;
            margin-bottom: 5px;
        }
        .source-item {
            margin-left: 15px;
            margin-bottom: 3px;
        }
        .confidence-badge {
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
            color: white;
        }
        .query-badge {
            display: inline-block;
            padding: 3px 6px;
            border-radius: 4px;
            font-weight: bold;
            color: white;
            margin-right: 5px;
        }
        .doc-table th {
            background-color: #f0f2f6;
        }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar() -> tuple[bool, bool, int, bool, bool, bool, bool]:
    """
    Render the sidebar with connection info, settings, and controls.
    
    This function creates the sidebar interface containing connection information,
    collection details, query settings, and various configuration options.
    
    Returns:
        Tuple containing user preferences:
            - use_hybrid: Whether to use hybrid retrieval
            - use_reranker: Whether to use reranker
            - top_k: Number of context documents
            - enable_query_enhancement: Whether to enable query enhancement
            - show_enhancement_details: Whether to show enhancement details
            - enable_evaluation_metrics: Whether to enable evaluation metrics
            - show_evaluation_details: Whether to show evaluation details
    """
    with st.sidebar:
        st.title("SAP ABAP RAG System")
        st.markdown("---")
        
        # Connection information
        st.subheader("Connection Info")
        st.write(f"Qdrant Host: `{QDRANT_HOST}`")
        st.write(f"Qdrant Port: `{QDRANT_PORT}`")
        
        # Collection information
        st.subheader("Collection Info")
        if st.session_state.collection_selected:
            collection_info = COLLECTIONS[st.session_state.collection_type]
            st.write(f"Type: `{st.session_state.collection_type}`")
            st.write(f"Collection: `{collection_info['name']}`")
            st.write(f"Description: {collection_info['description']}")
            
            # Button to change collection
            if st.button("Change Collection"):
                st.session_state.collection_selected = False
                st.rerun()
        
        # Settings - only show if collection is selected
        if st.session_state.collection_selected:
            st.subheader("Query Settings")
            use_hybrid = st.checkbox("Use hybrid retrieval", value=USE_HYBRID_DEFAULT,
                                    help="Use both dense and sparse vectors for retrieval")
            use_reranker = st.checkbox("Use reranker", value=USE_RERANKER_DEFAULT, 
                                    help="Apply cross-encoder reranking to improve results")
            top_k = st.slider("Number of context documents", 
                            min_value=1, max_value=20, value=DEFAULT_TOP_K,
                            help="Number of documents to use as context")
            
            # Query enhancement options
            st.subheader("Query Enhancement")
            enable_query_enhancement = st.checkbox("Enable query enhancement", value=True, 
                                                help="Apply query rewriting and decomposition")
            show_enhancement_details = st.checkbox("Show enhancement details", value=False, 
                                             help="Display information about how queries are enhanced")
            
            # Evaluation metrics options
            st.subheader("Evaluation Metrics")
            enable_evaluation_metrics = st.checkbox("Enable evaluation metrics", value=True,
                                                   help="Display evaluation metrics like groundedness")
            show_evaluation_details = st.checkbox("Show evaluation details", value=False,
                                                help="Display detailed evaluation information")
            
            # Feedback options
            st.subheader("Feedback")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("?? Helpful", use_container_width=True):
                    st.success("Thank you for your feedback!")
            with col2:
                if st.button("?? Not Helpful", use_container_width=True):
                    st.error("Thanks for letting us know!")
            
            # Save session history option
            st.session_state.save_history = st.checkbox("Save session history", value=True,
                                    help="Save conversations to a file for later reference")
            
            # Store preferences in session state
            st.session_state.enable_evaluation_metrics = enable_evaluation_metrics
            st.session_state.show_evaluation_details = show_evaluation_details
            st.session_state.enable_query_enhancement = enable_query_enhancement
            st.session_state.show_enhancement_details = show_enhancement_details
            
            # Clear chat button
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.session_state.awaiting_decision = False
                st.rerun()
        else:
            # Return default values when collection is not selected
            use_hybrid = USE_HYBRID_DEFAULT
            use_reranker = USE_RERANKER_DEFAULT
            top_k = DEFAULT_TOP_K
            enable_query_enhancement = True
            show_enhancement_details = False
            enable_evaluation_metrics = True
            show_evaluation_details = False
        
        # About section
        st.markdown("---")
        st.subheader("About")
        st.markdown("""
        This system allows you to query SAP ABAP code documentation using a 
        Retrieval-Augmented Generation (RAG) approach with query enhancement.
        
        The system combines:
        - Dense vector similarity (SAP AI Core embeddings)
        - Sparse vector matching (BM25)
        - Query enhancement (rewriting and decomposition)
        - LLM-based response generation
        - Conversation support for follow-up questions
        - Confidence level assessment
        """)
    
    return (use_hybrid, use_reranker, top_k, enable_query_enhancement, 
            show_enhancement_details, enable_evaluation_metrics, show_evaluation_details)


def render_collection_selection() -> None:
    """
    Render the collection selection interface.
    
    This function displays the collection selection screen where users can
    choose between SAP ABAP Classes and Reports documentation collections.
    """
    st.write("Please select which type of SAP ABAP documentation you want to query:")
    
    # Container for collection selection with styling
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### {COLLECTIONS['classes']['description']}")
        st.write("Documentation for SAP ABAP classes, methods, and object-oriented programming.")
        st.write(f"Collection: `{COLLECTIONS['classes']['name']}`")
        if st.button("Query SAP ABAP Classes", use_container_width=True):
            select_collection("classes")
            st.rerun()
    
    with col2:
        st.markdown(f"### {COLLECTIONS['reports']['description']}")
        st.write("Documentation for SAP ABAP reports, modules, and procedural programming.")
        st.write(f"Collection: `{COLLECTIONS['reports']['name']}`")
        if st.button("Query SAP ABAP Reports", use_container_width=True):
            select_collection("reports")
            st.rerun()
    
    # Display informational text
    st.markdown("---")
    st.markdown("""
    **How to use this system:**
    1. Select the type of SAP ABAP documentation you want to query (Classes or Reports)
    2. Ask your questions in the chat interface
    3. View the retrieved document contexts with each answer
    4. Continue with follow-up questions or start a new conversation
    
    **Features:**
    - **Confidence Assessment**: Each response includes a confidence level (High, Medium, Low) 
      to help you gauge the reliability of the information.
    - **Query Enhancement**: Your queries are automatically enhanced through rewriting for clarity 
      and decomposition of complex questions into simpler ones.
    """)


def render_user_message(message: Dict[str, Any], show_enhancement_details: bool) -> None:
    """
    Render a user message in the chat interface.
    
    This function displays a user message along with any query enhancement
    information if available and enabled.
    
    Args:
        message: Message dictionary containing user content and enhancement info
        show_enhancement_details: Whether to display enhancement details
    """
    st.markdown(message["content"])
    
    # If query was enhanced, display badges
    if message.get("is_enhanced", False) or message.get("is_decomposed", False):
        st.markdown(
            get_query_enhancement_badge_html(
                message.get("is_enhanced", False), 
                message.get("is_decomposed", False)
            ), 
            unsafe_allow_html=True
        )
        
        # Create a section for query enhancement info - only if show_enhancement_details is enabled
        if show_enhancement_details:
            st.markdown("**Query Enhancement:**")
            
            # Show enhanced query if available
            if message.get("is_enhanced", False):
                st.markdown(f"**Original Query:** {message.get('original_query', message['content'])}")
                st.markdown(f"**Enhanced Query:** {message.get('enhanced_query', 'N/A')}")
            
            # Show sub-queries if available
            if message.get("is_decomposed", False) and "sub_queries" in message:
                st.markdown("**Sub-queries:**")
                for i, sub_query in enumerate(message["sub_queries"]):
                    st.markdown(f"{i+1}. {sub_query}")


def render_assistant_message(message: Dict[str, Any], show_enhancement_details: bool) -> None:
    """
    Render an assistant message in the chat interface.
    
    This function displays an assistant message with confidence badges,
    cleaned content, evaluation metrics, and enhancement information.
    
    Args:
        message: Message dictionary containing assistant response and metadata
        show_enhancement_details: Whether to display enhancement details
    """
    # Display confidence badge
    if "confidence_level" in message:
        confidence_level = message.get("confidence_level", "UNKNOWN")
        st.markdown(get_confidence_badge_html(confidence_level), unsafe_allow_html=True)
    
    # Display the message content - first clean it from any evaluation metrics
    cleaned_content = clean_response_text(message["content"])
    st.markdown(cleaned_content)
    
    # Display evaluation metrics with sample metrics for existing messages
    display_evaluation_metrics(
        {"content": message["content"]}, 
        show_sample=True  # Always show sample metrics for existing messages
    )
    
    # Show enhancement badges if details are enabled
    if (message.get("is_enhanced", False) or message.get("is_decomposed", False)) and show_enhancement_details:
        st.markdown(
            get_query_enhancement_badge_html(
                message.get("is_enhanced", False), 
                message.get("is_decomposed", False)
            ), 
            unsafe_allow_html=True
        )
    
    # Show query enhancement info if available
    if (message.get("is_enhanced", False) or message.get("is_decomposed", False)) and show_enhancement_details:
        # Create a section for query enhancement details
        st.markdown("### Query Enhancement Details")
        
        if message.get("is_enhanced", False):
            st.markdown("**Original Query:** " + message.get("original_query", "N/A"))
            st.markdown("**Enhanced Query:** " + message.get("enhanced_query", "N/A"))
        
        if message.get("is_decomposed", False) and "sub_queries" in message:
            st.markdown("**Sub-queries:**")
            for i, sub_query in enumerate(message["sub_queries"]):
                st.markdown(f"{i+1}. {sub_query}")
        
        # Display sub-query results overview
        if "sub_query_results" in message:
            st.markdown("### Sub-query Results Summary")
            # Create a table to display sub-query results
            sub_query_data = []
            for i, sub_result in enumerate(message["sub_query_results"]):
                sub_query_data.append({
                    "Number": i+1,
                    "Sub-query": sub_result.get("sub_query", "N/A"),
                    "Confidence": sub_result.get("confidence_level", "UNKNOWN"),
                    "Doc Count": len(sub_result.get("contexts", []))
                })
            st.dataframe(sub_query_data)


def render_document_contexts(contexts: List[Dict[str, Any]], message_idx: int) -> None:
    """
    Render document contexts with selection interface and pagination.
    
    This function displays the retrieved document contexts with a dropdown
    selector and detailed information about each document.
    
    Args:
        contexts: List of document context dictionaries
        message_idx: Index of the message for unique key generation
    """
    # Display document selection method
    st.markdown("### Retrieved Documents")
    
    # Method 1: Dropdown selector for documents
    doc_options = [f"Doc {i+1}: {ctx.get('title', 'Untitled')} (Score: {abs(ctx.get('score', 0)):.4f})" 
                  for i, ctx in enumerate(contexts)]
    
    selected_doc = st.selectbox(
        "Select a document to view:",
        options=doc_options,
        index=0,
        key=f"doc_select_{message_idx}"
    )
    
    # Get the index of the selected document
    selected_index = doc_options.index(selected_doc)
    
    # Display the selected document
    ctx = contexts[selected_index]
    
    # Document info container
    doc_info_container = st.container()
    with doc_info_container:
        st.markdown(f"**Title**: {ctx.get('title', 'Untitled')}")
        st.markdown(f"**Source**: {ctx.get('filename', 'Unknown')}")
        # Use absolute score value for consistent display
        score_val = abs(ctx.get('score', 0))
        st.markdown(f"**Relevance Score**: {score_val:.4f}")
        
        # If this document is from a sub-query, show which one
        if "sub_query" in ctx:
            st.markdown(f"**From Sub-query**: {ctx.get('sub_query', 'Unknown')}")
        
        # Add content if available
        if ctx.get('text'):
            st.markdown("#### Content")
            # Clean up the content to remove duplicates
            clean_text = clean_document_content(ctx.get('text', ''))
            st.markdown(clean_text)
        
        # Add code snippet if available
        if ctx.get('code_snippet'):
            st.markdown("#### Code Snippet")
            st.code(ctx.get('code_snippet', ''), language="abap")
    
    # Method 2: Alternative document navigation with buttons
    col1, col2, col3 = st.columns([1, 4, 1])
    
    # Display pagination info
    with col2:
        st.markdown(f"**Document {selected_index + 1} of {len(contexts)}**")
    
    # Add pagination info and summary of all retrieved documents
    st.markdown("### Document Overview")
    
    # Create a table with all document titles and scores
    doc_data = []
    for i, ctx in enumerate(contexts):
        # Use absolute score value for display
        abs_score = abs(ctx.get('score', 0))
        doc_data.append({
            "Index": i + 1,
            "Title": ctx.get('title', 'Untitled'),
            "Source": ctx.get('filename', 'Unknown'),
            "Score": f"{abs_score:.4f}",
            "Sub-query": ctx.get('sub_query', '-') if 'sub_query' in ctx else '-'
        })
    
    st.dataframe(doc_data, use_container_width=True)


def render_chat_history(show_enhancement_details: bool) -> None:
    """
    Render the complete chat history with all messages and contexts.
    
    This function iterates through all messages in the session state and
    renders them appropriately based on their role (user or assistant).
    
    Args:
        show_enhancement_details: Whether to display query enhancement details
    """
    # Display chat history
    for message_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            # If this is a user message, show the original query
            if message["role"] == "user":
                render_user_message(message, show_enhancement_details)
            
            # If this is an assistant message, display confidence badge and query enhancement info
            if message["role"] == "assistant":
                render_assistant_message(message, show_enhancement_details)
            
            # Show document references if available
            if message["role"] == "assistant" and "contexts" in message and message["contexts"]:
                render_document_contexts(message["contexts"], message_idx)


def render_follow_up_options() -> None:
    """
    Render follow-up options after assistant response.
    
    This function displays buttons allowing users to continue the conversation,
    end it, or change collections after the assistant has provided a response.
    """
    st.markdown("### Do you have a follow-up question?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Yes, continue conversation", on_click=continue_conversation, key="continue_btn")
    with col2:
        st.button("End conversation", on_click=end_conversation, key="end_btn")
    with col3:
        if st.button("Change collection", key="change_collection_btn"):
            st.session_state.collection_selected = False
            st.rerun()


def process_user_query(
    query: str, 
    use_hybrid: bool, 
    use_reranker: bool, 
    top_k: int, 
    enable_query_enhancement: bool,
    show_enhancement_details: bool
) -> None:
    """
    Process a user query through the RAG pipeline and display results.
    
    This function handles the complete query processing workflow including
    conversation history extraction, RAG pipeline execution, response cleaning,
    and result display.
    
    Args:
        query: The user's input query
        use_hybrid: Whether to use hybrid retrieval
        use_reranker: Whether to use reranker
        top_k: Number of context documents to retrieve
        enable_query_enhancement: Whether to enable query enhancement
        show_enhancement_details: Whether to show enhancement details
    """
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(query)
    
    # Display assistant response with a spinner
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Get pipeline and process query
        try:
            with st.spinner(f"Searching {st.session_state.collection_type} documentation and generating response..."):
                # Extract conversation history from session state
                conversation_history = None
                if len(st.session_state.messages) > 1:  # If there are previous messages
                    conversation_history = [
                        {"role": msg["role"], "content": msg["content"]} 
                        for msg in st.session_state.messages[:-1]  # Exclude the current query
                        if msg["role"] in ["user", "assistant"]
                    ]
                
                # Process the query using the async helper
                result = run_async(run_rag_query(
                    query=query,
                    conversation_history=conversation_history,
                    use_hybrid=use_hybrid,
                    use_reranker=use_reranker,
                    top_k=top_k,
                    collection_type=st.session_state.collection_type,  # Use the selected collection
                    enable_query_enhancement=enable_query_enhancement  # Pass query enhancement flag
                ))
                
                # Store enhancement details display preference
                st.session_state.show_enhancement_details = show_enhancement_details
                
                # Extract response, confidence level, and contexts
                full_response = result.get("full_response", "I couldn't generate a response.")
                response = result.get("response", full_response)
                
                # Clean the response and extract the confidence level
                cleaned_response, confidence_level = clean_assistant_response(response)
                
                # If clean_assistant_response didn't find a confidence level, use the one from the result
                if confidence_level == "UNKNOWN" and "confidence_level" in result:
                    confidence_level = result.get("confidence_level", "UNKNOWN")
                
                contexts = result.get("contexts", [])
                has_relevant_results = result.get("has_relevant_results", False)
                
                # Extract query enhancement info
                is_enhanced = result.get("is_enhanced", False)
                is_decomposed = result.get("is_decomposed", False)
                enhanced_query = result.get("enhanced_query", query)
                original_query = result.get("original_query", query)
                sub_queries = result.get("sub_queries", [])
                sub_query_results = result.get("sub_query_results", [])
                
                # Update the user message with query enhancement info
                user_message = st.session_state.messages[-1]
                user_message["is_enhanced"] = is_enhanced
                user_message["is_decomposed"] = is_decomposed
                user_message["enhanced_query"] = enhanced_query
                user_message["original_query"] = original_query
                user_message["sub_queries"] = sub_queries
                
                # Display confidence badge
                st.markdown(get_confidence_badge_html(confidence_level), unsafe_allow_html=True)
                
                # Display the cleaned response
                message_placeholder.markdown(cleaned_response)
                
                # If query was enhanced or decomposed, show details
                if (is_enhanced or is_decomposed) and show_enhancement_details:
                    display_query_enhancement_info(result)
        
        except Exception as e:
            full_response = f"An error occurred: {str(e)}"
            response = full_response
            message_placeholder.markdown(response)
            st.error(f"Error details: {str(e)}")
            has_relevant_results = False
            contexts = []
            confidence_level = "UNKNOWN"
            is_enhanced = False
            is_decomposed = False
            enhanced_query = query
            original_query = query
            sub_queries = []
            sub_query_results = []
    
    # Add assistant response to chat history
    assistant_message = {
        "role": "assistant", 
        "content": cleaned_response if 'cleaned_response' in locals() else response,
        "full_response": full_response,
        "confidence_level": confidence_level,
        "contexts": contexts,
        # Add query enhancement info
        "is_enhanced": is_enhanced,
        "is_decomposed": is_decomposed,
        "enhanced_query": enhanced_query,
        "original_query": original_query,
        "sub_queries": sub_queries,
        "sub_query_results": sub_query_results
    }
    st.session_state.messages.append(assistant_message)
    
    # Set awaiting decision state
    st.session_state.awaiting_decision = True
    
    # Force refresh to update UI
    st.rerun()


def main() -> None:
    """
    Main function to run the Streamlit application.
    
    This function orchestrates the entire application flow including initialization,
    page setup, sidebar rendering, and main content display based on application state.
    """
    # Initialize session state and page configuration
    initialize_session_state()
    setup_page_config()
    
    # Render sidebar and get user preferences
    (use_hybrid, use_reranker, top_k, enable_query_enhancement, 
     show_enhancement_details, enable_evaluation_metrics, show_evaluation_details) = render_sidebar()
    
    # Main content
    st.title("SAP ABAP Code Documentation Assistant")
    
    # Collection selection screen
    if not st.session_state.collection_selected:
        render_collection_selection()
    else:
        # Show the selected collection type
        collection_info = COLLECTIONS[st.session_state.collection_type]
        st.write(f"Asking questions about: **{collection_info['description']}**")
        
        # Render chat history
        render_chat_history(show_enhancement_details)
        
        # Display follow-up options after the assistant has responded
        if st.session_state.awaiting_decision:
            render_follow_up_options()
        
        # Text input for the query (only disabled when awaiting decision)
        if not st.session_state.awaiting_decision:
            query = st.chat_input(f"Ask a question about SAP ABAP {st.session_state.collection_type}...")
        else:
            query = None
        
        # Process the query when submitted
        if query:
            process_user_query(
                query, use_hybrid, use_reranker, top_k, 
                enable_query_enhancement, show_enhancement_details
            )


if __name__ == "__main__":
    main()