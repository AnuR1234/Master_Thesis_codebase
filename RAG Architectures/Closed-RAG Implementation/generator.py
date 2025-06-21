#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Amazon Bedrock Claude Generator Module for RAG Pipeline.

This module provides a sophisticated response generation component that integrates
with Amazon Bedrock's Claude models to generate high-quality, contextually aware
responses for SAP ABAP documentation queries. The generator supports conversation
history, confidence level assessment, and specialized handling for different query types.

Key Features:
    - Amazon Bedrock Claude integration with native client support
    - Conversation history management for multi-turn interactions
    - Confidence level assessment for response reliability
    - Specialized query analysis and response formatting
    - Out-of-scope query handling with appropriate fallbacks
    - Robust error handling with multiple API call strategies
    - Context formatting optimized for ABAP documentation

The module implements advanced prompt engineering techniques including:
    - Dynamic system prompt enhancement based on query type
    - Query-specific instruction generation for focused responses
    - Context prioritization for method and interface queries
    - Conversational tone management for natural interactions

Architecture:
    The generator follows a layered architecture with separation of concerns:
    - RAGGenerator: Main class coordinating all generation operations
    - Bedrock Integration: Native Amazon Bedrock client management
    - Prompt Engineering: Dynamic prompt construction and enhancement
    - Response Processing: Text extraction and formatting
    - Error Handling: Multiple fallback strategies for reliability

Classes:
    RAGGenerator: Main generator class with comprehensive response generation capabilities

Dependencies:
    - gen_ai_hub.proxy.native.amazon.clients: Amazon Bedrock native client
    - config: System prompts and model configuration
    - dotenv: Environment variable management for credentials

Author: SAP ABAP RAG Team
Version: 1.0.0
Date: 2025
License: MIT
"""

# Standard library imports
import asyncio
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

# Third-party imports
import nest_asyncio
from dotenv import load_dotenv

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
RAG_ENV_PATH = "/home/user/Desktop/RAG_pipeline_enhanced_conversational_claude/claude.env"
load_dotenv(RAG_ENV_PATH)

# Import Amazon Bedrock client with graceful fallback
try:
    from gen_ai_hub.proxy.native.amazon.clients import Session
except ImportError:
    logging.error("Could not import Session from gen_ai_hub.proxy.native.amazon.clients")
    Session = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import system prompts and configuration from config module
from config import (
    BEDROCK_MAX_TOKENS,
    BEDROCK_MODEL,
    BEDROCK_TEMPERATURE,
    BEDROCK_TOP_P,
    CONFIDENCE_ASSESSMENT_INSTRUCTIONS,
    CONVERSATION_SYSTEM_PROMPT,
    DEFAULT_SYSTEM_PROMPT,
    OUT_OF_SCOPE_CONVERSATION_PROMPT,
    OUT_OF_SCOPE_PROMPT,
)


class RAGGenerator:
    """
    Generator component for the RAG pipeline with conversation support and confidence assessment.
    
    This class provides sophisticated response generation capabilities using Amazon Bedrock's
    Claude models. It implements advanced prompt engineering techniques, conversation history
    management, and specialized handling for different types of SAP ABAP documentation queries.
    
    The generator is designed to produce high-quality, contextually aware responses that:
    - Maintain conversation context across multiple turns
    - Provide confidence level assessments for reliability
    - Handle specialized ABAP queries (methods, interfaces, classes)
    - Generate appropriate responses for out-of-scope queries
    - Use conversational tone while maintaining technical accuracy
    
    Attributes:
        bedrock_session: Amazon Bedrock session for API access
        client: Bedrock client configured for Claude model
        model_name (str): Name of the Claude model being used
        
    Examples:
        >>> generator = RAGGenerator()
        >>> response = await generator.generate(
        ...     "How do ABAP classes work?",
        ...     context="ABAP classes are...",
        ...     system_prompt="You are an ABAP expert..."
        ... )
        >>> print(response)
        
        >>> # With conversation history
        >>> history = [{"role": "user", "content": "What is ABAP?"}]
        >>> response = await generator.generate_with_history(
        ...     "How do classes work in it?",
        ...     context="Class documentation...",
        ...     conversation_history=history
        ... )
    """
    
    def __init__(self) -> None:
        """
        Initialize the generator with Amazon Bedrock Claude integration.
        
        Sets up the Amazon Bedrock session and client for Claude model access.
        Performs necessary validation to ensure all required dependencies and
        configurations are available.
        
        Raises:
            ImportError: If gen_ai_hub package is not available
            Exception: If Bedrock session or client creation fails
            
        Note:
            Requires proper Amazon Bedrock credentials and permissions to be
            configured in the environment. The Session class from gen_ai_hub
            handles authentication automatically using environment credentials.
        """
        logger.info("Initializing RAG Generator with Amazon Bedrock Claude integration")
        
        # Check if Bedrock Session is available
        if Session is None:
            logger.error("Amazon Bedrock Session client is not available. Please install gen_ai_hub package.")
            raise ImportError("Missing required dependency: gen_ai_hub.proxy.native.amazon.clients")
        
        # Create Bedrock session
        try:
            self.bedrock_session = Session()
            self.model_name = BEDROCK_MODEL
            logger.info(f"Created Bedrock session for model: {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to create Bedrock session: {e}")
            raise
            
        # Create Bedrock client for Claude
        try:
            self.client = self.bedrock_session.client(model_name=self.model_name)
            logger.info(f"Successfully created Bedrock client for {self.model_name}")
        except Exception as e:
            logger.error(f"Failed to create Bedrock client: {e}")
            raise
    
    def format_context(self, results: List[Any]) -> str:
        """
        Format retrieval results into context for the LLM with better structure and clarity.
        
        This method processes retrieval results to create well-structured context that
        optimizes LLM performance. It implements intelligent prioritization by identifying
        query-specific contexts and organizing them for maximum relevance.
        
        Args:
            results: List of retrieval results from the vector database. Each result
                    can be either a tuple (result_obj, score) or a plain result object
                    with embedded score information.
                    
        Returns:
            Formatted context string with prioritized document organization, including:
            - Document numbering for reference consistency
            - Title and source information for context
            - Code snippets in ABAP format when available
            - Relevance scores for transparency
            - Special instructions for method-specific queries
            
        Examples:
            >>> generator = RAGGenerator()
            >>> results = [result1, result2, result3]
            >>> context = generator.format_context(results)
            >>> print(context)
            # [Document 1] Title: ABAP Class Implementation
            # Source: class_guide.md
            # Content: ABAP classes provide...
            # Relevance Score: 0.8547
            
        Note:
            The method implements intelligent context prioritization by:
            - Identifying method and interface patterns in queries
            - Placing query-specific contexts first
            - Adding focused instructions for method queries
            - Cleaning up text formatting for better readability
        """
        context_parts = []
        
        # First, check if we have any retrieved results that mention the specific method or interface in the query
        query_specific_contexts = []
        general_contexts = []
        
        # Extract method or interface pattern from results if present
        method_pattern = r'(\S+)~(\S+)'
        interface_pattern = r'(/\w+/if_\w+|if_\w+)'
        
        method_match = None
        interface_name = None
        
        # Look through all payloads to extract method and interface patterns
        for i, result in enumerate(results):
            # Handle tuple or plain result
            if isinstance(result, tuple):
                result_obj, score = result
            else:
                result_obj = result
                score = result.score
                
            # Extract fields from payload
            payload = result_obj.payload
            text = payload.get("text", "")
            
            # Check for method pattern in the text
            method_matches = re.findall(method_pattern, text)
            if method_matches:
                for m in method_matches:
                    if len(m) == 2:  # Tuple with interface and method name
                        method_match = m
                        interface_name = m[0]
                        break
                
            # If we found a method, break
            if method_match:
                break
                
            # Check for interface pattern if method wasn't found
            if not interface_name:
                interface_matches = re.findall(interface_pattern, text)
                if interface_matches:
                    interface_name = interface_matches[0]
                    break
        
        # Now process each result and categorize
        for i, result in enumerate(results):
            # Handle tuple or plain result
            if isinstance(result, tuple):
                result_obj, score = result
            else:
                result_obj = result
                score = result.score
            
            # Extract fields from payload
            payload = result_obj.payload
            text = payload.get("text", "")
            title = payload.get("title", "")
            code_snippet = payload.get("code_snippet", "")
            filename = payload.get("filename", "")
            
            # Check if this context contains the specific method or interface we're looking for
            is_specific_to_query = False
            if method_match and f"{method_match[0]}~{method_match[1]}" in text:
                is_specific_to_query = True
            elif interface_name and interface_name in text:
                is_specific_to_query = True
                
            # Format context
            context_part = f"[Document {i+1}] "
            if title:
                context_part += f"Title: {title}\n"
            if filename:
                context_part += f"Source: {filename}\n"
            if code_snippet:
                context_part += f"Code: ```abap\n{code_snippet}\n```\n"
            
            # Clean up text - remove excessive whitespace
            cleaned_text = re.sub(r'\n\s*\n+', '\n\n', text)
            
            # Add content
            context_part += f"Content: {cleaned_text}\n"
            
            # Add relevance score
            context_part += f"Relevance Score: {score:.4f}\n\n"
            
            # Categorize based on specificity
            if is_specific_to_query:
                query_specific_contexts.append(context_part)
            else:
                general_contexts.append(context_part)
        
        # First add the query-specific contexts, then the general ones
        context_parts = query_specific_contexts + general_contexts
        
        # If we have a method, add a clear instruction
        if method_match:
            method_name = f"{method_match[0]}~{method_match[1]}"
            instruction = f"""
    FOCUS INSTRUCTION: The user is asking about the method {method_name}. 
    Prioritize information from documents that specifically mention this exact method.
    Answer ONLY what was asked about this method - do not add extra information.
    Start with a direct, conversational answer.
    """
            return instruction + "\n" + "\n".join(context_parts)
        
        return "\n".join(context_parts)
    
    async def _generate_with_client(self, messages: List[Dict[str, str]], system_message: Optional[str] = None) -> str:
        """
        Generate response using Claude via Amazon Bedrock with comprehensive error handling.
        
        This is the core method that handles communication with Amazon Bedrock's Claude API.
        It implements multiple fallback strategies to ensure reliable operation even when
        specific API formats encounter issues.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     representing the conversation context for Claude
            system_message: Optional system prompt to guide Claude's behavior and response style
            
        Returns:
            Generated response text from Claude, or an error message if generation fails
            
        Raises:
            Exception: Re-raises API exceptions after attempting all fallback strategies
            
        Note:
            Implements multiple API call strategies:
            1. Primary converse API with system message
            2. Alternative invoke method with raw JSON
            3. Fallback converse without system message
            
            The method handles response extraction from various possible response formats
            to ensure compatibility with different Bedrock API versions.
        """
        try:
            logger.info(f"Generating response using Claude with model {self.model_name}")
            
            # Convert messages to Claude's expected format
            claude_messages = []
            
            # Add system message if provided - format as a list with a dict containing text
            if system_message:
                # Format the system message correctly for Claude API
                # Claude expects system to be formatted as a dict with 'text' key in a list
                system_content = [{"text": system_message}]
            else:
                # Default system message if none provided
                system_content = [{"text": DEFAULT_SYSTEM_PROMPT}]
            
            # Add each message to the messages array
            for message in messages:
                role = message["role"]
                content = message["content"]
                
                if role in ["user", "assistant"]:
                    claude_messages.append({
                        "role": role,
                        "content": [{"text": content}]
                    })
            
            # Run the Bedrock call in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: self._call_bedrock_converse(claude_messages, system_content)
            )
            
            # Log the full response structure to understand the format better
            response_str = json.dumps(response, indent=2) if response else "None"
            logger.info(f"Raw Bedrock response structure: {response_str[:1000]}...")
            
            # Extract text from the response - handle different possible formats
            response_text = self._extract_response_text(response)
            if response_text:
                return response_text
            else:
                logger.error("Could not extract text from Claude response")
                return "I couldn't generate a response due to an unexpected format."
                
        except Exception as e:
            logger.error(f"Error with Bedrock API call: {e}")
            # Return an error message that the user will see
            return f"Error generating response: {str(e)}"
    
    def _extract_response_text(self, response: Any) -> Optional[str]:
        """
        Extract text content from Claude response, handling different possible formats.
        
        This method implements robust response parsing to handle various API response
        formats that may be returned by different versions of the Bedrock Claude API.
        
        Args:
            response: Raw response object from Bedrock API, which can be in various formats
                     depending on the API version and call method used
                     
        Returns:
            Extracted response text if successful, None if extraction fails
            
        Note:
            Handles multiple response formats including:
            - Bedrock converse API format with output.message.content structure
            - Direct content list format with text objects
            - Alternative message formats with nested content
            - Fallback string conversion for unknown formats
            
            This robust parsing ensures compatibility across different API versions
            and deployment configurations.
        """
        if not response:
            return None
        
        try:
            # Handle Bedrock format (primary format)
            if "output" in response and "message" in response["output"]:
                message = response["output"]["message"]
                if "content" in message and isinstance(message["content"], list):
                    for content_item in message["content"]:
                        if isinstance(content_item, dict) and "text" in content_item:
                            return content_item["text"]
            
            # Try the expected format for other Claude APIs
            if "content" in response and isinstance(response["content"], list):
                for content_item in response["content"]:
                    if isinstance(content_item, dict) and "text" in content_item:
                        return content_item["text"]
            
            # If that doesn't work, try other possible formats
            if "message" in response:
                message = response["message"]
                if isinstance(message, dict) and "content" in message:
                    content = message["content"]
                    if isinstance(content, list):
                        for content_item in content:
                            if isinstance(content_item, dict) and "text" in content_item:
                                return content_item["text"]
                    elif isinstance(content, str):
                        return content
            
            # Try direct content format
            if "text" in response:
                return response["text"]
            
            # Try alternative formats
            if "completion" in response:
                return response["completion"]
                
            if isinstance(response, str):
                return response
                
            # If all else fails, convert to string
            return str(response)
        except Exception as e:
            logger.error(f"Error extracting response text: {e}")
            # Try one last approach - stringify the entire response
            try:
                if isinstance(response, dict):
                    return json.dumps(response)
                return str(response)
            except:
                return None
    
    def _call_bedrock_converse(self, messages: List[Dict], system_message: List[Dict]) -> Dict[str, Any]:
        """
        Make synchronous call to Bedrock converse API with multiple fallback strategies.
        
        This method handles the actual API communication with Amazon Bedrock, implementing
        multiple call strategies to ensure reliable operation across different API versions
        and configurations.
        
        Args:
            messages: List of formatted message dictionaries for Claude
            system_message: List containing system message dictionary with 'text' key
            
        Returns:
            Raw API response from Bedrock
            
        Raises:
            Exception: If all API call strategies fail
            
        Note:
            Implements three fallback strategies:
            1. Primary converse API with full configuration
            2. Alternative invoke method with raw JSON body
            3. Fallback converse without system message
            
            Logs detailed request information for debugging API issues.
        """
        try:
            # Log the exact request being sent
            request_debug = {
                "messages": messages,
                "system": system_message,
                "inferenceConfig": {
                    "maxTokens": BEDROCK_MAX_TOKENS,
                    "temperature": BEDROCK_TEMPERATURE,
                    "topP": BEDROCK_TOP_P
                }
            }
            logger.info(f"Sending request to Claude: {json.dumps(request_debug, indent=2)}")
            
            # Make API call with properly formatted system message
            # system_message is now a list containing a dict with 'text' key
            try:
                response = self.client.converse(
                    messages=messages,
                    system=system_message,
                    inferenceConfig={
                        "maxTokens": BEDROCK_MAX_TOKENS,
                        "temperature": BEDROCK_TEMPERATURE,
                        "topP": BEDROCK_TOP_P
                    }
                )
                return response
            except Exception as api_error:
                logger.error(f"API error: {api_error}")
                # If the API call fails with system message, try alternative format
                logger.info("Trying alternative API call format...")
                try:
                    # Try direct invoke method instead of converse
                    response = self.client.invoke(
                        body=json.dumps({
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": BEDROCK_MAX_TOKENS,
                            "temperature": BEDROCK_TEMPERATURE,
                            "top_p": BEDROCK_TOP_P,
                            "messages": messages,
                            "system": system_message[0]["text"] if system_message else DEFAULT_SYSTEM_PROMPT
                        })
                    )
                    # Parse the response body
                    response_body = json.loads(response.get('body').read())
                    return response_body
                except Exception as alt_error:
                    logger.error(f"Alternative API call also failed: {alt_error}")
                    # Try directly calling model with raw messages
                    response = self.client.converse(
                        messages=messages,
                        inferenceConfig={
                            "maxTokens": BEDROCK_MAX_TOKENS,
                            "temperature": BEDROCK_TEMPERATURE,
                            "topP": BEDROCK_TOP_P
                        }
                    )
                    return response
        except Exception as e:
            logger.error(f"Error in Bedrock converse call: {e}")
            raise
    
    def _enhance_system_prompt_with_confidence(self, system_prompt: str) -> str:
        """
        Add confidence assessment instructions to system prompt.
        
        Enhances any system prompt with standardized confidence assessment instructions
        to ensure Claude provides reliability indicators with its responses.
        
        Args:
            system_prompt: Base system prompt to enhance
            
        Returns:
            Enhanced system prompt with confidence assessment instructions appended
            
        Note:
            The confidence assessment instructions guide Claude to evaluate and
            report its confidence level (HIGH, MEDIUM, LOW) based on the quality
            and completeness of available information in the context.
        """
        enhanced_prompt = system_prompt + "\n\n" + CONFIDENCE_ASSESSMENT_INSTRUCTIONS
        return enhanced_prompt
    
    async def generate(self, query: str, context: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response using Claude via Amazon Bedrock with improved prompt formatting.
        
        This is the primary method for generating responses to user queries. It implements
        sophisticated query analysis to understand what the user is asking and formats
        the prompt to ensure focused, relevant responses.
        
        Args:
            query: User's question or information request
            context: Formatted context from document retrieval containing relevant information
            system_prompt: Optional custom system prompt; uses DEFAULT_SYSTEM_PROMPT if not provided
            
        Returns:
            Generated response text from Claude, formatted according to the query type and
            enhanced with confidence assessment
            
        Examples:
            >>> generator = RAGGenerator()
            >>> response = await generator.generate(
            ...     "What does the METHOD IF_EXAMPLE~PROCESS do?",
            ...     context="[Document 1] Method documentation...",
            ...     system_prompt="You are an ABAP expert..."
            ... )
            >>> print(response)
            # Confidence: HIGH
            # The IF_EXAMPLE~PROCESS method handles...
            
        Note:
            Implements intelligent query analysis including:
            - Method and interface pattern recognition
            - Query type classification (purpose, return, parameter queries)
            - Dynamic instruction generation for focused responses
            - Conversational tone guidelines for natural interaction
        """
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        
        # Enhance system prompt with confidence assessment instructions
        enhanced_system_prompt = self._enhance_system_prompt_with_confidence(system_prompt)
        
        logger.info(f"Generating response for query: '{query}'")
        
        # Parse the query to understand what's being asked
        query_lower = query.lower()
        
        # Identify key question types
        is_what_query = "what" in query_lower
        is_how_query = "how" in query_lower
        is_why_query = "why" in query_lower
        is_when_query = "when" in query_lower
        is_purpose_query = "purpose" in query_lower or "what does" in query_lower
        is_return_query = "return" in query_lower or "output" in query_lower
        is_parameter_query = "parameter" in query_lower or "input" in query_lower
        
        # Extract specific method or interface if referenced
        method_pattern = r'(\S+)~(\S+)'
        method_matches = re.findall(method_pattern, query)
        method_name = f"{method_matches[0][0]}~{method_matches[0][1]}" if method_matches else None
        
        # Create a focused instruction based on the query type
        query_instruction = "QUERY ANALYSIS:\n"
        
        if method_name:
            query_instruction += f"- User is asking about method: {method_name}\n"
            
            if is_purpose_query:
                query_instruction += "- They specifically want to know the PURPOSE of this method\n"
            if is_return_query:
                query_instruction += "- They specifically want to know what DATA it RETURNS\n"
            if is_parameter_query:
                query_instruction += "- They specifically want to know what PARAMETERS it takes\n"
        
        query_instruction += "\nRESPONSE REQUIREMENTS:\n"
        query_instruction += "1. Answer DIRECTLY and SPECIFICALLY what was asked\n"
        query_instruction += "2. Start with a clear, conversational statement that directly answers the question\n"
        query_instruction += "3. Do NOT use headings or formal structure\n"
        query_instruction += "4. Keep your response focused ONLY on what was asked\n"
        query_instruction += "5. Use a natural, conversational tone\n"
        query_instruction += "6. After fully answering what was asked, STOP - don't add extra information\n"
        
        # Prepare messages for Claude with the enhanced instructions
        messages = [
            {"role": "user", "content": f"I need information about: {query}\n\n{query_instruction}\n\nHere is the context from our documentation:\n\n{context}"}
        ]
        
        # Generate response with Claude
        try:
            return await self._generate_with_client(messages, enhanced_system_prompt)
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return f"Error generating response: {str(e)}"
    
    async def generate_with_history(self, query: str, context: str, conversation_history: List[Dict[str, str]], 
                                   system_prompt: Optional[str] = None) -> str:
        """
        Generate a response that takes conversation history into account with improved prompting.
        
        This method enables multi-turn conversations by incorporating previous conversation
        context while generating responses to follow-up questions. It maintains conversation
        flow while ensuring responses remain focused and relevant.
        
        Args:
            query: Current user query or follow-up question
            context: Formatted context from document retrieval for the current query
            conversation_history: List of previous conversation messages with 'role' and 'content'
            system_prompt: Optional custom system prompt; uses CONVERSATION_SYSTEM_PROMPT if not provided
            
        Returns:
            Generated response text that considers both the current query and conversation history
            
        Examples:
            >>> generator = RAGGenerator()
            >>> history = [
            ...     {"role": "user", "content": "What is an ABAP class?"},
            ...     {"role": "assistant", "content": "An ABAP class is..."}
            ... ]
            >>> response = await generator.generate_with_history(
            ...     "How do I create one?",
            ...     context="Class creation documentation...",
            ...     conversation_history=history
            ... )
            >>> print(response)
            # To create an ABAP class, you can...
            
        Note:
            The method processes conversation history to:
            - Maintain context from previous exchanges
            - Understand pronouns and references in follow-up questions
            - Generate responses that build upon previous information
            - Keep conversation flow natural and coherent
        """
        if system_prompt is None:
            system_prompt = CONVERSATION_SYSTEM_PROMPT
        
        # Enhance system prompt with confidence assessment instructions
        enhanced_system_prompt = self._enhance_system_prompt_with_confidence(system_prompt)
        
        logger.info(f"Generating response for follow-up query: '{query}' with conversation history")
        
        # Parse the query to understand what's being asked
        query_lower = query.lower()
        
        # Identify key question types for better response formatting
        is_what_query = "what" in query_lower
        is_how_query = "how" in query_lower
        is_why_query = "why" in query_lower
        is_when_query = "when" in query_lower
        is_purpose_query = "purpose" in query_lower or "what does" in query_lower
        is_return_query = "return" in query_lower or "output" in query_lower
        is_parameter_query = "parameter" in query_lower or "input" in query_lower
        
        # Extract specific method or interface if referenced
        method_pattern = r'(\S+)~(\S+)'
        method_matches = re.findall(method_pattern, query)
        method_name = f"{method_matches[0][0]}~{method_matches[0][1]}" if method_matches else None
        
        # Create a focused instruction based on the query type
        query_instruction = "QUERY ANALYSIS:\n"
        
        if method_name:
            query_instruction += f"- User is asking about method: {method_name}\n"
            
            if is_purpose_query:
                query_instruction += "- They specifically want to know the PURPOSE of this method\n"
            if is_return_query:
                query_instruction += "- They specifically want to know what DATA it RETURNS\n"
            if is_parameter_query:
                query_instruction += "- They specifically want to know what PARAMETERS it takes\n"
        
        query_instruction += "\nRESPONSE REQUIREMENTS:\n"
        query_instruction += "1. Answer DIRECTLY and SPECIFICALLY what was asked\n"
        query_instruction += "2. Start with a clear, conversational statement that directly answers the question\n"
        query_instruction += "3. Do NOT use headings or formal structure\n"
        query_instruction += "4. Keep your response focused ONLY on what was asked\n"
        query_instruction += "5. Use a natural, conversational tone\n"
        query_instruction += "6. After fully answering what was asked, STOP - don't add extra information\n"
        
        # Format conversation history for Claude
        messages = []
        
        # Add past conversation turns
        for message in conversation_history:
            # Only include user queries and assistant responses (not system messages or metadata)
            if message.get("role") in ["user", "assistant"] and "content" in message:
                messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })
        
        # Add the current query with context and focusing instructions
        messages.append({
            "role": "user", 
            "content": f"I need information about: {query}\n\n{query_instruction}\n\nHere is the context from our documentation:\n\n{context}"
        })
        
        # Generate response with Claude
        try:
            return await self._generate_with_client(messages, enhanced_system_prompt)
        except Exception as e:
            logger.error(f"Failed to generate response with history: {e}")
            return f"Error generating response: {str(e)}"
            
    async def generate_out_of_scope(self, query: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response for queries that are out of scope.
        
        This method handles queries that fall outside the SAP ABAP documentation domain
        by providing appropriate responses that redirect users to relevant topics while
        maintaining a helpful and professional tone.
        
        Args:
            query: User query that has been identified as out of scope
            system_prompt: Optional custom system prompt; uses OUT_OF_SCOPE_PROMPT if not provided
            
        Returns:
            Polite response explaining scope limitations and suggesting alternative approaches
            
        Examples:
            >>> generator = RAGGenerator()
            >>> response = await generator.generate_out_of_scope("What's the weather like?")
            >>> print(response)
            # I'm sorry, but your question is out of context...
            
        Note:
            Out-of-scope queries include:
            - Personal questions not related to ABAP
            - General knowledge questions
            - Questions about other programming languages
            - Non-technical queries
        """
        if system_prompt is None:
            system_prompt = OUT_OF_SCOPE_PROMPT
        
        # Enhance system prompt with confidence assessment instructions
        enhanced_system_prompt = self._enhance_system_prompt_with_confidence(system_prompt)
        
        logger.info(f"Generating out-of-scope response for query: '{query}'")
        
        # Prepare messages for Claude
        messages = [
            {"role": "user", "content": f"I need information about: {query}"}
        ]
        
        # Generate response with Claude
        try:
            return await self._generate_with_client(messages, enhanced_system_prompt)
        except Exception as e:
            logger.error(f"Failed to generate out-of-scope response: {e}")
            return "I'm sorry, but your question is out of context and not present in the SAP ABAP code files. I can only assist with questions related to SAP ABAP code and documentation."
    
    async def generate_out_of_scope_with_history(self, query: str, conversation_history: List[Dict[str, str]], 
                                                system_prompt: Optional[str] = None) -> str:
        """
        Generate a response for out-of-scope follow-up queries.
        
        This method handles out-of-scope queries that occur within an ongoing conversation,
        providing context-aware responses that acknowledge the conversation history while
        explaining scope limitations.
        
        Args:
            query: Out-of-scope follow-up query from the user
            conversation_history: List of previous conversation messages for context
            system_prompt: Optional custom system prompt; uses OUT_OF_SCOPE_CONVERSATION_PROMPT if not provided
            
        Returns:
            Context-aware response explaining scope limitations while maintaining conversation flow
            
        Examples:
            >>> generator = RAGGenerator()
            >>> history = [{"role": "user", "content": "Tell me about ABAP classes"}]
            >>> response = await generator.generate_out_of_scope_with_history(
            ...     "What's your favorite color?",
            ...     conversation_history=history
            ... )
            >>> print(response)
            # I understand we were discussing ABAP classes, but I can only help...
            
        Note:
            This method provides more nuanced responses than the basic out-of-scope
            handler by considering the conversation context and providing smoother
            transitions back to relevant topics.
        """
        if system_prompt is None:
            system_prompt = OUT_OF_SCOPE_CONVERSATION_PROMPT
        
        # Enhance system prompt with confidence assessment instructions
        enhanced_system_prompt = self._enhance_system_prompt_with_confidence(system_prompt)
        
        logger.info(f"Generating out-of-scope response for follow-up query: '{query}'")
        
        # Format conversation history for Claude
        messages = []
        
        # Add past conversation turns
        for message in conversation_history:
            # Only include user queries and assistant responses
            if message.get("role") in ["user", "assistant"] and "content" in message:
                messages.append({
                    "role": message["role"],
                    "content": message["content"]
                })
        
        # Add the current query
        messages.append({
            "role": "user", 
            "content": f"I need information about: {query}"
        })
        
        # Generate response with Claude
        try:
            return await self._generate_with_client(messages, enhanced_system_prompt)
        except Exception as e:
            logger.error(f"Failed to generate out-of-scope response with history: {e}")
            return "I'm sorry, but your question is out of context and not present in the SAP ABAP code files. I can only assist with questions related to SAP ABAP code and documentation."