#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced generator module with improved response quality and formatting.

This module provides the EnhancedRAGGenerator class that handles text generation
for the SAP ABAP documentation RAG system. It includes advanced response formatting,
context cleaning, and configurable generation parameters optimized for RTX 6000 Ada.

FIXED: Now uses config parameters instead of hardcoded values

The generator module implements sophisticated text generation capabilities including:
- Advanced prompt engineering with SAP ABAP context awareness
- Multiple response formatting modes (Enhanced, Basic, Raw)
- GPU memory optimization for RTX 6000 Ada architecture
- Conversation history integration for contextual responses
- Comprehensive response cleaning and post-processing
- Dynamic generation parameter updates during runtime
- Async/sync compatibility for various usage patterns

Key Components:
- EnhancedRAGGenerator: Main generator class with full functionality
- Response cleaning pipeline: Multi-stage text improvement
- Context preprocessing: Structure-preserving content cleaning
- Prompt engineering: SAP ABAP-specific prompt templates
- Memory management: GPU resource optimization
- Configuration management: Dynamic parameter updates

Technical Features:
- PyTorch integration with HuggingFace transformers
- CUDA acceleration with memory optimization
- Configurable generation parameters from config.py
- Multiple output formatting modes
- Error handling and graceful degradation
- Comprehensive logging and monitoring

Author: SAP ABAP RAG Team
Version: 1.0.0
License: MIT
"""

import os
import logging
from typing import List, Dict, Any, Optional, Union
import asyncio
import re
import concurrent.futures

from config import (
    LLM_MODEL, MODEL_PATHS, LLM_MAX_TOKENS, LLM_TEMPERATURE,
    LLM_TOP_P, LLM_TOP_K, MODEL_LOADING_CONFIG, RTX_6000_OPTIMIZATIONS,
    GENERATION_CONFIG
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    TORCH_AVAILABLE = True
except ImportError as e:
    logger.error(f"Required libraries not available: {e}")
    TORCH_AVAILABLE = False


class EnhancedRAGGenerator:
    """
    Enhanced generator with improved response quality and formatting.
    
    This class provides advanced text generation capabilities specifically designed
    for SAP ABAP documentation queries. It includes sophisticated prompt engineering,
    multiple response formatting modes, and optimizations for RTX 6000 Ada GPU.
    
    Features:
    - Enhanced prompt engineering for structured responses
    - Multiple formatting modes (Enhanced, Basic, Raw)
    - Context cleaning and preservation
    - Conversation history support
    - Dynamic parameter updates
    - GPU memory optimization
    
    Attributes:
        model_name (str): Name of the language model
        model_path (str): Path to the model (local or Hub)
        device (str): Device for model execution (cuda:0 or cpu)
        tokenizer: HuggingFace tokenizer instance
        model: HuggingFace model instance
        max_tokens (int): Maximum number of tokens to generate
        formatting_mode (str): Current formatting mode
        generation_config (Dict[str, Any]): Generation configuration parameters
    """
    
    def __init__(self) -> None:
        """
        Initialize the Enhanced RAG Generator.
        
        Sets up the model, tokenizer, and generation configuration using
        parameters from the config module.
        
        Raises:
            ImportError: If PyTorch and transformers are not available
            Exception: If model initialization fails
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch and transformers are required")
        
        self.model_name = LLM_MODEL
        self.model_path = MODEL_PATHS.get(self.model_name, self.model_name)
        
        logger.info(f"Initializing Enhanced Generator: {self.model_name}")
        
        self.device = self._setup_device()
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        
        # FIXED: Use exact config parameters - do not override
        self.max_tokens = LLM_MAX_TOKENS  # Use exact config value
        self.formatting_mode = "Enhanced"
        
        # FIXED: Use exact GENERATION_CONFIG from config.py
        self.generation_config = {
            "max_new_tokens": LLM_MAX_TOKENS,  # Use config value exactly
            "temperature": LLM_TEMPERATURE,   # Use config value exactly  
            "top_p": LLM_TOP_P,              # Use config value exactly
            "top_k": LLM_TOP_K,              # Use config value exactly
            "repetition_penalty": GENERATION_CONFIG.get("repetition_penalty", 1.1),
            "length_penalty": GENERATION_CONFIG.get("length_penalty", 1.0),
            "early_stopping": GENERATION_CONFIG.get("early_stopping", True),
            "no_repeat_ngram_size": GENERATION_CONFIG.get("no_repeat_ngram_size", 3),
            "do_sample": GENERATION_CONFIG.get("do_sample", True),
            "num_beams": GENERATION_CONFIG.get("num_beams", 1),
            "use_cache": GENERATION_CONFIG.get("use_cache", True)
        }
        
        logger.info(f"FIXED: Using EXACT config parameters:")
        logger.info(f"  LLM_MAX_TOKENS: {LLM_MAX_TOKENS}")
        logger.info(f"  LLM_TEMPERATURE: {LLM_TEMPERATURE}")
        logger.info(f"  LLM_TOP_P: {LLM_TOP_P}")
        logger.info(f"  LLM_TOP_K: {LLM_TOP_K}")
        logger.info(f"FIXED: Generation config initialized with config values")
        
        self._initialize_model()
    
    def _setup_device(self) -> str:
        """
        Set up the appropriate device for model execution.
        
        Detects available CUDA devices and logs GPU information.
        Falls back to CPU if CUDA is not available.
        
        Returns:
            str: Device string ('cuda:0' or 'cpu')
        """
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"Using GPU: {device_name} ({total_memory:.1f}GB)")
            return "cuda:0"
        else:
            logger.warning("CUDA not available, using CPU")
            return "cpu"

    def _get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration for loading with RTX 6000 optimizations.
        
        Combines configuration from config.py with device-specific optimizations
        for the RTX 6000 Ada GPU.
        
        Returns:
            Dict[str, Any]: Model configuration dictionary
        """
        # FIXED: Use MODEL_LOADING_CONFIG from config.py
        config = MODEL_LOADING_CONFIG.copy()
        if config.get("torch_dtype") == "float16":
            config["torch_dtype"] = torch.float16
        elif config.get("torch_dtype") == "float32":
            config["torch_dtype"] = torch.float32
        
        # Use RTX 6000 optimizations from config
        config["attn_implementation"] = RTX_6000_OPTIMIZATIONS.get("attention_implementation", "eager")
        config.update({
            "device_map": "auto",
            "max_memory": {0: f"{RTX_6000_OPTIMIZATIONS.get('max_memory_gb', 45)}GB"},
            "torch_dtype": torch.float16,
        })
        return config
    
    def _initialize_model(self) -> None:
        """
        Initialize the model and tokenizer.
        
        Loads the model either from a local path or from HuggingFace Hub,
        sets up the tokenizer with appropriate padding tokens, and applies
        RTX 6000 specific optimizations.
        
        Raises:
            Exception: If model or tokenizer loading fails
        """
        try:
            is_local_path = os.path.exists(self.model_path) and os.path.isdir(self.model_path)
            
            if is_local_path:
                logger.info(f"Loading model from: {self.model_path}")
                model_identifier = self.model_path
                load_kwargs = {"local_files_only": True, "trust_remote_code": True}
            else:
                logger.info(f"Loading model from Hub: {self.model_name}")
                model_identifier = self.model_name
                load_kwargs = {"trust_remote_code": True}
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_identifier, **load_kwargs)
            
            if self.tokenizer.pad_token is None:
                if hasattr(self.tokenizer, 'eos_token') and self.tokenizer.eos_token:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                else:
                    self.tokenizer.pad_token = "</s>"
            
            model_config = self._get_model_config()
            model_config.update(load_kwargs)
            self.model = AutoModelForCausalLM.from_pretrained(model_identifier, **model_config)
            
            logger.info("Enhanced model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def set_generation_config(
        self, 
        max_new_tokens: Optional[int] = None, 
        formatting_mode: Optional[str] = None, 
        **kwargs: Any
    ) -> None:
        """
        Set generation configuration parameters.
        
        Updates generation parameters while respecting config defaults.
        Only updates parameters that are explicitly provided.
        
        Args:
            max_new_tokens (Optional[int]): Maximum number of tokens to generate
            formatting_mode (Optional[str]): Response formatting mode ('Enhanced', 'Basic', 'Raw')
            **kwargs: Additional generation parameters
        """
        # FIXED: Only update if explicitly provided, otherwise keep config defaults
        if max_new_tokens is not None:
            self.generation_config["max_new_tokens"] = max_new_tokens
            self.max_tokens = max_new_tokens
            logger.info(f"FIXED: max_tokens explicitly set to: {max_new_tokens}")
        else:
            # Keep the config default
            logger.info(f"FIXED: Keeping config default max_tokens: {self.max_tokens}")
            
        if formatting_mode is not None:
            self.formatting_mode = formatting_mode
            logger.info(f"FIXED: formatting_mode set to: {formatting_mode}")
            
        # Update any additional parameters only if provided
        for key, value in kwargs.items():
            if key in self.generation_config and value is not None:
                self.generation_config[key] = value
                logger.info(f"FIXED: {key} set to: {value}")
        
        # Log final config
        logger.info(f"FIXED: Final generation config - max_tokens={self.generation_config['max_new_tokens']}, temp={self.generation_config['temperature']}, top_p={self.generation_config['top_p']}, top_k={self.generation_config['top_k']}")
    
    def _create_enhanced_prompt(self, query: str, context: str, system_prompt: str = "") -> str:
        """
        Create an enhanced prompt for better structured responses.
        
        Builds a comprehensive prompt that includes system instructions,
        cleaned context, and specific formatting guidelines to produce
        well-structured responses.
        
        Args:
            query (str): User's question or query
            context (str): Retrieved documentation context
            system_prompt (str): Optional system prompt override
            
        Returns:
            str: Enhanced prompt for the language model
        """
        # Clean context but preserve structure
        clean_context = self._clean_context_preserving_structure(context)
        
        # Use system prompt if provided, otherwise use enhanced default
        if not system_prompt:
            system_prompt = """You are an expert SAP ABAP documentation assistant. Provide comprehensive, well-structured answers based on the provided documentation. 

Your responses should be:
- Comprehensive and detailed
- Well-structured with clear headings
- Include specific code examples when available
- Use proper formatting with bullet points and numbered lists
- Be precise about method names, parameters, and functionality"""
        
        # Enhanced prompt with better formatting instructions
        enhanced_prompt = f"""<s>[INST] {system_prompt}

Based on the SAP ABAP documentation provided below, answer the user's question with a well-structured, comprehensive response.

SAP ABAP Documentation:
{clean_context}

User Question: {query}

Formatting Guidelines:
1. Start with a brief overview of the topic
2. Use clear section headings with ## syntax for main sections
3. Use ### for subsections when needed
4. Use bullet points (*) for lists and numbered lists (1. 2. 3.) when appropriate
5. Include code examples using proper formatting when available
6. Use proper paragraph breaks for readability
7. Be specific about method names, parameters, and functionality
8. Structure your response logically: Purpose ? Implementation ? Parameters ? Examples ? Notes
9. If discussing error handling, be specific about which errors are handled where
10. Use blank lines between sections for better readability

Provide a comprehensive, well-formatted answer based strictly on the documentation provided.
[/INST]"""
        
        return enhanced_prompt

    def _clean_context_preserving_structure(self, context: str) -> str:
        """
        Clean context while preserving important structure.
        
        Removes common artifacts and formatting issues while maintaining
        the logical structure of the documentation content.
        
        Args:
            context (str): Raw context from document retrieval
            
        Returns:
            str: Cleaned context with preserved structure
        """
        if not context:
            return ""
        
        # Remove obvious artifacts but preserve formatting
        cleaned = context.replace("DOCUMNET", "DOCUMENT")
        cleaned = cleaned.replace("##END OF CODE SNIPLET##", "")
        cleaned = cleaned.replace("##END OF CODE SNIPPET##", "")
        cleaned = cleaned.replace("#END OF CODE", "")
        cleaned = cleaned.replace("============", "")
        
        # Fix common formatting issues
        cleaned = cleaned.replace("job\\_", "job_")
        cleaned = cleaned.replace("\\_", "_")
        cleaned = cleaned.replace("__", "_")
        
        # Preserve document structure but clean up formatting
        lines = cleaned.split('\n')
        processed_lines = []
        
        for line in lines:
            # Clean up excess spaces within lines but preserve line structure
            processed_line = ' '.join(line.split())
            if processed_line:  # Only add non-empty lines
                processed_lines.append(processed_line)
            elif processed_lines and processed_lines[-1]:  # Preserve paragraph breaks
                processed_lines.append("")
        
        return '\n'.join(processed_lines)

    async def _generate_enhanced_response(
        self, 
        query: str, 
        context: str, 
        system_prompt: str = ""
    ) -> str:
        """
        Generate response with enhanced settings for better quality.
        
        Performs the actual text generation using the enhanced prompt and
        applies response cleaning based on the current formatting mode.
        
        Args:
            query (str): User's question
            context (str): Retrieved documentation context
            system_prompt (str): Optional system prompt
            
        Returns:
            str: Generated and cleaned response
            
        Raises:
            Exception: If generation fails
        """
        try:
            # Create enhanced prompt
            prompt = self._create_enhanced_prompt(query, context, system_prompt)
            
            # FIXED: Use RTX_6000_OPTIMIZATIONS for context window
            max_length = RTX_6000_OPTIMIZATIONS.get("max_sequence_length", 8192)
            
            # Tokenize with generous length limits
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            
            if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # FIXED: Use self.generation_config which now uses exact config parameters
            gen_config = self.generation_config.copy()
            gen_config.update({
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": True
            })
            
            logger.info(f"FIXED GENERATION: Using EXACT config parameters:")
            logger.info(f"  max_tokens: {gen_config['max_new_tokens']} (config: {LLM_MAX_TOKENS})")
            logger.info(f"  temperature: {gen_config['temperature']} (config: {LLM_TEMPERATURE})")
            logger.info(f"  top_p: {gen_config['top_p']} (config: {LLM_TOP_P})")
            logger.info(f"  top_k: {gen_config['top_k']} (config: {LLM_TOP_K})")
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_config)
            
            response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
            
            # Enhanced response cleaning based on formatting mode
            if self.formatting_mode == "Enhanced":
                cleaned_response = self._clean_response_enhanced(response)
            elif self.formatting_mode == "Basic":
                cleaned_response = self._clean_response_basic(response)
            else:  # Raw
                cleaned_response = self._clean_response_minimal(response)
            
            logger.info(f"FIXED GENERATION: Response generated - {len(cleaned_response)} chars")
            
            return cleaned_response
            
        except Exception as e:
            logger.error(f"Error in enhanced generation: {e}")
            return "I apologize, but I encountered an error while processing your question about SAP ABAP documentation."
    
    def _clean_response_enhanced(self, response: str) -> str:
        """
        Enhanced response cleaning with full formatting preservation and improvement.
        
        Applies comprehensive cleaning and formatting enhancements including:
        - Artifact removal
        - Formatting fixes for headings, lists, and code blocks
        - Smart paragraph handling
        - Method name corrections
        
        Args:
            response (str): Raw generated response
            
        Returns:
            str: Enhanced and formatted response
        """
        if not response:
            return ""
        
        logger.info("ENHANCED CLEANING: Starting comprehensive response cleanup")
        
        # Remove artifacts but preserve structure
        cleaned = response.replace("##END OF CODE SNIPLET##", "")
        cleaned = cleaned.replace("##END OF CODE SNIPPET##", "")
        cleaned = cleaned.replace("#END OF CODE", "")
        cleaned = cleaned.replace("DOCUMNET", "DOCUMENT")
        
        # Fix common OCR/formatting errors
        cleaned = re.sub(r'mo_bt[?c]_facade', 'mo_btch_facade', cleaned)
        cleaned = cleaned.replace("job\\_", "job_")
        cleaned = cleaned.replace("\\_", "_")
        cleaned = cleaned.replace("__", "_")
        
        # Fix broken method names
        cleaned = re.sub(r'(\w+)_ (\w+)', r'\1_\2', cleaned)
        
        # Enhanced heading formatting
        lines = cleaned.split('\n')
        formatted_lines = []
        
        for line in lines:
            stripped = line.strip()
            
            # Fix main headings
            if stripped.startswith('##') and not stripped.startswith('## '):
                stripped = '## ' + stripped[2:].strip()
            elif stripped.startswith('#') and not stripped.startswith('# '):
                stripped = '# ' + stripped[1:].strip()
            
            # Fix subheadings
            elif stripped.startswith('###') and not stripped.startswith('### '):
                stripped = '### ' + stripped[3:].strip()
            
            # Fix bullet points
            elif re.match(r'^[*+-][^\s]', stripped):
                stripped = re.sub(r'^([*+-])', r'\1 ', stripped)
            
            # Fix numbered lists
            elif re.match(r'^\d+\.[^\s]', stripped):
                stripped = re.sub(r'^(\d+\.)', r'\1 ', stripped)
            
            # Fix code blocks
            elif stripped.startswith('```') and len(stripped) > 3 and not stripped.startswith('``` '):
                stripped = '```' + stripped[3:]
            
            formatted_lines.append(stripped if stripped else "")
        
        # Smart paragraph handling
        final_lines = []
        empty_line_count = 0
        
        for line in formatted_lines:
            if not line:
                empty_line_count += 1
                # Allow up to 2 consecutive empty lines for better structure
                if empty_line_count <= 2:
                    final_lines.append("")
            else:
                empty_line_count = 0
                # Clean up extra spaces within the line
                clean_line = ' '.join(line.split())
                final_lines.append(clean_line)
        
        # Remove empty lines at the start and end
        while final_lines and not final_lines[0]:
            final_lines.pop(0)
        while final_lines and not final_lines[-1]:
            final_lines.pop()
        
        result = '\n'.join(final_lines)
        
        # Final formatting improvements
        result = self._apply_final_formatting_enhancements(result)
        
        logger.info(f"ENHANCED CLEANING: Response enhanced and formatted - {len(result)} chars")
        
        return result
    
    def _clean_response_basic(self, response: str) -> str:
        """
        Basic response cleaning with minimal formatting.
        
        Applies essential cleaning while preserving the original structure
        of the response with minimal modifications.
        
        Args:
            response (str): Raw generated response
            
        Returns:
            str: Basically cleaned response
        """
        if not response:
            return ""
        
        # Remove artifacts
        cleaned = response.replace("##END OF CODE SNIPLET##", "")
        cleaned = cleaned.replace("##END OF CODE SNIPPET##", "")
        cleaned = cleaned.replace("#END OF CODE", "")
        cleaned = cleaned.replace("DOCUMNET", "DOCUMENT")
        
        # Basic formatting fixes
        cleaned = cleaned.replace("job\\_", "job_")
        cleaned = cleaned.replace("\\_", "_")
        
        # Clean up excessive whitespace
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)  # Reduce excessive newlines
        cleaned = re.sub(r' {2,}', ' ', cleaned)      # Reduce excessive spaces
        
        return cleaned.strip()
    
    def _clean_response_minimal(self, response: str) -> str:
        """
        Minimal response cleaning for raw output.
        
        Removes only the most obvious artifacts while preserving
        the raw structure of the generated response.
        
        Args:
            response (str): Raw generated response
            
        Returns:
            str: Minimally cleaned response
        """
        if not response:
            return ""
        
        # Only remove obvious artifacts
        cleaned = response.replace("##END OF CODE SNIPLET##", "")
        cleaned = cleaned.replace("##END OF CODE SNIPPET##", "")
        cleaned = cleaned.replace("#END OF CODE", "")
        
        return cleaned.strip()
    
    def _apply_final_formatting_enhancements(self, text: str) -> str:
        """
        Apply final formatting enhancements to the text.
        
        Performs final pass formatting to ensure proper spacing around
        various elements like headings, lists, and code blocks.
        
        Args:
            text (str): Text to enhance
            
        Returns:
            str: Text with final formatting enhancements
        """
        # Ensure proper spacing around headings
        text = re.sub(r'\n(#{1,3} [^\n]+)\n(?!\n)', r'\n\1\n\n', text)
        
        # Ensure proper spacing before bullet points
        text = re.sub(r'\n([*+-] [^\n]+)', r'\n\n\1', text)
        
        # Ensure proper spacing for numbered lists
        text = re.sub(r'\n(\d+\. [^\n]+)', r'\n\n\1', text)
        
        # Clean up multiple consecutive newlines (max 2)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Ensure code blocks have proper spacing
        text = re.sub(r'\n```([a-z]*)\n', r'\n\n```\1\n', text)
        text = re.sub(r'\n```\n', r'\n```\n\n', text)
        
        return text
    
    async def generate(self, prompt: str, context: str = "", system_prompt: str = "") -> str:
        """
        Main generation method for async usage.
        
        Primary interface for generating responses using the enhanced
        generation pipeline with context and optional system prompts.
        
        Args:
            prompt (str): User's query or prompt
            context (str): Retrieved documentation context
            system_prompt (str): Optional system prompt override
            
        Returns:
            str: Generated response
        """
        return await self._generate_enhanced_response(prompt, context, system_prompt)
    
    async def generate_with_history(
        self, 
        prompt: str, 
        context: str = "", 
        conversation_history: Optional[List[Dict[str, str]]] = None, 
        system_prompt: str = ""
    ) -> str:
        """
        Generate response with conversation history context.
        
        Incorporates conversation history into the generation process
        for better contextual understanding and continuity.
        
        Args:
            prompt (str): User's current query
            context (str): Retrieved documentation context
            conversation_history (Optional[List[Dict[str, str]]]): Previous conversation
            system_prompt (str): Optional system prompt override
            
        Returns:
            str: Generated response with history context
        """
        # Enhanced context with history
        if conversation_history:
            history_context = self._format_conversation_history(conversation_history)
            enhanced_system_prompt = f"{system_prompt}\n\nConversation History:\n{history_context}"
        else:
            enhanced_system_prompt = system_prompt
            
        return await self._generate_enhanced_response(prompt, context, enhanced_system_prompt)
    
    def _format_conversation_history(self, history: List[Dict[str, str]]) -> str:
        """
        Format conversation history for context inclusion.
        
        Formats the conversation history into a readable format
        for inclusion in the generation prompt.
        
        Args:
            history (List[Dict[str, str]]): List of conversation messages
            
        Returns:
            str: Formatted conversation history
        """
        formatted_history = []
        for item in history[-4:]:  # Last 4 exchanges
            role = item.get('role', 'unknown')
            content = item.get('content', '')[:200]  # Limit length
            formatted_history.append(f"{role.title()}: {content}")
        return '\n'.join(formatted_history)
    
    def generate_response(
        self, 
        query: str, 
        context: str = "", 
        conversation_history: Optional[List] = None, 
        system_prompt: str = ""
    ) -> str:
        """
        Synchronous generate method for RAG pipeline compatibility.
        
        Provides synchronous interface for the generation functionality
        while internally handling async operations properly.
        
        Args:
            query (str): User's query
            context (str): Retrieved documentation context
            conversation_history (Optional[List]): Previous conversation messages
            system_prompt (str): Optional system prompt override
            
        Returns:
            str: Generated response
        """
        try:
            # Run async function synchronously
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create new loop for nested async
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        future = executor.submit(
                            asyncio.run, 
                            self._generate_enhanced_response(query, context, system_prompt)
                        )
                        return future.result()
                else:
                    return loop.run_until_complete(
                        self._generate_enhanced_response(query, context, system_prompt)
                    )
            except RuntimeError:
                return asyncio.run(self._generate_enhanced_response(query, context, system_prompt))
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error while processing your question."
    
    def _unload_model(self) -> None:
        """
        Unload model from GPU memory.
        
        Properly unloads the model and tokenizer from memory and
        clears the GPU cache to free up resources.
        """
        try:
            if self.model is not None:
                del self.model
                self.model = None
                logger.info("Model unloaded from memory")
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                logger.info("Tokenizer unloaded from memory")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
                
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Retrieves detailed information about the model, generation configuration,
        and system resources for monitoring and debugging purposes.
        
        Returns:
            Dict[str, Any]: Comprehensive model information including:
                - Model configuration
                - Generation parameters
                - Device information
                - Memory usage (if available)
                - Parameter counts
        """
        info = {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "device": self.device,
            "generation_mode": "enhanced_generation",
            "max_tokens": self.max_tokens,
            "formatting_mode": self.formatting_mode,
            "temperature": self.generation_config.get("temperature", LLM_TEMPERATURE),
            "top_p": self.generation_config.get("top_p", LLM_TOP_P),
            "top_k": self.generation_config.get("top_k", LLM_TOP_K),
            "repetition_penalty": self.generation_config.get("repetition_penalty", 1.1),
            "formatting_enabled": True,
            "enhanced_prompting": True,
            "config_parameters_used": True  # FIXED: Now using config parameters
        }
        
        if self.model:
            try:
                info.update({
                    "torch_dtype": str(next(self.model.parameters()).dtype),
                    "param_count": sum(p.numel() for p in self.model.parameters()),
                    "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                })
            except Exception as e:
                logger.warning(f"Could not get model parameter info: {e}")
        
        if torch.cuda.is_available():
            try:
                info.update({
                    "device_name": torch.cuda.get_device_name(0),
                    "total_memory": f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB",
                    "allocated_memory": f"{torch.cuda.memory_allocated(0) / (1024**3):.1f}GB",
                    "cached_memory": f"{torch.cuda.memory_reserved(0) / (1024**3):.1f}GB"
                })
            except Exception as e:
                logger.warning(f"Could not get GPU memory info: {e}")
        
        return info
    
    def update_generation_settings(self, **kwargs: Any) -> None:
        """
        Update generation settings dynamically.
        
        Allows runtime modification of generation parameters for
        experimentation and optimization.
        
        Args:
            **kwargs: Generation parameters to update (must be valid keys
                     in the generation_config dictionary)
        """
        updated_params = []
        for key, value in kwargs.items():
            if key in self.generation_config:
                old_value = self.generation_config[key]
                self.generation_config[key] = value
                updated_params.append(f"{key}: {old_value} -> {value}")
        
        if updated_params:
            logger.info(f"Updated generation settings: {', '.join(updated_params)}")
        else:
            logger.warning("No valid parameters provided for update")


# Backward compatibility alias
RAGGenerator = EnhancedRAGGenerator


def create_generator() -> EnhancedRAGGenerator:
    """
    Factory function to create an enhanced generator.
    
    Provides a simple factory interface for creating generator instances
    with default configuration.
    
    Returns:
        EnhancedRAGGenerator: Configured generator instance
    """
    return EnhancedRAGGenerator()


def test_generator() -> bool:
    """
    Test the enhanced generator functionality.
    
    Performs a comprehensive test of the generator including initialization,
    model info retrieval, and basic generation functionality.
    
    Returns:
        bool: True if all tests pass, False otherwise
    """
    try:
        generator = EnhancedRAGGenerator()
        print("? Enhanced Generator initialized successfully")
        
        info = generator.get_model_info()
        print(f"Model info: {info}")
        
        # Test generation
        test_context = "This is test documentation about SAP ABAP methods."
        test_query = "What is this about?"
        
        response = asyncio.run(generator.generate(test_query, test_context))
        print(f"Test response: {response[:100]}...")
        
        return True
    except Exception as e:
        print(f"? Enhanced Generator test failed: {e}")
        return False


if __name__ == "__main__":
    test_generator()