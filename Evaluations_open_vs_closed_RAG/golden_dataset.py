"""
# Enhanced RAGET-style Evaluation Dataset Generator with tqdm Progress Bars
# 
# This script generates evaluation datasets for RAG systems with all RAGET question types:
# - Simple questions
# - Complex questions
# - Distracting questions
# - Situational questions
# - Double questions
# - Conversational questions
#
# Saves the data in both JSON and CSV formats
# Improved to ensure the target number of questions are generated for each type
"""

# Import required libraries
import os
import sys
import logging
import pandas as pd
import json
import asyncio
import nest_asyncio
import random
import time
import re
from typing import List, Dict, Any, Optional, Sequence
from dotenv import load_dotenv
from tqdm import tqdm
from difflib import SequenceMatcher
# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested event loops in Jupyter/scripts
nest_asyncio.apply()

# Load environment variables
load_dotenv(r"claude.env")

# Verify environment variables are loaded
logger.info(f"QDRANT_HOST: {os.getenv('QDRANT_HOST', 'Not found')}")
logger.info(f"Collection Classes: {os.getenv('COLLECTION_NAME_CLASSES', 'Not found')}")
logger.info(f"Collection Reports: {os.getenv('COLLECTION_NAME_REPORTS', 'Not found')}")
logger.info(f"Using Bedrock model: {os.getenv('BEDROCK_MODEL', 'Not found')}")

# Import necessary modules from existing RAG pipeline
from config import (
    COLLECTIONS,
    DEFAULT_COLLECTION_TYPE
)
from retriever import RAGRetriever
from pipeline import RAGPipeline
from generator import RAGGenerator

# Extended RAG Generator for evaluation dataset creation
class RAGGeneratorExtended(RAGGenerator):
    """Extended RAG Generator with additional methods for testing"""
    
    async def generate_simple_prompt(self, prompt, max_retries=3, retry_delay=1):
        """
        Generate a simple response to a prompt without context, with retry logic
        
        Args:
            prompt: The text prompt to generate from
            max_retries: Maximum number of retries on failure
            retry_delay: Seconds to wait between retries
            
        Returns:
            Generated text response
        """
        system_prompt = """You are an expert assistant for SAP ABAP documentation.
        Answer questions based on your knowledge of SAP ABAP.
        Be comprehensive, accurate, and precise."""
        
        for attempt in range(max_retries):
            try:
                # Use the existing generate method with an empty context
                response = await self.generate(prompt, "")
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Retry {attempt+1}/{max_retries} after error: {e}")
                    await asyncio.sleep(retry_delay)  # Add delay between retries
                else:
                    logger.error(f"All {max_retries} attempts failed in generate_simple_prompt: {e}")
                    # Fall back to a simpler generation method if possible
                    try:
                        response = await self._generate_with_model(system_prompt, prompt)
                        return response
                    except Exception as fallback_error:
                        logger.error(f"Fallback generation also failed: {fallback_error}")
                        # Return a placeholder to avoid breaking the pipeline
                        return f"[Generation failed after {max_retries} attempts]"

# Function to create a pool of documents for question generation
def get_document_pool(collection_type, min_pool_size=200, quality_threshold=100):
    """
    Get a pool of documents for question generation with enhanced filtering
    
    Args:
        collection_type: Type of collection ('classes' or 'reports')
        min_pool_size: Minimum number of documents to retrieve
        quality_threshold: Minimum text length to consider a document valid
        
    Returns:
        List of documents with payload, list of topic names
    """
    # Initialize retriever
    retriever = RAGRetriever()
    collection_name = retriever.set_collection(collection_type)
    
    # Get documents from collection - request more to ensure enough quality docs
    requested_docs = min_pool_size * 3  # Request 3x the needed docs
    client = retriever.client
    
    # Use pagination to get more documents if needed
    all_docs = []
    offset = None
    while len(all_docs) < requested_docs:
        batch, offset = client.scroll(
            collection_name=collection_name,
            limit=1000,  # Get in large batches
            with_payload=True,
            offset=offset
        )
        
        if not batch:
            break  # No more documents
            
        all_docs.extend(batch)
        
        # Stop if we've retrieved enough or no more pagination
        if len(all_docs) >= requested_docs or offset is None:
            break
    
    logger.info(f"Retrieved {len(all_docs)} documents from collection {collection_name}")
    
    # Filter documents that have sufficient content
    valid_docs = []
    for doc in all_docs:
        title = doc.payload.get("title", "")
        text = doc.payload.get("text", "")
        
        if title and text and len(text) > quality_threshold:
            valid_docs.append(doc)
    
    logger.info(f"Found {len(valid_docs)} valid documents with sufficient content")
    
    # If we don't have enough valid docs, lower the threshold
    if len(valid_docs) < min_pool_size and quality_threshold > 50:
        logger.warning(f"Not enough quality documents. Lowering threshold and retrying.")
        return get_document_pool(collection_type, min_pool_size, quality_threshold // 2)
    
    # Extract topic information
    topics = {}
    for doc in valid_docs:
        title = doc.payload.get("title", "")
        if title:
            parts = title.split('.')
            if len(parts) > 1:
                topic = parts[0]
                if topic not in topics:
                    topics[topic] = 0
                topics[topic] += 1
    
    # Get top topics or use defaults if none found
    if topics:
        top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10]
        top_topic_names = [t[0] for t in top_topics]
    else:
        # Default topics if none are found
        logger.warning("No topics found in documents. Using default topics.")
        if collection_type == "classes":
            top_topic_names = ["CL", "IF", "Object", "Method", "Class", "Interface", "ABAP OO", "SAP", "Data", "Utilities"]
        else:
            top_topic_names = ["Report", "Module", "Function", "Program", "Processing", "SAP", "ABAP", "Data", "Utilities", "Forms"]
    
    return valid_docs, top_topic_names

# Function to get topic for a document
def get_topic_for_document(doc, top_topic_names, doc_index):
    """Determine the topic for a document"""
    title = doc.payload.get("title", "")
    
    if title:
        parts = title.split('.')
        if len(parts) > 1:
            topic = parts[0]
        else:
            topic = title
    else:
        topic = "General"
    
    # Make sure topic exists in our list
    if top_topic_names and (topic not in top_topic_names):
        topic = top_topic_names[doc_index % len(top_topic_names)]
    elif not top_topic_names:
        # Fallback if no topics were found
        default_topics = ["ABAP", "SAP", "Documentation", "Programming", "Utilities"]
        topic = default_topics[doc_index % len(default_topics)]
    
    return topic

# Function to format document content
def format_document_content(doc):
    """Format document content for reference context"""
    title = doc.payload.get("title", "")
    text = doc.payload.get("text", "")
    code = doc.payload.get("code_snippet", "")
    
    reference_context = f"Document {doc.id}: Title: {title}\n\n{text}"
    if code:
        reference_context += f"\n\nCode:\n```\n{code}\n```"
    
    return reference_context

# Function to clean up question text by removing confidence levels
def clean_question_format(question_text):
    """Remove confidence levels and other formatting from questions"""
    if not question_text:
        return question_text
        
    # Common confidence patterns to remove
    confidence_patterns = [
        r"^Confidence:\s*(HIGH|MEDIUM|LOW)\s*\n+",
        r"^(HIGH|MEDIUM|LOW)\s+Confidence:\s*\n+",
        r"^\[(HIGH|MEDIUM|LOW)\]\s*\n+"
    ]
    
    # Remove confidence levels
    for pattern in confidence_patterns:
        question_text = re.sub(pattern, "", question_text, flags=re.IGNORECASE)
    
    return question_text.strip()

# Generator for different question types
class RAGETQuestionGenerator:
    """Generator for RAGET-style questions of different types"""
    
    def __init__(self, collection_type, document_pool, top_topic_names):
        """Initialize with document pool and collection info"""
        self.collection_type = collection_type
        self.document_pool = document_pool
        self.top_topic_names = top_topic_names
        self.generator = RAGGeneratorExtended()
    
    def _cleanup_question_text(self, question_text):
        """Remove explanatory text and prefixes from questions"""
        # Remove common prefixes
        prefixes_to_remove = [
            "Question:", "Here's a question:", "Simple question:", "QUESTION:", 
            "Here is a simple question:", "My question is:", "Based on the document:",
            "Complex question:", "Distracting question:", "Situational question:",
            "Double question:", "Follow-up question:", "Conversational question:"
        ]
        
        # Try to clean up each prefix
        cleaned_question = question_text
        for prefix in prefixes_to_remove:
            if cleaned_question.startswith(prefix):
                cleaned_question = cleaned_question[len(prefix):].strip()
        
        # Remove any suffix explanations
        explanatory_suffixes = [
            "This question is based on", "This question relates to", 
            "This question is about", "This is a question about",
            "This is asking about", "The question is asking"
        ]
        
        for suffix in explanatory_suffixes:
            if suffix in cleaned_question:
                # Only keep the part before the suffix if it's a complete sentence
                suffix_pos = cleaned_question.find(suffix)
                if suffix_pos > 0:
                    candidate = cleaned_question[:suffix_pos].strip()
                    if candidate.endswith('?'):
                        cleaned_question = candidate
        
        # Remove any explanations in parentheses
        cleaned_question = re.sub(r'\([^)]*\)', '', cleaned_question).strip()
        
        # Ensure the question ends with a question mark
        if not cleaned_question.endswith('?'):
            if '?' in cleaned_question:
                # If there's a question mark in the middle, keep only up to that point
                cleaned_question = cleaned_question.split('?')[0] + '?'
            else:
                # If no question mark, we might have a malformed question
                logger.warning(f"Generated text doesn't appear to be a proper question: '{cleaned_question}'")
        
        # Remove confidence levels
        cleaned_question = clean_question_format(cleaned_question)
        
        return cleaned_question
    
    async def generate_simple_question(self, doc, doc_index, max_retries=3):
        """Generate a simple question based on document content"""
        reference_context = format_document_content(doc)
        topic = get_topic_for_document(doc, self.top_topic_names, doc_index)
        
        # Prompt for simple question generation
        question_prompt = f"""
        Based on the following document about SAP ABAP {self.collection_type}, create a simple, 
        straightforward question that directly asks about information in the document.
        
        DOCUMENT:
        {reference_context}
        
        Create a simple question that:
        1. Is specific to this document's content
        2. Requires basic understanding of SAP ABAP concepts
        3. Can be answered directly from the text
        4. IMPORTANT: Write only the question itself with no additional explanations, reasoning, or context
        
        Write only the QUESTION and nothing else:
        """
        
        for attempt in range(max_retries):
            try:
                question = await self.generator.generate_simple_prompt(question_prompt)
                
                # Cleanup any explanatory text
                question = self._cleanup_question_text(question)
                
                # Generate reference answer
                answer_prompt = f"""
                Answer the following question about SAP ABAP {self.collection_type} based on the provided documentation.
                Be comprehensive, accurate, and detailed in your response.
                
                DOCUMENT:
                {reference_context}
                
                QUESTION:
                {question}
                
                ANSWER:
                """
                
                reference_answer = await self.generator.generate_simple_prompt(answer_prompt)
                
                # Clean up any confidence indicators from the reference answer
                reference_answer = clean_question_format(reference_answer)
                
                # Only return if both question and answer generation succeeded
                if question and reference_answer and not question.startswith("[Generation failed"):
                    return {
                        "question": question,
                        "reference_context": reference_context,
                        "reference_answer": reference_answer,
                        "conversation_history": [],
                        "question_type": "simple",
                        "metadata": {
                            "question_type": "simple",
                            "seed_document_id": str(doc.id),
                            "topic": topic,
                            "collection_type": self.collection_type,
                            "targeted_components": ["Generator", "Retriever", "Router"]
                        }
                    }
                else:
                    raise Exception("Generation returned empty or error result")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Simple question generation attempt {attempt+1} failed: {e}. Retrying...")
                    await asyncio.sleep(1)  # Add delay between retries
                else:
                    logger.error(f"All {max_retries} attempts to generate simple question failed: {e}")
                    raise
    
    async def generate_complex_question(self, doc, doc_index, max_retries=3):
        """Generate a complex question with paraphrasing"""
        reference_context = format_document_content(doc)
        topic = get_topic_for_document(doc, self.top_topic_names, doc_index)
        
        # Prompt for complex question generation with paraphrasing
        question_prompt = f"""
        Based on the following document about SAP ABAP {self.collection_type}, create a complex,
        paraphrased question that requires deep understanding of the content.
        
        DOCUMENT:
        {reference_context}
        
        Create a complex question that:
        1. Is specific to this document's content
        2. Uses paraphrasing and advanced terminology
        3. Requires deep understanding of SAP ABAP concepts
        4. Tests ability to interpret code and technical details
        5. Would challenge an expert SAP ABAP developer
        6. IMPORTANT: Write only the question itself with no additional explanations, reasoning, or context
        
        Write only the QUESTION and nothing else:
        """
        
        for attempt in range(max_retries):
            try:
                question = await self.generator.generate_simple_prompt(question_prompt)
                
                # Cleanup any explanatory text
                question = self._cleanup_question_text(question)
                
                # Generate reference answer
                answer_prompt = f"""
                Answer the following complex question about SAP ABAP {self.collection_type} based on the provided documentation.
                Be comprehensive, accurate, and detailed in your response.
                
                DOCUMENT:
                {reference_context}
                
                QUESTION:
                {question}
                
                ANSWER:
                """
                
                reference_answer = await self.generator.generate_simple_prompt(answer_prompt)
                
                # Clean up any confidence indicators from the reference answer
                reference_answer = clean_question_format(reference_answer)
                
                # Only return if both question and answer generation succeeded
                if question and reference_answer and not question.startswith("[Generation failed"):
                    return {
                        "question": question,
                        "reference_context": reference_context,
                        "reference_answer": reference_answer,
                        "conversation_history": [],
                        "question_type": "complex",
                        "metadata": {
                            "question_type": "complex",
                            "seed_document_id": str(doc.id),
                            "topic": topic,
                            "collection_type": self.collection_type,
                            "targeted_components": ["Generator"]
                        }
                    }
                else:
                    raise Exception("Generation returned empty or error result")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Complex question generation attempt {attempt+1} failed: {e}. Retrying...")
                    await asyncio.sleep(1)
                else:
                    logger.error(f"All {max_retries} attempts to generate complex question failed: {e}")
                    raise
    
    async def generate_distracting_question(self, doc, doc_index, max_retries=3):
        """Generate a question with distracting elements"""
        reference_context = format_document_content(doc)
        topic = get_topic_for_document(doc, self.top_topic_names, doc_index)
        
        # Get another random document for distraction
        if len(self.document_pool) > 1:
            distractor_docs = [d for d in self.document_pool if d.id != doc.id]
            distractor_doc = random.choice(distractor_docs)
            distractor_title = distractor_doc.payload.get("title", "")
        else:
            # Fallback if no other document available
            distractor_title = "Another ABAP component"
        
        # Prompt for distracting question
        question_prompt = f"""
        Based on the following document about SAP ABAP {self.collection_type}, create a question that includes
        distracting information but still focuses on the main document content.
        
        DOCUMENT:
        {reference_context}
        
        DISTRACTOR:
        {distractor_title}
        
        Create a question that:
        1. Includes a mention of the distractor ("{distractor_title}") in a way that's irrelevant to the main question
        2. Still asks specifically about information from the main document
        3. Example template: "I was looking at {distractor_title}, but I need to know [something about the main document]"
        4. IMPORTANT: Write only the question itself with no additional explanations, reasoning, or context
        
        Write only the QUESTION and nothing else:
        """
        
        for attempt in range(max_retries):
            try:
                question = await self.generator.generate_simple_prompt(question_prompt)
                
                # Cleanup any explanatory text
                question = self._cleanup_question_text(question)
                
                # Generate reference answer
                answer_prompt = f"""
                Answer the following question about SAP ABAP {self.collection_type} based on the provided documentation.
                Focus only on answering the main question and ignore any distracting elements.
                
                DOCUMENT:
                {reference_context}
                
                QUESTION:
                {question}
                
                ANSWER:
                """
                
                reference_answer = await self.generator.generate_simple_prompt(answer_prompt)
                
                # Clean up any confidence indicators from the reference answer
                reference_answer = clean_question_format(reference_answer)
                
                # Only return if both question and answer generation succeeded
                if question and reference_answer and not question.startswith("[Generation failed"):
                    return {
                        "question": question,
                        "reference_context": reference_context,
                        "reference_answer": reference_answer,
                        "conversation_history": [],
                        "question_type": "distracting",
                        "metadata": {
                            "question_type": "distracting",
                            "seed_document_id": str(doc.id),
                            "topic": topic,
                            "collection_type": self.collection_type,
                            "distractor": distractor_title,
                            "targeted_components": ["Generator", "Retriever", "Rewriter"]
                        }
                    }
                else:
                    raise Exception("Generation returned empty or error result")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Distracting question generation attempt {attempt+1} failed: {e}. Retrying...")
                    await asyncio.sleep(1)
                else:
                    logger.error(f"All {max_retries} attempts to generate distracting question failed: {e}")
                    raise
    
    async def generate_situational_question(self, doc, doc_index, max_retries=3):
        """Generate a question with situational context"""
        reference_context = format_document_content(doc)
        topic = get_topic_for_document(doc, self.top_topic_names, doc_index)
        
        # Create possible situational contexts
        situations = [
            "I'm a developer new to SAP ABAP and",
            "Our project is facing an issue with performance and",
            "As part of a code review, I need to understand",
            "When developing an SAP module,",
            "During troubleshooting an application issue,",
            "In a migration project from legacy systems,",
            "While optimizing our ABAP codebase,"
        ]
        
        situation = random.choice(situations)
        
        # Prompt for situational question
        question_prompt = f"""
        Based on the following document about SAP ABAP {self.collection_type}, create a question that includes
        user context/situation but still focuses on the document content.
        
        DOCUMENT:
        {reference_context}
        
        SITUATION:
        {situation}
        
        Create a question that:
        1. Starts with the situation: "{situation}"
        2. Then asks specifically about information from the document
        3. Makes the question relevant to the user's situation
        4. IMPORTANT: Write only the question itself with no additional explanations, reasoning, or context
        
        Write only the QUESTION and nothing else:
        """
        
        for attempt in range(max_retries):
            try:
                question = await self.generator.generate_simple_prompt(question_prompt)
                
                # Cleanup any explanatory text
                question = self._cleanup_question_text(question)
                
                # Generate reference answer
                answer_prompt = f"""
                Answer the following situational question about SAP ABAP {self.collection_type} based on the provided documentation.
                Consider the user's context in your answer, but ensure the information is accurate based on the documentation.
                
                DOCUMENT:
                {reference_context}
                
                QUESTION:
                {question}
                
                ANSWER:
                """
                
                reference_answer = await self.generator.generate_simple_prompt(answer_prompt)
                
                # Clean up any confidence indicators from the reference answer
                reference_answer = clean_question_format(reference_answer)
                
                # Only return if both question and answer generation succeeded
                if question and reference_answer and not question.startswith("[Generation failed"):
                    return {
                        "question": question,
                        "reference_context": reference_context,
                        "reference_answer": reference_answer,
                        "conversation_history": [],
                        "question_type": "situational",
                        "metadata": {
                            "question_type": "situational",
                            "seed_document_id": str(doc.id),
                            "topic": topic,
                            "collection_type": self.collection_type,
                            "situation": situation,
                            "targeted_components": ["Generator"]
                        }
                    }
                else:
                    raise Exception("Generation returned empty or error result")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Situational question generation attempt {attempt+1} failed: {e}. Retrying...")
                    await asyncio.sleep(1)
                else:
                    logger.error(f"All {max_retries} attempts to generate situational question failed: {e}")
                    raise
    
    async def generate_double_question(self, doc, doc_index, max_retries=3):
        """Generate a double question asking for two pieces of information"""
        reference_context = format_document_content(doc)
        topic = get_topic_for_document(doc, self.top_topic_names, doc_index)
        
        # Prompt for double question
        question_prompt = f"""
        Based on the following document about SAP ABAP {self.collection_type}, create a question that asks
        for two distinct pieces of information from the document.
        
        DOCUMENT:
        {reference_context}
        
        Create a double question that:
        1. Asks for two distinct pieces of information from the document
        2. Connects the two parts with "and" or similar conjunction
        3. Both parts should be answerable from the document
        4. Example format: "What is X and how does Y work in this context?"
        5. IMPORTANT: Write only the question itself with no additional explanations, reasoning, or context
        
        Write only the QUESTION and nothing else:
        """
        
        for attempt in range(max_retries):
            try:
                question = await self.generator.generate_simple_prompt(question_prompt)
                
                # Cleanup any explanatory text
                question = self._cleanup_question_text(question)
                
                # Generate reference answer
                answer_prompt = f"""
                Answer the following double question about SAP ABAP {self.collection_type} based on the provided documentation.
                Make sure to address both parts of the question.
                
                DOCUMENT:
                {reference_context}
                
                QUESTION:
                {question}
                
                ANSWER:
                """
                
                reference_answer = await self.generator.generate_simple_prompt(answer_prompt)
                
                # Clean up any confidence indicators from the reference answer
                reference_answer = clean_question_format(reference_answer)
                
                # Only return if both question and answer generation succeeded
                if question and reference_answer and not question.startswith("[Generation failed"):
                    return {
                        "question": question,
                        "reference_context": reference_context,
                        "reference_answer": reference_answer,
                        "conversation_history": [],
                        "question_type": "double",
                        "metadata": {
                            "question_type": "double",
                            "seed_document_id": str(doc.id),
                            "topic": topic,
                            "collection_type": self.collection_type,
                            "targeted_components": ["Generator", "Rewriter"]
                        }
                    }
                else:
                    raise Exception("Generation returned empty or error result")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Double question generation attempt {attempt+1} failed: {e}. Retrying...")
                    await asyncio.sleep(1)
                else:
                    logger.error(f"All {max_retries} attempts to generate double question failed: {e}")
                    raise
    
    async def generate_conversational_questions(self, doc, doc_index, max_retries=3):
        """Generate a pair of questions forming a conversation"""
        reference_context = format_document_content(doc)
        topic = get_topic_for_document(doc, self.top_topic_names, doc_index)
        
        for attempt in range(max_retries):
            try:
                # Prompt for context-setting first question
                context_prompt = f"""
                Based on the following document about SAP ABAP {self.collection_type}, create an initial question
                that sets up context for a follow-up question.
                
                DOCUMENT:
                {reference_context}
                
                Create an initial context question that:
                1. Asks about the general topic or area covered by the document
                2. Doesn't ask for specific details yet
                3. Sets up context for a more specific follow-up question
                4. IMPORTANT: Write only the question itself with no additional explanations, reasoning, or context
                
                Write only the INITIAL QUESTION and nothing else:
                """
                
                initial_question = await self.generator.generate_simple_prompt(context_prompt)
                
                # Cleanup any explanatory text
                initial_question = self._cleanup_question_text(initial_question)
                
                # Generate answer to initial question
                initial_answer_prompt = f"""
                Answer the following initial question about SAP ABAP {self.collection_type} based on the provided documentation.
                Provide a general overview without going into too much detail.
                
                DOCUMENT:
                {reference_context}
                
                QUESTION:
                {initial_question}
                
                ANSWER:
                """
                
                initial_answer = await self.generator.generate_simple_prompt(initial_answer_prompt)
                initial_answer = clean_question_format(initial_answer)
                
                # Prompt for follow-up question
                followup_prompt = f"""
                Based on the previous question and answer, create a follow-up question that asks for more specific details.
                
                DOCUMENT:
                {reference_context}
                
                PREVIOUS QUESTION:
                {initial_question}
                
                PREVIOUS ANSWER:
                {initial_answer}
                
                Create a follow-up question that:
                1. Refers to the previous question/answer implicitly or explicitly
                2. Asks for specific details from the document
                3. Doesn't repeat information from the initial question
                4. Could reasonably be asked by someone who read the first answer
                5. IMPORTANT: Write only the question itself with no additional explanations, reasoning, or context
                
                Write only the FOLLOW-UP QUESTION and nothing else:
                """
                
                followup_question = await self.generator.generate_simple_prompt(followup_prompt)
                
                # Cleanup any explanatory text
                followup_question = self._cleanup_question_text(followup_question)
                
                # Generate answer to follow-up question
                followup_answer_prompt = f"""
                Answer the following follow-up question based on the previous conversation and the documentation.
                
                DOCUMENT:
                {reference_context}
                
                PREVIOUS QUESTION:
                {initial_question}
                
                PREVIOUS ANSWER:
                {initial_answer}
                
                FOLLOW-UP QUESTION:
                {followup_question}
                
                ANSWER:
                """
                
                followup_answer = await self.generator.generate_simple_prompt(followup_answer_prompt)
                followup_answer = clean_question_format(followup_answer)
                
                # Verify all parts were generated successfully
                if (initial_question and initial_answer and followup_question and followup_answer and
                    not initial_question.startswith("[Generation failed") and
                    not followup_question.startswith("[Generation failed")):
                    
                    # Create conversation history
                    conversation_history = [
                        {"role": "user", "content": initial_question},
                        {"role": "assistant", "content": initial_answer}
                    ]
                    
                    return {
                        "question": followup_question,
                        "reference_context": reference_context,
                        "reference_answer": followup_answer,
                        "conversation_history": conversation_history,
                        "question_type": "conversational",
                        "metadata": {
                            "question_type": "conversational",
                            "seed_document_id": str(doc.id),
                            "topic": topic,
                            "collection_type": self.collection_type,
                            "initial_question": initial_question,
                            "targeted_components": ["Rewriter"]
                        }
                    }
                else:
                    raise Exception("One or more conversational elements failed to generate properly")
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Conversational question generation attempt {attempt+1} failed: {e}. Retrying...")
                    await asyncio.sleep(1)
                else:
                    logger.error(f"All {max_retries} attempts to generate conversational question failed: {e}")
                    raise
def is_question_too_similar(new_question, existing_questions, similarity_threshold=0.7):
    """Check if a question is too similar to existing questions"""
    
    
    new_question = new_question.lower().strip()
    for existing in existing_questions:
        existing = existing.lower().strip()
        similarity = SequenceMatcher(None, new_question, existing).ratio()
        if similarity > similarity_threshold:
            return True
    return False
# Function to process a single question type
async def process_question_type(question_type, question_generator, full_doc_pool, 
                               questions_per_type, top_topics, all_previous_questions):
    """
    Process a single question type to ensure we get the required number of unique questions
    
    Args:
        question_type: Type of question to generate
        question_generator: Initialized question generator
        full_doc_pool: Full pool of documents to use
        questions_per_type: Number of questions to generate for this type
        top_topics: List of top topics
        all_previous_questions: List of all questions generated so far, for uniqueness check
        
    Returns:
        List of generated questions of this type
    """
    logger.info(f"Generating {questions_per_type} questions of type: {question_type}")
    
    # We'll keep generating until we have enough questions
    questions_of_type = []
    attempts = 0
    max_attempts = questions_per_type * 5  # Increased to allow for rejections due to similarity
    
    # For tracking progress
    with tqdm(total=questions_per_type, desc=f"  {question_type}", leave=False) as q_pbar:
        while len(questions_of_type) < questions_per_type and attempts < max_attempts:
            # Select a random document from the pool
            doc = random.choice(full_doc_pool)
            doc_index = attempts % len(full_doc_pool)
            
            try:
                # Generate question based on type
                if question_type == "simple":
                    question_data = await question_generator.generate_simple_question(doc, doc_index)
                elif question_type == "complex":
                    question_data = await question_generator.generate_complex_question(doc, doc_index)
                elif question_type == "distracting":
                    question_data = await question_generator.generate_distracting_question(doc, doc_index)
                elif question_type == "situational":
                    question_data = await question_generator.generate_situational_question(doc, doc_index)
                elif question_type == "double":
                    question_data = await question_generator.generate_double_question(doc, doc_index)
                elif question_type == "conversational":
                    question_data = await question_generator.generate_conversational_questions(doc, doc_index)
                else:
                    logger.warning(f"Unknown question type: {question_type}")
                    attempts += 1
                    continue
                
                question = question_data["question"]
                
                # Check if this question is too similar to any existing question
                is_too_similar = False
                for existing_question in all_previous_questions:
                    # Use SequenceMatcher to check similarity
                    similarity = SequenceMatcher(None, 
                                               question.lower().strip(), 
                                               existing_question.lower().strip()).ratio()
                    if similarity > 0.7:  # Threshold for similarity detection
                        is_too_similar = True
                        logger.warning(f"Rejected similar question: {question[:50]}... (similarity: {similarity:.2f})")
                        break
                
                if is_too_similar:
                    attempts += 1
                    await asyncio.sleep(0.2)  # Small delay before retrying
                    continue
                
                # Add the question data
                questions_of_type.append(question_data)
                all_previous_questions.append(question)  # Track the question text for uniqueness checking
                q_pbar.update(1)
                
            except Exception as e:
                logger.error(f"Error generating {question_type} question: {e}")
                # Add a small delay to avoid overwhelming the API
                await asyncio.sleep(0.5)
            
            attempts += 1
            
            # If we're struggling to generate enough questions, log a warning
            if attempts > questions_per_type * 3 and len(questions_of_type) < questions_per_type * 0.5:
                logger.warning(f"Having difficulty generating unique {question_type} questions. " 
                               f"Generated {len(questions_of_type)}/{questions_per_type} after {attempts} attempts.")
    
    # Log the final result
    logger.info(f"Generated {len(questions_of_type)}/{questions_per_type} unique {question_type} questions after {attempts} attempts")
    
    return questions_of_type

# Function to create a comprehensive evaluation dataset
async def create_raget_evaluation_dataset(collection_type=DEFAULT_COLLECTION_TYPE, 
                                         questions_per_type=20):
    """
    Create a comprehensive evaluation dataset with all RAGET question types
    
    Args:
        collection_type: Type of collection ('classes' or 'reports')
        questions_per_type: Number of questions to generate per type
        
    Returns:
        List of question data
    """
    logger.info(f"Creating RAGET evaluation dataset for {collection_type}")
    logger.info(f"Target: {questions_per_type} questions per type")
    
    # Get document pool - increased size to ensure enough quality docs
    doc_pool_size = questions_per_type * 10  # 10x the needed questions
    doc_pool, top_topics = get_document_pool(collection_type, doc_pool_size)
    
    if len(doc_pool) < questions_per_type * 6:
        logger.warning(f"Document pool smaller than ideal: {len(doc_pool)} < {questions_per_type * 6}")
    
    # Make a full copy of the doc pool for each question type
    full_doc_pool = doc_pool.copy()
    
    # Initialize question generator
    question_generator = RAGETQuestionGenerator(collection_type, full_doc_pool, top_topics)
    
    # Question types to generate
    question_types = [
        "simple", 
        "complex", 
        "distracting", 
        "situational", 
        "double", 
        "conversational"
    ]
    
    # Generate questions of each type - process types in parallel
    all_questions = []
    all_question_texts = []  # For tracking uniqueness
    
    # Process each question type sequentially to ensure uniqueness
    with tqdm(total=len(question_types), desc=f"Generating {collection_type} question types") as type_pbar:
        for q_type in question_types:
            type_pbar.set_description(f"Generating {q_type} questions")
            
            # Process this question type
            questions_of_type = await process_question_type(
                q_type, 
                question_generator, 
                full_doc_pool, 
                questions_per_type, 
                top_topics,
                all_question_texts  # Pass in all previously generated questions
            )
            
            all_questions.extend(questions_of_type)
            all_question_texts.extend([q["question"] for q in questions_of_type])
            type_pbar.update(1)
    
    total_questions = len(all_questions)
    logger.info(f"Generated total of {total_questions} questions across all types")
    
    # Show breakdown of question types
    question_counts = {}
    for q_type in question_types:
        count = len([q for q in all_questions if q["question_type"] == q_type])
        question_counts[q_type] = count
    
    logger.info("Question type breakdown:")
    for q_type, count in question_counts.items():
        logger.info(f"  - {q_type}: {count}/{questions_per_type} ({count/total_questions*100:.1f}%)")
    
    return all_questions

# Modified save_dataset function to include only required fields
# Modified save_dataset function to include only required fields
def save_dataset(dataset, base_filename):
    """Save dataset in both JSON and CSV formats using a specified directory"""
    if not dataset:
        logger.error("No dataset to save")
        return
    
    # Set the output directory
    output_dir = "/home/user/Desktop/RAG_pipeline_enhanced_conversational_claude/pair_wise_test_res"
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n==== SAVING FILES ====")
    print(f"Output directory: {output_dir}")
    
    # Select the fields to keep
    simplified_dataset = []
    for item in dataset:
        # Create a simplified version with only the required fields
        simplified_item = {
            "question": item["question"],
            "reference_context": item["reference_context"],
            "reference_answer": item["reference_answer"],
            "conversation_history": item["conversation_history"],
            "question_type": item["question_type"]
        }
        simplified_dataset.append(simplified_item)
    
    # Save JSON file
    json_path = os.path.join(output_dir, f"{base_filename}.json")
    with open(json_path, 'w') as f:
        json.dump(simplified_dataset, f, indent=2)
    print(f"✅ Saved JSON file: {json_path}")
    
    # Save CSV file
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    # Convert to DataFrame for CSV
    df = pd.DataFrame(simplified_dataset)
    
    # Convert complex columns to string representation
    df['conversation_history'] = df['conversation_history'].apply(json.dumps)
    
    # Save to CSV
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV file: {csv_path}")
    
    # List saved files with sizes
    json_size = os.path.getsize(json_path) / 1024  # Size in KB
    csv_size = os.path.getsize(csv_path) / 1024  # Size in KB
    print(f"\n{os.path.basename(json_path)}: {json_size:.2f} KB")
    print(f"{os.path.basename(csv_path)}: {csv_size:.2f} KB")
    print("====================\n")

# Function to generate evaluation datasets for all collections
async def generate_all_raget_datasets(questions_per_type=15):
    """
    Generate RAGET evaluation datasets for all collections
    
    Args:
        questions_per_type: Number of questions to generate per type
    """
    # Calculate total questions
    total_collections = len(COLLECTIONS.keys())
    total_question_types = 6  # simple, complex, distracting, situational, double, conversational
    total_questions = total_collections * total_question_types * questions_per_type
    
    print(f"\n===== RAGET DATASET GENERATION =====")
    print(f"Generating {questions_per_type} questions per type")
    print(f"Question types: simple, complex, distracting, situational, double, conversational")
    print(f"Collections: {', '.join(COLLECTIONS.keys())}")
    print(f"Total questions to generate: {total_questions}")
    print(f"=====================================\n")
    
    # Generate datasets for each collection sequentially
    with tqdm(total=len(COLLECTIONS), desc="Collections") as pbar:
        for collection_type in COLLECTIONS.keys():
            # Set base filename
            base_filename = f"raget_evaluation_dataset_10_{collection_type}"
            
            pbar.set_description(f"Collection: {collection_type}")
            logger.info(f"Generating RAGET dataset for {collection_type}...")
            
            try:
                # Create comprehensive RAGET evaluation dataset
                dataset = await create_raget_evaluation_dataset(
                    collection_type=collection_type,
                    questions_per_type=questions_per_type
                )
                
                # Save dataset in both formats
                save_dataset(dataset, base_filename)
                
                logger.info(f"Completed RAGET dataset generation for {collection_type}")
                pbar.update(1)
                
            except Exception as e:
                logger.error(f"Failed to generate RAGET dataset for {collection_type}: {e}")
                logger.error(f"Error details: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    print(f"\n✅ RAGET dataset generation completed!")
    print(f"Generated {total_questions} questions across {total_collections} collections")

# Main execution
if __name__ == "__main__":
    try:
        # Generate RAGET evaluation datasets for all collections
        print("\n🚀 Starting RAGET dataset generation")
        asyncio.run(generate_all_raget_datasets(questions_per_type=30))  # Default set to 40 questions per type
        
        print("\n✅ Script execution completed successfully")
    except Exception as e:
        print(f"\n❌ Script execution failed: {e}")
        import traceback
        traceback.print_exc()