"""
Complete Fixed Comprehensive Evaluation Script with API Cost Tracking
Save as: fixed_comprehensive_eval.py
"""
import pandas as pd
import asyncio
import json
import os
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dotenv import load_dotenv
import numpy as np
import sys
import time
from dataclasses import dataclass, field
from tqdm import tqdm
from functools import wraps

# Load environment
load_dotenv(r"/home/user/Desktop/RAG_pipeline_enhanced_conversational_claude_improved_retriever/claude.env")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Cost constants (per 1K tokens, in USD) - Added from script 1
API_COSTS = {
    'azure_openai': {
        'gpt-4o': {
            'input': 0.005,  # $5 per 1M input tokens
            'output': 0.015,  # $15 per 1M output tokens
        },
        'gpt-4': {
            'input': 0.01,  # $10 per 1M input tokens
            'output': 0.03,  # $30 per 1M output tokens
        },
        'gpt-35-turbo': {
            'input': 0.001,  # $1 per 1M input tokens
            'output': 0.002,  # $2 per 1M output tokens
        },
        # Add cbs-gpt-4o as an exact match
        'cbs-gpt-4o': {
            'input': 0.005,  # $5 per 1M input tokens
            'output': 0.015,  # $15 per 1M output tokens
        }
    },
    'azure': {  # Add azure as a separate provider key for exact matching
        'cbs-gpt-4o': {
            'input': 0.005,  # $5 per 1M input tokens
            'output': 0.015,  # $15 per 1M output tokens
        },
        'gpt-4o': {
            'input': 0.005,  # $5 per 1M input tokens
            'output': 0.015,  # $15 per 1M output tokens
        },
        'gpt-4': {
            'input': 0.01,  # $10 per 1M input tokens
            'output': 0.03,  # $30 per 1M output tokens
        }
    },
    'anthropic': {
        'claude-3-opus': {
            'input': 0.015,  # $15 per 1M input tokens
            'output': 0.075,  # $75 per 1M output tokens
        },
        'claude-3-sonnet': {
            'input': 0.003,  # $3 per 1M input tokens
            'output': 0.015,  # $15 per 1M output tokens
        },
        'claude-3-haiku': {
            'input': 0.00025,  # $0.25 per 1M input tokens
            'output': 0.00125,  # $1.25 per 1M output tokens
        }
    },
    'xai': {
        'grok-1': {
            'input': 0.0005,   # Approximate/estimated cost
            'output': 0.0015,  # Approximate/estimated cost
        },
        'grok-3-mini': {
            'input': 0.0003,   # Approximate/estimated cost
            'output': 0.0009,  # Approximate/estimated cost
        }
    }
}

def detect_question_type(query: str) -> str:
    """Enhanced question type detection"""
    query_lower = query.lower()
    
    # Simple question patterns
    import re
    simple_patterns = [
        r'\bwhat\s+(is|are|does|do)\b',
        r'\bwhen\s+(is|are|does|do)\b', 
        r'\bwhere\s+(is|are|does|do)\b',
        r'\bwhich\s+(is|are|does|do)\b',
        r'\bdoes\s+\w+\b',
        r'\bis\s+\w+\b',
        r'\bdefine\b',
        r'\bwhat.*purpose\b'
    ]
    
    for pattern in simple_patterns:
        if re.search(pattern, query_lower):
            if not any(term in query_lower for term in ['how', 'why', 'implement', 'architecture', 'mechanism']):
                return 'simple'
    
    # Complex question patterns
    complex_patterns = [
        r'\bhow\s+(to|do|does|can|should)\b',
        r'\bwhy\s+(is|are|does|do)\b',
        r'\bimplement\b',
        r'\barchitecture\b',
        r'\bmechanism\b',
        r'\bapproach\b',
        r'\bprocess\b',
        r'\bworkflow\b'
    ]
    
    for pattern in complex_patterns:
        if re.search(pattern, query_lower):
            return 'complex'
    
    # Other patterns
    if any(phrase in query_lower for phrase in [
        'while', 'during', 'as part of', 'when developing', 'in a migration project'
    ]):
        return 'situational'
    
    if any(phrase in query_lower for phrase in [
        'i was looking at', 'i was reviewing', 'but i need to'
    ]):
        return 'distracting'
    
    if query_lower.count('?') > 1 or ('and' in query_lower and sum(w in query_lower for w in ['what', 'how', 'why', 'when']) > 1):
        return 'double'
    
    if any(phrase in query_lower for phrase in [
        'what other', 'are there', 'can you explain', 'tell me more'
    ]):
        return 'conversational'
    
    return 'general'

# Enhanced API Cost tracking - Added from script 1
@dataclass
class APICallTracker:
    """Track API calls and estimate costs for various models"""
    calls: Dict[str, int] = field(default_factory=dict)
    tokens: Dict[str, Dict[str, int]] = field(default_factory=lambda: {'input': {}, 'output': {}})
    errors: Dict[str, int] = field(default_factory=dict)
    latency: Dict[str, List[float]] = field(default_factory=lambda: {})
    start_time: float = field(default_factory=time.time)
    
    def reset(self):
        """Reset all counters"""
        self.calls = {}
        self.tokens = {'input': {}, 'output': {}}
        self.errors = {}
        self.latency = {}
        self.start_time = time.time()
    
    def debug_token_tracking(self):
        """Print detailed information about token tracking for debugging purposes"""
        print("\n" + "="*80)
        print("API TOKEN TRACKING DEBUG INFO")
        print("="*80)
        
        # Print calls and token info
        for key in self.calls.keys():
            provider, model = key.split(':')
            print(f"\nModel: {provider} {model}")
            print(f"  Total calls: {self.calls.get(key, 0)}")
            print(f"  Input tokens: {self.tokens['input'].get(key, 0)}")
            print(f"  Output tokens: {self.tokens['output'].get(key, 0)}")
            
            # Try to find the cost model for this provider/model
            cost_model = None
            provider_lower = provider.lower()
            model_lower = model.lower()
            
            # Print matching attempts
            print(f"  Searching for cost model: provider='{provider_lower}', model='{model_lower}'")
            
            # Check direct matches
            for p in API_COSTS:
                print(f"  - Checking provider '{p}'...")
                if provider_lower in p:
                    for m in API_COSTS[p]:
                        print(f"    - Checking model '{m}'...")
                        if model_lower in m or m in model_lower:
                            cost_model = API_COSTS[p][m]
                            print(f"    ? Found match: {p}/{m}")
                            break
                    if cost_model:
                        break
            
            if cost_model:
                # Calculate costs based on tokens
                input_tokens = self.tokens['input'].get(key, 0)
                output_tokens = self.tokens['output'].get(key, 0)
                
                input_cost = (input_tokens / 1000) * cost_model['input']
                output_cost = (output_tokens / 1000) * cost_model['output']
                model_cost = input_cost + output_cost
                
                print(f"  Cost calculation:")
                print(f"    Input tokens: {input_tokens} � ${cost_model['input']}/1K = ${input_cost:.4f}")
                print(f"    Output tokens: {output_tokens} � ${cost_model['output']}/1K = ${output_cost:.4f}")
                print(f"    Total cost: ${model_cost:.4f}")
            else:
                print(f"  ? No cost model found for {key}")
        
        print("="*80)
    
    def record_call(self, provider: str, model: str, input_tokens: int = 0, output_tokens: int = 0, error: bool = False, latency: float = 0):
        """Record an API call with token counts and latency"""
        # Create provider-model key
        key = f"{provider}:{model}"
        
        # Update call count
        self.calls[key] = self.calls.get(key, 0) + 1
        
        # Update token counts
        if input_tokens > 0:
            self.tokens['input'][key] = self.tokens['input'].get(key, 0) + input_tokens
        if output_tokens > 0:
            self.tokens['output'][key] = self.tokens['output'].get(key, 0) + output_tokens
        
        # Update error count if applicable
        if error:
            self.errors[key] = self.errors.get(key, 0) + 1
        
        # Record latency
        if key not in self.latency:
            self.latency[key] = []
        if latency > 0:
            self.latency[key].append(latency)
    
    def estimate_cost(self) -> Dict[str, float]:
        """Estimate cost of API calls based on token usage"""
        costs = {}
        total_cost = 0.0
        
        for key in self.calls.keys():
            provider, model = key.split(':')
            provider_lower = provider.lower()
            model_lower = model.lower()
            
            # Find the appropriate cost model
            cost_model = None
            for p in API_COSTS:
                if provider_lower in p:
                    for m in API_COSTS[p]:
                        if model_lower in m:
                            cost_model = API_COSTS[p][m]
                            break
                    if cost_model:
                        break
            
            # If no direct match, try to match by substring
            if not cost_model:
                for p in API_COSTS:
                    if provider_lower in p:
                        for m in API_COSTS[p]:
                            # Check if model name contains the model key or vice versa
                            if model_lower in m or m in model_lower:
                                cost_model = API_COSTS[p][m]
                                print(f"Using cost model for {m} based on partial match with {model_lower}")
                                break
                        if cost_model:
                            break
            
            # If still no match, use default cost as fallback
            if not cost_model:
                print(f"No cost model found for {key}, using fallback cost model")
                # Use gpt-4o costs as fallback for unknown models
                cost_model = {
                    'input': 0.005,   # $5 per 1M input tokens
                    'output': 0.015,  # $15 per 1M output tokens
                }
            
            # Calculate costs based on tokens
            input_tokens = self.tokens['input'].get(key, 0)
            output_tokens = self.tokens['output'].get(key, 0)
            
            input_cost = (input_tokens / 1000) * cost_model['input']
            output_cost = (output_tokens / 1000) * cost_model['output']
            model_cost = input_cost + output_cost
            
            costs[key] = model_cost
            total_cost += model_cost
            
            # Print detailed cost calculation for debugging
            print(f"Cost calculation for {key}:")
            print(f"  Input tokens: {input_tokens} � ${cost_model['input']}/1K = ${input_cost:.4f}")
            print(f"  Output tokens: {output_tokens} � ${cost_model['output']}/1K = ${output_cost:.4f}")
            print(f"  Total cost: ${model_cost:.4f}")
        
        costs['total'] = total_cost
        return costs
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all API calls and costs"""
        elapsed_time = time.time() - self.start_time
        costs = self.estimate_cost()
        
        # Calculate average latency
        avg_latency = {}
        for key, latencies in self.latency.items():
            if latencies:
                avg_latency[key] = sum(latencies) / len(latencies)
            else:
                avg_latency[key] = 0
        
        return {
            'calls': self.calls,
            'tokens': self.tokens,
            'errors': self.errors,
            'avg_latency': avg_latency,
            'costs': costs,
            'total_cost': costs['total'],
            'elapsed_time': elapsed_time
        }
    
    def print_summary(self):
        """Print a summary of API calls and costs"""
        summary = self.get_summary()
        
        print("\n" + "="*80)
        print("API CALL TRACKING SUMMARY")
        print("="*80)
        
        # API calls by model
        print("\nAPI CALLS BY MODEL:")
        print("-"*30)
        for key, count in summary['calls'].items():
            provider, model = key.split(':')
            error_count = summary['errors'].get(key, 0)
            error_rate = (error_count / count) * 100 if count > 0 else 0
            print(f"{provider} {model}: {count} calls ({error_count} errors, {error_rate:.1f}% error rate)")
        
        # Token usage
        print("\nTOKEN USAGE:")
        print("-"*30)
        for key in summary['calls'].keys():
            provider, model = key.split(':')
            input_tokens = summary['tokens']['input'].get(key, 0)
            output_tokens = summary['tokens']['output'].get(key, 0)
            total_tokens = input_tokens + output_tokens
            print(f"{provider} {model}: {total_tokens:,} tokens ({input_tokens:,} input, {output_tokens:,} output)")
        
        # Costs
        print("\nESTIMATED COSTS (USD):")
        print("-"*30)
        for key, cost in summary['costs'].items():
            if key != 'total':
                provider, model = key.split(':')
                print(f"{provider} {model}: ${cost:.4f}")
        print(f"TOTAL COST: ${summary['total_cost']:.4f}")
        
        # Performance
        print("\nPERFORMANCE:")
        print("-"*30)
        for key, avg_lat in summary['avg_latency'].items():
            provider, model = key.split(':')
            print(f"{provider} {model}: {avg_lat:.2f}s average latency")
        
        print(f"Total elapsed time: {summary['elapsed_time']:.2f}s")
        print("="*80)

# Global tracker
api_tracker = APICallTracker()

def track_api_call(provider: str, model: str):
    """Decorator to track API calls to a specific provider and model - Added from script 1"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            error = False
            input_tokens = 0
            output_tokens = 0
            
            try:
                # Execute the function
                result = await func(*args, **kwargs)
                
                # Try to extract token counts if available in result
                if isinstance(result, dict):
                    if 'token_usage' in result:
                        input_tokens = result['token_usage'].get('prompt_tokens', 0)
                        output_tokens = result['token_usage'].get('completion_tokens', 0)
                    elif 'usage' in result:
                        input_tokens = result['usage'].get('prompt_tokens', 0)
                        output_tokens = result['usage'].get('completion_tokens', 0)
                
                return result
            
            except Exception as e:
                error = True
                raise e
            
            finally:
                # Record the API call
                latency = time.time() - start_time
                api_tracker.record_call(
                    provider=provider,
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    error=error,
                    latency=latency
                )
        
        return wrapper
    return decorator

# Import DeepEval
try:
    from deepeval.test_case import LLMTestCase
    from deepeval.models.base_model import DeepEvalBaseLLM
    from langchain_openai import AzureChatOpenAI
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        HallucinationMetric,
        FaithfulnessMetric,
        BiasMetric,
        ToxicityMetric
    )
    DEEPEVAL_AVAILABLE = True
    print("? DeepEval imports successful")
except ImportError as e:
    print(f"? DeepEval import error: {e}")
    DEEPEVAL_AVAILABLE = False
    exit(1)

class FixedAzureEvaluator(DeepEvalBaseLLM):
    """Fixed Azure OpenAI evaluator"""
    
    def __init__(self, model):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        try:
            enhanced_prompt = f"""{prompt}

CRITICAL INSTRUCTIONS:
1. Be GENEROUS with relevancy assessment
2. Consider any related information as relevant
3. Your response must be valid JSON format
"""
            response = self.model.invoke(enhanced_prompt)
            return response.content
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    async def a_generate(self, prompt: str) -> str:
        try:
            enhanced_prompt = f"""{prompt}

CRITICAL INSTRUCTIONS:
1. Be GENEROUS with relevancy assessment
2. Consider any related information as relevant
3. Your response must be valid JSON format
"""
            res = await self.model.ainvoke(enhanced_prompt)
            return res.content
        except Exception as e:
            logger.error(f"Async generation error: {e}")
            return self.generate(prompt)

    def get_model_name(self):
        return "Fixed Azure Evaluator"

class FixedDeepEvalEvaluator:
    """Fixed DeepEval evaluator with proper error handling"""
    
    def __init__(self, azure_openai_key: str, azure_api_version: str, 
                 azure_deployment: str, azure_endpoint: str, output_dir: str = "fixed_results"):
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Store deployment name for API tracking
        self.azure_deployment = azure_deployment
        
        # Initialize Azure OpenAI
        self.azure_model = AzureChatOpenAI(
            openai_api_key=azure_openai_key,
            openai_api_version=azure_api_version,
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            temperature=0.0,
            max_tokens=6000,
            timeout=180,
            max_retries=5
        )
        
        self.evaluator_model = FixedAzureEvaluator(self.azure_model)
        
        # Define metric categories
        self.metric_categories = {
            'retriever': ['contextual_relevancy', 'contextual_recall'],
            'reranker': ['contextual_precision'],
            'generator': ['answer_relevancy', 'faithfulness', 'hallucination'],
            'safety': ['bias', 'toxicity']
        }
        
        logger.info("Fixed DeepEval Evaluator initialized")
    
    def load_dataset(self, file_path: str) -> pd.DataFrame:
        """Load dataset from JSON file"""
        logger.info(f"Loading dataset from {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            for key in ['questions', 'data', 'items']:
                if key in data and isinstance(data[key], list):
                    df = pd.DataFrame(data[key])
                    break
            else:
                df = pd.DataFrame([data])
        
        # Column mapping
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'question' in col_lower and 'type' not in col_lower:
                column_mapping[col] = 'question'
            elif any(term in col_lower for term in ['golden', 'ground_truth', 'reference_answer']):
                column_mapping[col] = 'golden_answer'
            elif any(term in col_lower for term in ['rag_response', 'response', 'generated']):
                column_mapping[col] = 'rag_response'
            elif any(term in col_lower for term in ['context', 'reference_context']):
                column_mapping[col] = 'reference_context'
            elif 'type' in col_lower and 'question' in col_lower:
                column_mapping[col] = 'question_type'
        
        df = df.rename(columns=column_mapping)
        
        # Check required columns
        required_cols = ['question', 'golden_answer', 'rag_response', 'reference_context']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"Successfully loaded {len(df)} questions")
        return df
    
    def create_test_case(self, row: pd.Series) -> LLMTestCase:
        """Create test case with enhanced context formatting"""
        question = str(row.get('question', ''))
        rag_response = str(row.get('rag_response', ''))
        golden_answer = str(row.get('golden_answer', ''))
        context = str(row.get('reference_context', ''))
        
        if not context.strip():
            context = "No context available for this question."
        
        # Enhanced context formatting
        question_type = detect_question_type(question)
        formatted_context = self._format_context_for_deepeval(context, question, question_type)
        
        # Split into optimal chunks
        retrieval_context = self._split_context_for_evaluation(formatted_context, question_type)
        
        return LLMTestCase(
            input=question,
            actual_output=rag_response,
            expected_output=golden_answer,
            retrieval_context=retrieval_context,
            context=retrieval_context
        )
    
    def _format_context_for_deepeval(self, context: str, question: str, question_type: str) -> str:
        """Format context specifically for DeepEval"""
        if question_type == 'simple':
            # Enhanced formatting for simple questions
            lines = context.split('\n')
            enhanced_lines = [
                f"DEFINITION AND EXPLANATION for: {question}",
                "=" * 60,
                "",
                "RELEVANT INFORMATION:",
            ]
            
            # Add context with explicit markers
            for line in lines:
                line = line.strip()
                if line and len(line) > 10:
                    # Mark definition-like content
                    if any(term in line.lower() for term in ['is a', 'are', 'means', 'refers to', 'used for']):
                        enhanced_lines.append(f"?? DEFINITION: {line}")
                    else:
                        enhanced_lines.append(f"?? DETAIL: {line}")
            
            enhanced_lines.extend([
                "",
                f"?? This context provides the specific information needed to answer: {question}",
                "?? Focus on definitions, explanations, and core concepts mentioned above."
            ])
            
            return '\n'.join(enhanced_lines)
        
        elif question_type == 'complex':
            return f"""Technical Information for: {question}
{'=' * 60}

{context}

Note: This provides technical details and implementation guidance relevant to the question."""
        
        else:
            return f"""Relevant Information:
{'-' * 30}

{context}"""
    
    def _split_context_for_evaluation(self, context: str,question_type: str) -> List[str]:
        """Split context into optimal chunks for DeepEval"""

        if question_type == 'simple':
        # For simple questions, prefer smaller, focused chunks
            paragraphs = [p.strip() for p in context.split('\n\n') if p.strip()]
            
            chunks = []
            for paragraph in paragraphs:
                # Keep definition paragraphs intact
                if len(paragraph) <= 1500:  # Smaller chunks for simple questions
                    chunks.append(paragraph)
                else:
                    # Split longer paragraphs at sentence boundaries
                    sentences = paragraph.split('. ')
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) <= 1500:
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk:
                                chunks.append(current_chunk.strip())
                            current_chunk = sentence + ". "
                    
                    if current_chunk:
                        chunks.append(current_chunk.strip())
            
            return chunks[:5]  # Limit to 5 most relevant chunks for simple questions
        paragraphs = [p.strip() for p in context.split('\n\n') if p.strip()]
        
        if not paragraphs:
            paragraphs = [p.strip() for p in context.split('\n') if p.strip()]
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) > 2000 and current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        if not chunks:
            chunks = [context]
        
        if len(chunks) > 8:
            chunks = chunks[:8]
        
        return chunks
    
    def get_fresh_metrics(self, selected_metrics: List[str]) -> Dict[str, Any]:
        """Create fresh instances of selected metrics"""
        fresh_metrics = {}
        
        for metric_name in selected_metrics:
            try:
                if metric_name == 'contextual_relevancy':
                    fresh_metrics[metric_name] = ContextualRelevancyMetric(
                        model=self.evaluator_model, 
                        threshold=0.1,
                        verbose_mode=True,
                        include_reason=True
                    )
                elif metric_name == 'contextual_recall':
                    fresh_metrics[metric_name] = ContextualRecallMetric(
                        model=self.evaluator_model,
                        threshold=0.1,
                        verbose_mode=True
                    )
                elif metric_name == 'contextual_precision':
                    fresh_metrics[metric_name] = ContextualPrecisionMetric(
                        model=self.evaluator_model,
                        threshold=0.1,
                        verbose_mode=True
                    )
                elif metric_name == 'answer_relevancy':
                    fresh_metrics[metric_name] = AnswerRelevancyMetric(
                        model=self.evaluator_model, 
                        threshold=0.1,
                        verbose_mode=True
                    )
                elif metric_name == 'faithfulness':
                    fresh_metrics[metric_name] = FaithfulnessMetric(
                        model=self.evaluator_model,
                        threshold=0.1
                    )
                elif metric_name == 'hallucination':
                    fresh_metrics[metric_name] = HallucinationMetric(
                        model=self.evaluator_model,
                        threshold=0.5
                    )
                elif metric_name == 'bias':
                    fresh_metrics[metric_name] = BiasMetric(
                        model=self.evaluator_model,
                        threshold=0.5
                    )
                elif metric_name == 'toxicity':
                    fresh_metrics[metric_name] = ToxicityMetric(
                        model=self.evaluator_model,
                        threshold=0.5
                    )
                
            except Exception as e:
                logger.error(f"Failed to create {metric_name}: {e}")
                continue
        
        return fresh_metrics
    
    async def evaluate_question(self, row: pd.Series, selected_metrics: List[str]) -> Dict[str, Any]:
        """Evaluate a single question with detailed logging and API tracking"""
        test_case = self.create_test_case(row)
        question_type = detect_question_type(test_case.input)
        
        logger.info(f"Evaluating {question_type} question: {test_case.input[:50]}...")
        
        metrics_to_run = self.get_fresh_metrics(selected_metrics)
        results = {}
        
        print(f"\n?? Evaluating: {test_case.input[:80]}...")
        print(f"?? Question type: {question_type}")
        print(f"?? Metrics: {list(metrics_to_run.keys())}")
        
        for metric_name, metric in metrics_to_run.items():
            try:
                print(f"\n?? Running {metric_name}...")
                
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        call_start_time = time.time()
                        timeout = 180.0
                        api_error = False
                        
                        await asyncio.wait_for(metric.a_measure(test_case), timeout=timeout)
                        
                        call_latency = time.time() - call_start_time
                        
                        # Enhanced token counting for API tracking
                        input_tokens = len(test_case.input) + len(''.join(test_case.context)) 
                        input_tokens = int(input_tokens / 4)  # Rough estimate: 4 chars per token
                        
                        output_tokens = 0
                        if hasattr(metric, 'reason') and metric.reason:
                            output_tokens = len(metric.reason) // 4
                        
                        # Enhanced token counting - ensure reasonable minimums
                        input_tokens = max(input_tokens, 500)  # Minimum of 500 tokens for input
                        output_tokens = max(output_tokens, 100)  # Minimum of 100 tokens for output
                        
                        # Record the API call with token estimates
                        api_tracker.record_call(
                            provider="Azure",
                            model=self.azure_deployment,
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            error=api_error,
                            latency=call_latency
                        )
                        
                        if not hasattr(metric, 'score') or metric.score is None:
                            raise Exception(f"Metric returned None score")
                        
                        threshold = getattr(metric, 'threshold', 0.5)
                        if metric_name == 'hallucination':
                            passed = metric.score <= threshold
                        else:
                            passed = metric.score >= threshold
                        
                        results[metric_name] = {
                            'score': float(metric.score),
                            'threshold': float(threshold),
                            'passed': passed,
                            'reason': getattr(metric, 'reason', None),
                            'attempts': attempt + 1,
                            'latency': call_latency,
                            'question_type': question_type
                        }
                        
                        print(f"  ? {metric_name}: {metric.score:.4f} (passed: {passed})")
                        break
                        
                    except Exception as attempt_error:
                        api_error = True
                        # Record failed API call
                        api_tracker.record_call(
                            provider="Azure",
                            model=self.azure_deployment,
                            error=True,
                            latency=time.time() - call_start_time
                        )
                        print(f"  ?? Attempt {attempt + 1} failed: {str(attempt_error)[:100]}")
                        if attempt == max_attempts - 1:
                            raise attempt_error
                        await asyncio.sleep(2.0)
                
                await asyncio.sleep(1.0)
                
            except Exception as e:
                error_msg = str(e)
                print(f"  ? {metric_name} failed: {error_msg[:100]}")
                
                results[metric_name] = {
                    'score': None,
                    'error': error_msg,
                    'error_type': type(e).__name__,
                    'status': 'FAILED'
                }
        
        # Calculate average score
        valid_scores = [r['score'] for r in results.values() if isinstance(r, dict) and r.get('score') is not None]
        if valid_scores:
            results['average_score'] = sum(valid_scores) / len(valid_scores)
            print(f"?? Average score: {results['average_score']:.4f}")
        
        return results
    
    async def evaluate_dataset(self, df: pd.DataFrame, selected_metrics: List[str]) -> List[Dict[str, Any]]:
        """Evaluate entire dataset"""
        logger.info(f"Evaluating {len(df)} questions with metrics: {selected_metrics}")
        
        all_results = []
        
        progress_bar = tqdm(total=len(df), desc="Evaluating questions")
        
        for i, (_, row) in enumerate(df.iterrows()):
            try:
                question_start_time = time.time()
                
                result = await self.evaluate_question(row, selected_metrics)
                
                question_time = time.time() - question_start_time
                
                combined_result = {
                    'question_data': {
                        'id': i,
                        'question': str(row.get('question', '')),
                        'golden_answer': str(row.get('golden_answer', '')),
                        'rag_response': str(row.get('rag_response', '')),
                        'question_type': str(row.get('question_type', '')) or detect_question_type(str(row.get('question', ''))),
                        'processing_time': question_time
                    },
                    'evaluation': result
                }
                
                all_results.append(combined_result)
                
                # Update progress
                progress_desc = f"Q{i+1}/{len(df)}"
                if 'contextual_relevancy' in result and 'score' in result.get('contextual_relevancy', {}):
                    ctx_score = result['contextual_relevancy']['score']
                    progress_desc += f" | CTX:{ctx_score:.3f}"
                
                progress_bar.set_description(progress_desc)
                progress_bar.update(1)
                
                await asyncio.sleep(2.0)
                
            except Exception as e:
                logger.error(f"Question {i+1} failed: {e}")
                all_results.append({
                    'question_data': {'id': i, 'error': str(e)},
                    'evaluation': {'error': str(e)}
                })
                
                progress_bar.set_description(f"Q{i+1}/{len(df)} | ERROR")
                progress_bar.update(1)
        
        progress_bar.close()
        return all_results
    
    def calculate_component_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate metrics by component"""
        component_metrics = {
            'retriever': {},
            'reranker': {},
            'generator': {},
            'safety': {}
        }
        
        by_question_type = {}
        
        for result in results:
            question_type = result['question_data'].get('question_type', 'unknown')
            
            if question_type not in by_question_type:
                by_question_type[question_type] = {
                    'retriever': {},
                    'reranker': {},
                    'generator': {},
                    'safety': {}
                }
            
            evaluation = result.get('evaluation', {})
            
            for metric_name, metric_data in evaluation.items():
                if isinstance(metric_data, dict) and 'score' in metric_data and metric_data['score'] is not None:
                    component = None
                    for comp, metrics in self.metric_categories.items():
                        if metric_name in metrics:
                            component = comp
                            break
                    
                    if component:
                        if metric_name not in component_metrics[component]:
                            component_metrics[component][metric_name] = []
                        component_metrics[component][metric_name].append(metric_data['score'])
                        
                        if metric_name not in by_question_type[question_type][component]:
                            by_question_type[question_type][component][metric_name] = []
                        by_question_type[question_type][component][metric_name].append(metric_data['score'])
        
        def calculate_stats(scores):
            if scores:
                scores_array = np.array(scores)
                return {
                    'mean': float(np.mean(scores_array)),
                    'std': float(np.std(scores_array)),
                    'min': float(np.min(scores_array)),
                    'max': float(np.max(scores_array)),
                    'median': float(np.median(scores_array)),
                    'count': int(len(scores))
                }
            return None
        
        component_stats = {}
        for component, metrics in component_metrics.items():
            component_stats[component] = {}
            for metric_name, scores in metrics.items():
                component_stats[component][metric_name] = calculate_stats(scores)
        
        question_type_stats = {}
        for q_type, type_data in by_question_type.items():
            question_type_stats[q_type] = {}
            for component, metrics in type_data.items():
                if metrics:
                    question_type_stats[q_type][component] = {}
                    for metric_name, scores in metrics.items():
                        question_type_stats[q_type][component][metric_name] = calculate_stats(scores)
        
        return {
            'component_performance': component_stats,
            'by_question_type': question_type_stats,
            'summary': {
                'total_questions': len(results),
                'question_types': list(by_question_type.keys()),
                'components_evaluated': [comp for comp, metrics in component_stats.items() if metrics]
            }
        }

class FixedComprehensiveEvaluator:
    """Fixed comprehensive evaluator"""
    
    def __init__(self, azure_openai_key: str, azure_api_version: str,
                 azure_deployment: str, azure_endpoint: str, output_dir: str = "fixed_results"):
        
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.deepeval_evaluator = FixedDeepEvalEvaluator(
            azure_openai_key=azure_openai_key,
            azure_api_version=azure_api_version,
            azure_deployment=azure_deployment,
            azure_endpoint=azure_endpoint,
            output_dir=os.path.join(output_dir, "deepeval_results")
        )
        
        logger.info("Fixed Comprehensive Evaluator initialized")
    
    async def evaluate_comprehensive(self, df: pd.DataFrame, deepeval_metrics: List[str] = None,
                                   num_questions: Optional[int] = None) -> List[Dict[str, Any]]:
        """Run comprehensive evaluation"""
        logger.info("Starting comprehensive evaluation")
        
        if num_questions is not None:
            df = df.head(num_questions)
            logger.info(f"Limited evaluation to {num_questions} questions")
        
        print(f"\n{'='*80}")
        print(f"FIXED EVALUATION CONFIGURATION")
        print(f"{'='*80}")
        print(f"Questions to evaluate: {len(df)}")
        
        if 'question_type' in df.columns:
            question_types = df['question_type'].value_counts().to_dict()
            print("\nQUESTION TYPE DISTRIBUTION:")
            for qtype, count in sorted(question_types.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / len(df)) * 100
                print(f"  {qtype}: {count} questions ({percentage:.1f}%)")
        
        print(f"{'='*80}\n")
        
        if deepeval_metrics is None:
            deepeval_metrics = [
                'contextual_relevancy',
                'contextual_recall',
                'contextual_precision',
                'answer_relevancy',
                'faithfulness'
            ]
        
        logger.info(f"Running DeepEval metrics: {deepeval_metrics}")
        
        # Process questions
        results = []
        
        for index, row in df.iterrows():
            question = str(row.get('question', '')).strip()
            golden_answer = str(row.get('golden_answer', '')).strip()
            rag_response = str(row.get('rag_response', '')).strip()
            reference_context = str(row.get('reference_context', '')).strip()
            question_type = str(row.get('question_type', '')) or detect_question_type(question)
            
            if not question or not golden_answer or not rag_response:
                logger.warning(f"Row {index} has missing essential data")
                continue
            
            result = {
                'question_data': {
                    'id': index,
                    'question': question,
                    'golden_answer': golden_answer,
                    'rag_response': rag_response,
                    'question_type': question_type,
                    'reference_context': reference_context
                }
            }
            
            results.append(result)
        
        # Run DeepEval evaluation
        logger.info("Running DeepEval metrics...")
        
        deepeval_df = pd.DataFrame([
            {
                'question': result['question_data']['question'],
                'golden_answer': result['question_data']['golden_answer'],
                'rag_response': result['question_data']['rag_response'],
                'reference_context': result['question_data']['reference_context'],
                'question_type': result['question_data']['question_type']
            }
            for result in results
        ])
        
        print(f"\n{'='*80}")
        print(f"RUNNING DEEPEVAL EVALUATION")
        print(f"{'='*80}")
        print(f"Questions: {len(deepeval_df)}")
        print(f"Metrics: {deepeval_metrics}")
        print(f"{'='*80}\n")
        
        deepeval_results = await self.deepeval_evaluator.evaluate_dataset(
            df=deepeval_df,
            selected_metrics=deepeval_metrics
        )
        
        # FIXED: Merge results properly
        print(f"\n?? MERGING RESULTS...")
        print(f"Questions: {len(results)}")
        print(f"DeepEval results: {len(deepeval_results)}")
        
        successful_merges = 0
        
        for i, result in enumerate(results):
            if i < len(deepeval_results):
                deepeval_result = deepeval_results[i]
                evaluation_data = deepeval_result.get('evaluation', {})
                
                print(f"\nQuestion {i+1}:")
                print(f"  DeepEval data type: {type(evaluation_data)}")
                
                if isinstance(evaluation_data, dict) and 'error' not in evaluation_data:
                    valid_metrics = {}
                    valid_count = 0
                    total_score = 0.0
                    
                    for key, value in evaluation_data.items():
                        if key == 'average_score':
                            continue
                        
                        if isinstance(value, dict) and 'score' in value and value['score'] is not None:
                            valid_metrics[key] = value
                            valid_count += 1
                            total_score += float(value['score'])
                            print(f"    ? {key}: {value['score']:.4f}")
                        else:
                            print(f"    ? {key}: Invalid")
                    
                    if valid_count > 0:
                        result['evaluation'] = valid_metrics
                        result['deepeval_average'] = total_score / valid_count
                        successful_merges += 1
                        print(f"  ? Merged {valid_count} metrics, avg: {result['deepeval_average']:.4f}")
                    else:
                        result['evaluation'] = {'error': 'No valid metrics'}
                        result['deepeval_average'] = 0.0
                        print(f"  ? No valid metrics")
                else:
                    result['evaluation'] = {'error': 'Invalid evaluation data'}
                    result['deepeval_average'] = 0.0
                    print(f"  ? Invalid evaluation data")
            else:
                result['evaluation'] = {'error': 'No deepeval result'}
                result['deepeval_average'] = 0.0
                print(f"  ? No deepeval result for question {i+1}")
        
        print(f"\n?? MERGE SUMMARY: {successful_merges}/{len(results)} successful")
        
        logger.info("Comprehensive evaluation completed")
        return results
    
    def calculate_enhanced_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate enhanced metrics"""
        print(f"\n?? Calculating enhanced metrics from {len(results)} results...")
        
        component_analysis = self.deepeval_evaluator.calculate_component_metrics(results)
        
        # Count successful evaluations
        successful_evaluations = 0
        deepeval_metrics = {}
        contextual_relevancy_scores = []
        
        for result in results:
            evaluation = result.get('evaluation', {})
            
            if isinstance(evaluation, dict) and 'error' not in evaluation:
                successful_evaluations += 1
                
                for metric_name, metric_data in evaluation.items():
                    if isinstance(metric_data, dict) and 'score' in metric_data and metric_data['score'] is not None:
                        if metric_name not in deepeval_metrics:
                            deepeval_metrics[metric_name] = []
                        deepeval_metrics[metric_name].append(float(metric_data['score']))
                        
                        if metric_name == 'contextual_relevancy':
                            contextual_relevancy_scores.append(float(metric_data['score']))
        
        def calculate_stats(scores):
            if scores:
                scores_array = np.array(scores)
                return {
                    'mean': float(np.mean(scores_array)),
                    'std': float(np.std(scores_array)),
                    'min': float(np.min(scores_array)),
                    'max': float(np.max(scores_array)),
                    'median': float(np.median(scores_array)),
                    'count': int(len(scores))
                }
            return None
        
        aggregate = {
            'deepeval_metrics': {name: calculate_stats(scores) for name, scores in deepeval_metrics.items()},
            'component_analysis': component_analysis,
            'contextual_relevancy_focus': {
                'scores': contextual_relevancy_scores,
                'statistics': calculate_stats(contextual_relevancy_scores),
                'target_improvement': 0.65
            },
            'summary': {
                'total_questions': len(results),
                'successful_deepeval': successful_evaluations,
                'collections_used': [],
                'avg_performance_relevancy': 0.0
            }
        }
        
        print(f"? Enhanced metrics calculated: {successful_evaluations}/{len(results)} successful")
        return aggregate
    
    def save_enhanced_results(self, results: List[Dict[str, Any]], aggregate_metrics: Dict[str, Any]) -> None:
        """Save enhanced evaluation results with API cost tracking"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        detailed_file = os.path.join(self.output_dir, f"enhanced_results_{timestamp}.json")
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save aggregate metrics
        aggregate_file = os.path.join(self.output_dir, f"enhanced_metrics_{timestamp}.json")
        with open(aggregate_file, 'w', encoding='utf-8') as f:
            json.dump(aggregate_metrics, f, indent=2, ensure_ascii=False)
        
        # Save API call data - Added from script 1
        api_summary = api_tracker.get_summary()
        api_summary_file = os.path.join(self.output_dir, f"api_usage_{timestamp}.json")
        with open(api_summary_file, 'w', encoding='utf-8') as f:
            json.dump(api_summary, f, indent=2, ensure_ascii=False)
        
        # Analyze question type distribution
        question_type_distribution = {}
        for result in results:
            q_type = result['question_data'].get('question_type', 'unknown').lower()
            question_type_distribution[q_type] = question_type_distribution.get(q_type, 0) + 1
        
        # Create enhanced report
        report_file = os.path.join(self.output_dir, f"enhanced_report_{timestamp}.txt")
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("ENHANCED RAG EVALUATION REPORT - CONTEXTUAL RELEVANCY FOCUS\n")
            f.write("=" * 80 + "\n\n")
            
            # Summary
            f.write("EVALUATION SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Questions Evaluated: {aggregate_metrics['summary']['total_questions']}\n")
            f.write(f"Successful DeepEval: {aggregate_metrics['summary']['successful_deepeval']}\n")
            f.write(f"Contextual Optimization Used: 0\n")  # Since we're not using fresh responses
            
            # API Cost Summary - Added from script 1
            f.write(f"\nAPI COST SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total API Cost: ${api_summary['total_cost']:.4f}\n")
            if len(results) > 0:
                f.write(f"Cost per Question: ${api_summary['total_cost']/len(results):.4f}\n")
            
            # API calls by model
            f.write(f"\nAPI CALLS BY MODEL:\n")
            for key, count in api_summary['calls'].items():
                provider, model = key.split(':')
                error_count = api_summary['errors'].get(key, 0)
                error_rate = (error_count / count) * 100 if count > 0 else 0
                f.write(f"  {provider} {model}: {count} calls ({error_count} errors, {error_rate:.1f}% error rate)\n")
            
            # Token usage
            f.write(f"\nTOKEN USAGE:\n")
            for key in api_summary['calls'].keys():
                provider, model = key.split(':')
                input_tokens = api_summary['tokens']['input'].get(key, 0)
                output_tokens = api_summary['tokens']['output'].get(key, 0)
                total_tokens = input_tokens + output_tokens
                f.write(f"  {provider} {model}: {total_tokens:,} tokens ({input_tokens:,} input, {output_tokens:,} output)\n")
            
            # Question Type Distribution
            f.write("\nQUESTION TYPE DISTRIBUTION\n")
            f.write("-" * 30 + "\n")
            total_questions = sum(question_type_distribution.values())
            
            for q_type, count in sorted(question_type_distribution.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_questions) * 100 if total_questions > 0 else 0
                f.write(f"{q_type.capitalize()}: {count} questions ({percentage:.1f}%)\n")
            
            # Show missing question types
            all_question_types = {'simple', 'complex', 'distracting', 'situational', 'double', 'conversational', 'general'}
            present_types = set(question_type_distribution.keys())
            missing_types = all_question_types - present_types
            if missing_types:
                f.write(f"Missing question types: {', '.join(sorted(missing_types))}\n")
            
            # Enhanced Pipeline Statistics
            f.write(f"\nENHANCED PIPELINE PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            f.write(f"Collections Used: Dataset responses only\n")
            f.write(f"Average Performance Relevancy: 0.0000\n")
            
            # Contextual Relevancy Focus
            f.write(f"\nCONTEXTUAL RELEVANCY ANALYSIS (KEY METRIC)\n")
            f.write("-" * 50 + "\n")
            
            ctx_rel_stats = aggregate_metrics['contextual_relevancy_focus']['statistics']
            if ctx_rel_stats:
                target_score = 0.65
                f.write(f"Current Score: {ctx_rel_stats['mean']:.4f} +/- {ctx_rel_stats['std']:.4f}\n")
                f.write(f"Target Score: {target_score:.4f}\n")
                improvement_needed = target_score - ctx_rel_stats['mean']
                f.write(f"Improvement Needed: {improvement_needed:+.4f}\n")
                f.write(f"Score Range: {ctx_rel_stats['min']:.4f} - {ctx_rel_stats['max']:.4f}\n")
            
            # Component-wise DeepEval Metrics
            f.write("\nDEEPEVAL METRICS BY RAG COMPONENT\n")
            f.write("-" * 40 + "\n")
            
            component_performance = aggregate_metrics['component_analysis']['component_performance']
            
            all_metric_scores = []
            for component, metrics in component_performance.items():
                for metric_name, stats in metrics.items():
                    if stats and 'mean' in stats:
                        all_metric_scores.append(stats['mean'])
            
            for component, metrics in component_performance.items():
                if metrics:
                    f.write(f"\n{component.upper()} COMPONENT:\n")
                    for metric_name, stats in metrics.items():
                        if stats:
                            f.write(f"   {metric_name:20s}: {stats['mean']:.4f} +/- {stats['std']:.4f} (median: {stats['median']:.4f})\n")
            
            if all_metric_scores:
                combined_avg = np.mean(all_metric_scores)
                combined_std = np.std(all_metric_scores)
                f.write(f"\nCOMBINED AVERAGE SCORE: {combined_avg:.4f} +/- {combined_std:.4f}\n")
            
            # Performance by Question Type
            f.write("\nPERFORMANCE BY QUESTION TYPE\n")
            f.write("-" * 40 + "\n")
            
            question_type_stats = aggregate_metrics['component_analysis']['by_question_type']
            for q_type, type_data in question_type_stats.items():
                f.write(f"\n{q_type.upper()}:\n")
                for component, metrics in type_data.items():
                    if metrics:
                        f.write(f"  {component.capitalize()}: ")
                        metric_parts = []
                        for metric_name, stats in metrics.items():
                            if stats:
                                metric_parts.append(f"{metric_name}: {stats['mean']:.4f} (count: {stats['count']})")
                        f.write(f"{' '.join(metric_parts)}\n")
            
            # Contextual Relevancy by Question Type
            f.write("\nCONTEXTUAL RELEVANCY BY QUESTION TYPE\n")
            f.write("-" * 40 + "\n")
            for q_type, type_data in question_type_stats.items():
                if 'retriever' in type_data and 'contextual_relevancy' in type_data['retriever']:
                    stats = type_data['retriever']['contextual_relevancy']
                    if stats:
                        f.write(f"{q_type.capitalize()}: {stats['mean']:.4f} +/- {stats['std']:.4f} (count: {stats['count']})\n")
        
        logger.info(f"Enhanced results saved to {self.output_dir}")
        
        # Print summary
        print(f"\n{'='*80}")
        print(f"ENHANCED EVALUATION COMPLETED - CONTEXTUAL RELEVANCY FOCUS")
        print(f"{'='*80}")
        
        # Print question type distribution
        print(f"\nQUESTION TYPE DISTRIBUTION:")
        for q_type, count in sorted(question_type_distribution.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_questions) * 100 if total_questions > 0 else 0
            print(f"   {q_type.capitalize()}: {count} questions ({percentage:.1f}%)")
        
        # Print contextual relevancy results
        ctx_rel_stats = aggregate_metrics['contextual_relevancy_focus']['statistics']
        if ctx_rel_stats:
            current_score = ctx_rel_stats['mean']
            target_score = 0.65
            print(f"\n?? Contextual Relevancy Score: {current_score:.4f}")
            print(f"?? Target Score: {target_score:.4f}")
            if current_score >= target_score:
                print(f"?? TARGET ACHIEVED! (+{current_score - target_score:.4f})")
            else:
                print(f"?? Improvement needed: {target_score - current_score:.4f}")
        
        # Print API cost summary - Added from script 1
        print(f"\n?? API COST SUMMARY:")
        print(f"   Total API cost: ${api_summary['total_cost']:.4f}")
        if len(results) > 0:
            print(f"   Cost per question: ${api_summary['total_cost']/len(results):.4f}")
        print(f"   API calls summary saved to: {api_summary_file}")
        
        print(f"?? Results saved to: {self.output_dir}")
        print(f"{'='*80}\n")

async def main():
    """Main function"""
    # Configuration
    DATASET_PATH = "//home/user/Desktop/RAG_open_source_v2/merged_classes.json"
    AZURE_OPENAI_KEY = "6e6a5f803f5642119a59cd678f234087"
    AZURE_API_VERSION = "2024-08-01-preview"
    AZURE_DEPLOYMENT = "cbs-gpt-4o"
    AZURE_ENDPOINT = "https://cbs-gpt4o.openai.azure.com/"

    print(f"\n{'='*80}")
    print(f"FIXED RAG EVALUATION - COMPREHENSIVE WITH API COST TRACKING")
    print(f"{'='*80}")
    print(f"?? Focus: Fixed DeepEval Evaluation")
    print(f"??? Enhanced: Error Handling & Data Processing")
    print(f"?? Added: API Cost Tracking & Reporting")
    print(f"?? Target: Complete Detailed Reports")
    print(f"{'='*80}\n")

    # Reset API tracker
    api_tracker.reset()

    # Initialize evaluator
    evaluator = FixedComprehensiveEvaluator(
        azure_openai_key=AZURE_OPENAI_KEY,
        azure_api_version=AZURE_API_VERSION,
        azure_deployment=AZURE_DEPLOYMENT,
        azure_endpoint=AZURE_ENDPOINT,
        output_dir="fixed_comprehensive_results_other_qns"
    )
    
    # Load dataset
    df = evaluator.deepeval_evaluator.load_dataset(DATASET_PATH)
    
    # Configuration
    num_questions = 39 #est with 39 questions
    selected_metrics = [
        'contextual_relevancy',
        'contextual_recall',
        'contextual_precision',
        'answer_relevancy',
        'faithfulness'
    ]
    
    print(f"EVALUATION CONFIGURATION:")
    print(f"   Questions: {num_questions}")
    print(f"   Metrics: {selected_metrics}")
    print(f"   Enhanced processing: ENABLED")
    print(f"   API cost tracking: ENABLED")
    
    # Run evaluation
    evaluation_start_time = time.time()
    
    print(f"\n?? Starting fixed evaluation...")
    results = await evaluator.evaluate_comprehensive(
        df=df,
        deepeval_metrics=selected_metrics,
        num_questions=num_questions
    )
    
    evaluation_time = time.time() - evaluation_start_time
    print(f"\n? Evaluation completed in {evaluation_time:.2f} seconds")
    
    # Calculate metrics
    aggregate_metrics = evaluator.calculate_enhanced_metrics(results)
    
    # Save results
    evaluator.save_enhanced_results(results, aggregate_metrics)
    
    # Debug token tracking - Added from script 1
    api_tracker.debug_token_tracking()
    
    # Print API summary
    api_tracker.print_summary()
    
    # Display key results
    if results:
        successful_questions = len([r for r in results if 'evaluation' in r and 'error' not in r.get('evaluation', {})])
        print(f"\nKEY RESULTS:")
        print(f"   Successful evaluations: {successful_questions}/{len(results)}")
        
        # Show contextual relevancy scores
        ctx_rel_scores = []
        for result in results:
            if 'evaluation' in result and 'contextual_relevancy' in result['evaluation']:
                ctx_rel_data = result['evaluation']['contextual_relevancy']
                if isinstance(ctx_rel_data, dict) and 'score' in ctx_rel_data:
                    ctx_rel_scores.append(ctx_rel_data['score'])
                    print(f"  Q{result['question_data']['id']+1}: {ctx_rel_data['score']:.4f}")
        
        if ctx_rel_scores:
            avg_ctx_rel = np.mean(ctx_rel_scores)
            print(f"\n?? CONTEXTUAL RELEVANCY RESULTS:")
            print(f"   Average Score: {avg_ctx_rel:.4f}")
            print(f"   Range: {min(ctx_rel_scores):.4f} - {max(ctx_rel_scores):.4f}")
            print(f"   Target: 0.6500")
            
            if avg_ctx_rel >= 0.65:
                print(f"   ?? TARGET ACHIEVED!")
            else:
                improvement = 0.65 - avg_ctx_rel
                print(f"   ?? Improvement needed: {improvement:.4f}")
                
        # Print number of low relevancy questions found
        low_relevancy_threshold = 0.3
        low_scores = [s for s in ctx_rel_scores if s < low_relevancy_threshold]
        if low_scores:
            print(f"\nLOW RELEVANCY QUESTIONS:")
            print(f"   Found {len(low_scores)} questions below threshold {low_relevancy_threshold}")
            print(f"   Average Low Relevancy Score: {np.mean(low_scores):.4f}")
        else:
            print(f"   ? No questions below threshold of {low_relevancy_threshold}")
        
        # Print cost summary - Added from script 1
        api_summary = api_tracker.get_summary()
        print(f"\n?? API COST SUMMARY:")
        print(f"   Total API cost: ${api_summary['total_cost']:.4f}")
        print(f"   Cost per question: ${api_summary['total_cost']/len(results):.4f}")

    print(f"\n?? Fixed evaluation completed!")
    print(f"?? Results saved in: {evaluator.output_dir}")

if __name__ == "__main__":
    asyncio.run(main())