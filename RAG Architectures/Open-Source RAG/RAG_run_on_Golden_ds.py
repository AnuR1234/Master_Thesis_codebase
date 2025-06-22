#!/usr/bin/env python3
"""
Enhanced RAG Evaluation Script

This script evaluates the RAG pipeline on a golden dataset of questions, with additional
features for robust error handling, memory management, and comprehensive result tracking.

Features:
1. Improved error handling with retries for failed queries
2. Memory management to prevent GPU OOM errors
3. Enhanced progress tracking and reporting
4. Flexible dataset loading supporting multiple formats
5. Periodic result saving to prevent data loss
"""
import json
import time
import asyncio
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
import os
import traceback
import gc
from tqdm import tqdm
import argparse
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_evaluation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rag_evaluation")

# Import the RAG pipeline
from pipeline import EnhancedRAGPipeline

# Apply nest_asyncio to allow asyncio to work in environments like Jupyter
import nest_asyncio
nest_asyncio.apply()

class EnhancedRAGEvaluator:
    """Enhanced evaluator for RAG pipeline on golden dataset questions"""
    
    def __init__(self, 
                golden_dataset_path: str, 
                output_path: str = "rag_evaluation_results.json",
                pipeline_config: Dict = None,
                enable_retries: bool = True,
                max_retries: int = 2):
        """
        Initialize the enhanced RAG evaluator
        
        Args:
            golden_dataset_path: Path to the golden dataset CSV or JSON
            output_path: Path to save the evaluation results
            pipeline_config: Optional configuration for the RAG pipeline
            enable_retries: Whether to enable retries for failed queries
            max_retries: Maximum number of retries for failed queries
        """
        self.golden_dataset_path = golden_dataset_path
        self.output_path = output_path
        self.results = []
        self.pipeline_config = pipeline_config or {}
        self.enable_retries = enable_retries
        self.max_retries = max_retries
        
        # Statistics tracking
        self.stats = {
            "total_questions": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "retried_queries": 0,
            "empty_responses": 0,
            "avg_latency": 0,
            "start_time": time.time(),
            "end_time": None,
            "total_runtime": 0,
            "collection_type": self.pipeline_config.get("collection_type", "default")
        }
        
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        self.pipeline = EnhancedRAGPipeline()
        
        # Set collection type if specified
        if "collection_type" in self.pipeline_config:
            collection_type = self.pipeline_config["collection_type"]
            logger.info(f"Setting collection type to: {collection_type}")
            self.pipeline.set_collection_type(collection_type)
        
        # Load the golden dataset
        logger.info(f"Loading golden dataset from {golden_dataset_path}...")
        self.load_golden_dataset()
    
    def clear_gpu_memory(self):
        """Clear GPU memory to prevent OOM errors"""
        try:
            # Run garbage collection
            gc.collect()
            
            # Clear PyTorch cache if available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("Cleared GPU cache")
            except ImportError:
                pass
        except Exception as e:
            logger.warning(f"Error clearing GPU memory: {e}")
        
    def load_golden_dataset(self):
        """Load the golden dataset with robust format handling"""
        try:
            # Determine file extension
            file_ext = os.path.splitext(self.golden_dataset_path)[1].lower()
            
            if file_ext == '.json':
                # Load JSON dataset
                with open(self.golden_dataset_path, 'r') as f:
                    data = json.load(f)
                
                # Handle various JSON structures
                if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict) and "samples" in data[0]:
                    data = data[0]["samples"]
                elif isinstance(data, dict) and "samples" in data:
                    data = data["samples"]
                elif isinstance(data, dict):
                    data = [data]
                
                # Convert to DataFrame
                self.dataset = pd.DataFrame(data)
                
            elif file_ext == '.csv':
                # Load CSV dataset
                self.dataset = pd.read_csv(self.golden_dataset_path)
                
            else:
                # Try JSON first, then CSV
                try:
                    with open(self.golden_dataset_path, 'r') as f:
                        data = json.load(f)
                    
                    # Handle various JSON structures
                    if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict) and "samples" in data[0]:
                        data = data[0]["samples"]
                    elif isinstance(data, dict) and "samples" in data:
                        data = data["samples"]
                    elif isinstance(data, dict):
                        data = [data]
                    
                    # Convert to DataFrame
                    self.dataset = pd.DataFrame(data)
                    
                except json.JSONDecodeError:
                    # Try CSV
                    self.dataset = pd.read_csv(self.golden_dataset_path)
            
            # Validate and standardize column names (case-insensitive)
            required_columns_lower = ["question", "question_type"]
            
            # Get all column names in lowercase for case-insensitive comparison
            available_columns_lower = [col.lower() for col in self.dataset.columns]
            
            # Create a mapping from lowercase column names to actual column names
            col_case_mapping = {col.lower(): col for col in self.dataset.columns}
            
            # Check if required columns exist (case-insensitive)
            missing_columns = []
            column_mapping = {}
            
            for req_col in required_columns_lower:
                if req_col in available_columns_lower:
                    # Map to standardized column name
                    actual_col = col_case_mapping[req_col]
                    if req_col == "question" and actual_col != "Question":
                        column_mapping[actual_col] = "Question"
                    elif req_col == "question_type" and actual_col != "Question_type":
                        column_mapping[actual_col] = "Question_type"
                else:
                    missing_columns.append(req_col)
            
            if missing_columns:
                raise ValueError(f"Dataset missing required columns: {', '.join(missing_columns)}")
                
            # Rename columns if needed
            if column_mapping:
                self.dataset = self.dataset.rename(columns=column_mapping)
                
            # Set index for easier access
            if "id" in self.dataset.columns:
                self.dataset.set_index("id", inplace=True, drop=False)
            
            self.stats["total_questions"] = len(self.dataset)
            logger.info(f"Successfully loaded dataset with {len(self.dataset)} questions")
            logger.info(f"Dataset columns: {list(self.dataset.columns)}")
            
            # Print first row for debugging
            if not self.dataset.empty:
                logger.info(f"First question sample: '{self.dataset.iloc[0]['Question'][:100]}...'")
                
        except Exception as e:
            logger.error(f"Error loading golden dataset: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def process_question_with_retry(self, question: str, question_type: str, 
                                         question_id: Any = None) -> Dict[str, Any]:
        """
        Process a question with retry mechanism for better reliability
        
        Args:
            question: The question to process
            question_type: Type of question (from golden dataset)
            question_id: Optional question ID for tracking
            
        Returns:
            Dictionary with query results and metadata
        """
        retry_count = 0
        max_retries = self.max_retries if self.enable_retries else 0
        errors = []
        
        while retry_count <= max_retries:
            try:
                # If not first attempt, log retry information
                if retry_count > 0:
                    logger.info(f"Retry {retry_count}/{max_retries} for question: '{question[:100]}...'")
                    self.stats["retried_queries"] += 1
                else:
                    logger.info(f"Processing question: '{question[:100]}...'")
                
                # Record start time for latency measurement
                start_time = time.time()
                
                # Determine retrieval settings based on question type and retry count
                use_hybrid = True
                use_reranker = True
                
                # Increase top_k for retries
                top_k = 5 + retry_count
                
                logger.info(f"Using settings: hybrid={use_hybrid}, reranker={use_reranker}, top_k={top_k}")
                
                # Process the query through RAG pipeline
                try:
                    # Clear memory before processing
                    self.clear_gpu_memory()
                    
                    # Call process_query without enable_query_enhancement parameter
                    result = await self.pipeline.process_query(
                        query=question,
                        conversation_history=None,  # No conversation history for evaluation
                        use_hybrid=use_hybrid,
                        use_reranker=use_reranker,
                        top_k=top_k,
                        collection_type=self.pipeline_config.get("collection_type")
                    )
                except Exception as e:
                    logger.error(f"Error in pipeline.process_query: {e}")
                    logger.error(traceback.format_exc())
                    errors.append(f"Process error: {str(e)}")
                    result = {
                        "error": str(e),
                        "response": f"Error processing query: {str(e)}",
                        "contexts": [],
                        "has_relevant_results": False
                    }
                
                # Calculate latency and round to 2 decimal places
                latency = round(time.time() - start_time, 2)
                
                # Check if we got a valid response
                response_text = result.get("response", "")
                
                # If response is empty or error, and we have retries left, try again
                if ((not response_text or response_text.startswith("Error") or 
                     "not have enough information" in response_text or
                     result.get("has_relevant_results") is False) and 
                    retry_count < max_retries):
                    
                    retry_count += 1
                    errors.append(f"Empty or error response, retrying ({retry_count}/{max_retries})")
                    continue
                
                # Extract relevant information from the result
                processed_result = {
                    "Question": question,
                    "question_id": question_id,
                    "question_type": question_type,
                    "expected_response": None,  # Not using reference answer for now
                    "retrieved_docs": [
                        {
                            "id": ctx.get("id", ""),
                            "title": ctx.get("title", ""),
                            "filename": ctx.get("filename", ""),
                            "score": ctx.get("score", 0),
                            "text": ctx.get("text", ""),
                            "code_snippet": ctx.get("code_snippet", "")
                        }
                        for ctx in result.get("contexts", [])
                    ],
                    "response": response_text,
                    "confidence_level": result.get("confidence_level", "LOW"),
                    "has_relevant_results": result.get("has_relevant_results", False),
                    "query_enhancement_used": result.get("query_enhancement_used", False),
                    "retries": retry_count,
                    "errors": errors,
                    "latency": latency
                }
                
                # Track statistics
                if response_text and not response_text.startswith("Error"):
                    self.stats["successful_queries"] += 1
                else:
                    self.stats["failed_queries"] += 1
                
                if not response_text or len(response_text.strip()) < 10:
                    self.stats["empty_responses"] += 1
                
                logger.info(f"Processed question in {latency:.2f} seconds (retries: {retry_count})")
                return processed_result
            
            except Exception as e:
                logger.error(f"Error processing question '{question[:50]}...': {e}")
                logger.error(traceback.format_exc())
                errors.append(str(e))
                retry_count += 1
                
                # Clear memory after error
                self.clear_gpu_memory()
        
        # If we get here, all retries failed
        self.stats["failed_queries"] += 1
        
        # Return error result
        return {
            "Question": question,
            "question_id": question_id,
            "question_type": question_type,
            "expected_response": None,
            "retrieved_docs": [],
            "response": f"Error after {retry_count} retries: {'; '.join(errors)}",
            "confidence_level": "LOW",
            "has_relevant_results": False,
            "retries": retry_count,
            "errors": errors,
            "latency": 0.0
        }
    
    async def evaluate_dataset(self, limit: Optional[int] = None, 
                              start_index: int = 0,
                              save_interval: int = 10):
        """
        Evaluate questions in the golden dataset
        
        Args:
            limit: Optional limit on number of questions to process
            start_index: Index to start processing from (for resuming)
            save_interval: How often to save intermediate results
        """
        try:
            # Get questions to process (with optional limit)
            questions_df = self.dataset.iloc[start_index:start_index + limit] if limit else self.dataset.iloc[start_index:]
            total_questions = len(questions_df)
            
            logger.info(f"Starting evaluation of {total_questions} questions (starting from index {start_index})")
            
            # Create progress bar
            pbar = tqdm(total=total_questions, desc="Evaluating questions", unit="question")
            
            # Track latencies for statistics
            latencies = []
            
            # Process each question
            for idx, row in questions_df.iterrows():
                try:
                    # Get question and question_type from the row
                    question = row["Question"]
                    question_type = row["Question_type"]
                    question_id = row.get("id", idx)
                    
                    logger.info(f"Processing question {idx} (ID: {question_id}): '{question[:50]}...'")
                    
                    # Process question with retry
                    result = await self.process_question_with_retry(question, question_type, question_id)
                    self.results.append(result)
                    
                    # Track latency for statistics
                    if result["latency"] > 0:
                        latencies.append(result["latency"])
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Save results periodically
                    if (len(self.results) % save_interval == 0) or (len(self.results) == total_questions):
                        self.save_results()
                        logger.info(f"Saved intermediate results after {len(self.results)} questions")
                
                except Exception as e:
                    logger.error(f"Error processing question at index {idx}: {e}")
                    logger.error(traceback.format_exc())
                    # Update progress bar even if there's an error
                    pbar.update(1)
                    # Continue with next question
                    continue
                
                # Clear memory periodically
                if idx % 5 == 0:
                    self.clear_gpu_memory()
            
            # Close progress bar
            pbar.close()
            
            # Update statistics
            self.stats["end_time"] = time.time()
            self.stats["total_runtime"] = self.stats["end_time"] - self.stats["start_time"]
            if latencies:
                self.stats["avg_latency"] = np.mean(latencies)
                self.stats["min_latency"] = np.min(latencies)
                self.stats["max_latency"] = np.max(latencies)
            
            logger.info(f"Completed evaluation of {total_questions} questions")
            logger.info(f"Success rate: {self.stats['successful_queries']}/{total_questions} "
                       f"({self.stats['successful_queries']/total_questions*100:.1f}%)")
            
            # Save final results with statistics
            self.save_results(include_stats=True)
            
        except Exception as e:
            logger.error(f"Error during dataset evaluation: {e}")
            logger.error(traceback.format_exc())
            # Save whatever results we have
            if self.results:
                self.save_results(include_stats=True)
                logger.info(f"Saved partial results ({len(self.results)} questions) due to error")
    
    def save_results(self, include_stats: bool = False):
        """
        Save evaluation results to JSON file
        
        Args:
            include_stats: Whether to include statistics in the saved results
        """
        try:
            logger.info(f"Saving evaluation results to {self.output_path}...")
            
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
            
            # Prepare data to save
            output_data = {
                "results": self.results,
                "timestamp": datetime.now().isoformat(),
                "total_questions": len(self.results)
            }
            
            # Include statistics if requested
            if include_stats:
                output_data["statistics"] = self.stats
            
            # Save to file
            with open(self.output_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            logger.info(f"Successfully saved results for {len(self.results)} questions")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            raise
    
    def generate_report(self, output_path: Optional[str] = None):
        """
        Generate a comprehensive evaluation report
        
        Args:
            output_path: Optional path to save the report, defaults to 'report_{timestamp}.json'
        """
        if not self.results:
            logger.warning("No results to generate report from")
            return
        
        try:
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"report_{timestamp}.json"
            
            logger.info(f"Generating evaluation report to {output_path}...")
            
            # Calculate statistics
            total_questions = len(self.results)
            successful = sum(1 for r in self.results if r.get("response") and not r.get("response").startswith("Error"))
            failed = total_questions - successful
            with_docs = sum(1 for r in self.results if r.get("retrieved_docs") and len(r.get("retrieved_docs")) > 0)
            empty_responses = sum(1 for r in self.results if not r.get("response") or len(r.get("response").strip()) < 10)
            
            # Calculate average latency
            latencies = [r.get("latency", 0) for r in self.results if r.get("latency", 0) > 0]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            
            # Group by question type
            question_types = {}
            for r in self.results:
                q_type = r.get("question_type", "unknown").lower()
                if q_type not in question_types:
                    question_types[q_type] = {
                        "count": 0,
                        "successful": 0,
                        "failed": 0,
                        "avg_latency": 0,
                        "latencies": []
                    }
                
                question_types[q_type]["count"] += 1
                
                if r.get("response") and not r.get("response").startswith("Error"):
                    question_types[q_type]["successful"] += 1
                else:
                    question_types[q_type]["failed"] += 1
                
                if r.get("latency", 0) > 0:
                    question_types[q_type]["latencies"].append(r.get("latency", 0))
            
            # Calculate averages for each question type
            for q_type in question_types:
                latencies = question_types[q_type]["latencies"]
                question_types[q_type]["avg_latency"] = sum(latencies) / len(latencies) if latencies else 0
                # Remove raw latencies list to keep report compact
                del question_types[q_type]["latencies"]
            
            # Compile report
            report = {
                "summary": {
                    "total_questions": total_questions,
                    "successful_queries": successful,
                    "failed_queries": failed,
                    "success_rate": successful / total_questions if total_questions > 0 else 0,
                    "queries_with_documents": with_docs,
                    "empty_responses": empty_responses,
                    "avg_latency": avg_latency,
                    "total_runtime": self.stats.get("total_runtime", 0)
                },
                "by_question_type": question_types,
                "pipeline_config": self.pipeline_config,
                "timestamp": datetime.now().isoformat(),
                "dataset_path": self.golden_dataset_path
            }
            
            # Save report
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Report generated successfully to {output_path}")
            
            # Print summary
            logger.info("=== Evaluation Summary ===")
            logger.info(f"Total questions: {total_questions}")
            logger.info(f"Success rate: {successful}/{total_questions} ({successful/total_questions*100:.1f}%)")
            logger.info(f"Average latency: {avg_latency:.2f} seconds")
            logger.info(f"Total runtime: {self.stats.get('total_runtime', 0)/60:.1f} minutes")
            logger.info("=========================")
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            logger.error(traceback.format_exc())

async def main():
    """Main function to run the evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline on golden dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Path to golden dataset")
    parser.add_argument("--output", type=str, default="rag_evaluation_results.json", help="Output path for results")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of questions to process")
    parser.add_argument("--start", type=int, default=0, help="Start index for processing")
    parser.add_argument("--collection", type=str, default=None, help="Collection type to use")
    parser.add_argument("--no-retries", action="store_true", help="Disable retries for failed queries")
    parser.add_argument("--max-retries", type=int, default=2, help="Maximum retries for failed queries")
    parser.add_argument("--save-interval", type=int, default=10, help="Save results every N questions")
    
    args = parser.parse_args()
    
    # Prepare pipeline configuration
    pipeline_config = {}
    if args.collection:
        pipeline_config["collection_type"] = args.collection
    
    # Create evaluator
    evaluator = EnhancedRAGEvaluator(
        golden_dataset_path=args.dataset,
        output_path=args.output,
        pipeline_config=pipeline_config,
        enable_retries=not args.no_retries,
        max_retries=args.max_retries
    )
    
    # Run evaluation
    await evaluator.evaluate_dataset(
        limit=args.limit,
        start_index=args.start,
        save_interval=args.save_interval
    )
    
    # Generate report
    evaluator.generate_report()
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())