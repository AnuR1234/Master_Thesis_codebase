#!/usr/bin/env python3
"""
Script to evaluate RAG pipeline on golden dataset questions and save results as JSON

This script:
1. Loads the golden dataset
2. Processes each question through the RAG pipeline
3. Measures the latency of each query
4. Saves results to a JSON file with specified structure
"""
import json
import time
import asyncio
import pandas as pd
from typing import List, Dict, Any
import logging
import os
import traceback
from tqdm import tqdm

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
from pipeline import RAGPipeline

# Apply nest_asyncio to allow asyncio to work in environments like Jupyter
import nest_asyncio
nest_asyncio.apply()

class RAGEvaluator:
    """Class to evaluate RAG pipeline on golden dataset questions"""
    
    def __init__(self, golden_dataset_path: str, output_path: str = "rag_evaluation_results.json"):
        """
        Initialize the RAG evaluator
        
        Args:
            golden_dataset_path: Path to the golden dataset CSV or JSON
            output_path: Path to save the evaluation results
        """
        self.golden_dataset_path = golden_dataset_path
        self.output_path = output_path
        self.results = []
        
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        self.pipeline = RAGPipeline()
        
        # Load the golden dataset
        logger.info(f"Loading golden dataset from {golden_dataset_path}...")
        self.load_golden_dataset()
        
    def load_golden_dataset(self):
        """Load the golden dataset from JSON file"""
        try:
            # Load JSON dataset
            with open(self.golden_dataset_path, 'r') as f:
                data = json.load(f)
            
            # If data is a list containing a single item, and that item is a dict of samples, extract samples
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict) and "samples" in data[0]:
                data = data[0]["samples"]
            
            # If data is just a dict (not a list of dicts), wrap it in a list
            if isinstance(data, dict):
                data = [data]
                
            # Convert to DataFrame for easier processing
            self.dataset = pd.DataFrame(data)
                
            # Validate dataset contains required columns, case-insensitive
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
                    # Map to standardized column name if needed
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
                
            logger.info(f"Successfully loaded dataset with {len(self.dataset)} questions")
            logger.info(f"Dataset columns after mapping: {list(self.dataset.columns)}")
            
            # Print first row for debugging
            if not self.dataset.empty:
                logger.info(f"First question sample: '{self.dataset.iloc[0]['Question'][:100]}...'")
                
        except Exception as e:
            logger.error(f"Error loading golden dataset: {e}")
            logger.error(traceback.format_exc())
            raise
            
    async def process_question(self, question: str, question_type: str) -> Dict[str, Any]:
        """
        Process a single question through the RAG pipeline
        
        Args:
            question: The question to process
            question_type: Type of question (from golden dataset)
            
        Returns:
            Dictionary with query results and metadata
        """
        try:
            logger.info(f"Processing question: '{question[:100]}...'")
            
            # Record start time for latency measurement
            start_time = time.time()
            
            # Determine whether to use query enhancement based on question type
            # Disable enhancement for simple questions
            enable_enhancement = question_type.lower() != "simple"
            logger.info(f"Query enhancement {'enabled' if enable_enhancement else 'disabled'} for {question_type} question")
            
            # Process the query through RAG pipeline with appropriate enhancement setting
            result = await self.pipeline.process_query(
                query=question,
                conversation_history=None,  # No conversation history for evaluation
                use_hybrid=True,            # Use hybrid retrieval
                use_reranker=True,          # Use reranker
                top_k=5,                    # Use top 5 documents
                collection_type=None,       # Use default collection
                enable_query_enhancement=enable_enhancement  # Conditionally enable enhancement
            )
            
            # Calculate latency and round to 2 decimal places
            latency = round(time.time() - start_time, 2)
            
            # Extract relevant information from the result
            processed_result = {
                "Question": question,
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
                "response": result.get("response", ""),
                "question_type": question_type,
                "latency": latency  # Rounded to 2 decimal places
                # Removed confidence_level as requested
            }
            
            logger.info(f"Processed question in {latency:.2f} seconds")
            return processed_result
            
        except Exception as e:
            logger.error(f"Error processing question '{question[:50]}...': {e}")
            logger.error(traceback.format_exc())
            # Return error result
            return {
                "Question": question,
                "expected_response": None,
                "retrieved_docs": [],
                "response": f"Error: {str(e)}",
                "question_type": question_type,
                "latency": 0.0
            }
    
    async def evaluate_dataset(self, limit: int = None):
        """
        Evaluate all questions in the golden dataset
        
        Args:
            limit: Optional limit on number of questions to process
        """
        try:
            # Get questions to process (with optional limit)
            questions_df = self.dataset[:limit] if limit else self.dataset
            total_questions = len(questions_df)
            
            logger.info(f"Starting evaluation of {total_questions} questions...")
            
            # Create progress bar
            pbar = tqdm(total=total_questions, desc="Evaluating questions", unit="question")
            
            # Process each question
            for idx, row in questions_df.iterrows():
                try:
                    # Get question and question_type from the row
                    question = row["Question"]
                    question_type = row["Question_type"]
                    
                    logger.info(f"Processing question {idx+1}/{total_questions}: '{question[:50]}...'")
                    
                    # Process question
                    result = await self.process_question(question, question_type)
                    self.results.append(result)
                    
                    # Update progress bar
                    pbar.update(1)
                    
                    # Save results periodically (every 10 questions)
                    if (idx + 1) % 10 == 0:
                        self.save_results()
                        logger.info(f"Saved intermediate results after {idx+1} questions")
                
                except Exception as e:
                    logger.error(f"Error processing question at index {idx}: {e}")
                    logger.error(traceback.format_exc())
                    # Update progress bar even if there's an error
                    pbar.update(1)
                    # Continue with next question
                    continue
            
            # Close progress bar
            pbar.close()
            logger.info(f"Completed evaluation of {total_questions} questions")
            
        except Exception as e:
            logger.error(f"Error during dataset evaluation: {e}")
            logger.error(traceback.format_exc())
            # Save whatever results we have
            if self.results:
                self.save_results()
                logger.info(f"Saved partial results ({len(self.results)} questions) due to error")
    
    def save_results(self):
        """Save evaluation results to JSON file"""
        try:
            logger.info(f"Saving evaluation results to {self.output_path}...")
            
            # Save to file
            with open(self.output_path, 'w') as f:
                json.dump(self.results, f, indent=2)
                
            logger.info(f"Successfully saved results for {len(self.results)} questions")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            raise

async def main():
    """Main function to run the evaluation"""
    # Set paths
    golden_dataset_path = "class_demo.json"  # Update with your dataset path
    output_path = "class_demo_RAG_full.json"
    
    # Create evaluator
    evaluator = RAGEvaluator(golden_dataset_path, output_path)
    
    # Run evaluation (optionally with a limit for testing)
    # Uncomment the next line to use a limit for testing
    # await evaluator.evaluate_dataset(limit=5)
    
    # For full evaluation, use:
    await evaluator.evaluate_dataset()
    
    # Save results
    evaluator.save_results()
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())