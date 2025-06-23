#!/usr/bin/env python3
"""
RAG Pipeline Evaluation with DeepEval Framework
Compares against Claude baseline performance
"""

import json
import time
import asyncio
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
import numpy as np
from datetime import datetime
import os

# DeepEval imports
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

# Additional evaluation imports
from rouge import Rouge
from sentence_transformers import SentenceTransformer
import nltk
from sklearn.metrics.pairwise import cosine_similarity

# Import your RAG pipeline
from pipeline import EnhancedRAGPipeline

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepEvalRAGEvaluator:
    """RAG evaluator using DeepEval framework"""
    
    def __init__(self, openai_api_key: str = None):
        logger.info("Initializing DeepEval RAG Evaluator...")
        
        # Set OpenAI API key for DeepEval
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            logger.info("OpenAI API key configured for DeepEval")
        elif not os.getenv("OPENAI_API_KEY"):
            logger.error("OpenAI API key required for DeepEval metrics")
            raise ValueError("Please provide OpenAI API key via --api-key or OPENAI_API_KEY environment variable")
        
        # Initialize DeepEval metrics with OpenAI
        self.answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
        self.faithfulness_metric = FaithfulnessMetric(threshold=0.7)
        
        # Initialize additional metrics
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.rouge = Rouge()
        
        # Initialize RAG pipeline
        self.rag_pipeline = EnhancedRAGPipeline()
        
        # Claude baseline (from your data)
        self.claude_baseline = {
            'answer_relevancy': 0.875,
            'faithfulness': 1.0,
            'rouge_l': 0.433,
            'bert_similarity': 0.909,
            'generation_time': 6.051
        }
        
        logger.info("DeepEval RAG Evaluator initialized successfully")
    
    def calculate_bert_similarity(self, reference: str, candidate: str) -> float:
        """Calculate BERT similarity"""
        try:
            reference_embedding = self.bert_model.encode([reference])
            candidate_embedding = self.bert_model.encode([candidate])
            similarity = cosine_similarity(reference_embedding, candidate_embedding)[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Error calculating BERT similarity: {e}")
            return 0.0
    
    def calculate_rouge_l(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L score"""
        try:
            if not reference.strip() or not candidate.strip():
                return 0.0
            scores = self.rouge.get_scores(candidate.strip(), reference.strip())
            return float(scores[0]['rouge-l']['f'])
        except Exception as e:
            logger.error(f"Error calculating ROUGE-L: {e}")
            return 0.0
    
    def extract_context_from_results(self, rag_results: Dict) -> str:
        """Extract context from RAG results"""
        contexts = rag_results.get('contexts', [])
        if not contexts:
            return ""
        
        context_parts = []
        for ctx in contexts:
            if ctx.get('text'):
                context_parts.append(ctx['text'])
            if ctx.get('code_snippet'):
                context_parts.append(ctx['code_snippet'])
        
        return "\n\n".join(context_parts)
    
    async def evaluate_single_question(self, 
                                     question_data: Dict, 
                                     max_tokens: int,
                                     temperature: float,
                                     top_p: float,
                                     top_k: int = 5) -> Dict[str, Any]:
        """Evaluate single question with DeepEval metrics"""
        
        question = question_data['question']
        reference_answer = question_data['reference_answer']
        conversation_history = question_data.get('conversation_history', [])
        
        logger.info(f"Evaluating: {question[:50]}...")
        
        # Modify generation config temporarily
        original_config = None
        try:
            from transformers import GenerationConfig
            new_config = GenerationConfig(
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=50,
                do_sample=True,
                pad_token_id=self.rag_pipeline.generator.tokenizer.pad_token_id,
                eos_token_id=self.rag_pipeline.generator.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                length_penalty=1.0
            )
            
            if hasattr(self.rag_pipeline.generator, 'model'):
                original_config = self.rag_pipeline.generator.model.generation_config
                self.rag_pipeline.generator.model.generation_config = new_config
        except Exception as e:
            logger.warning(f"Could not modify generation config: {e}")
        
        start_time = time.time()
        
        try:
            # Run RAG query
            rag_results = await self.rag_pipeline.process_query(
                query=question,
                conversation_history=conversation_history if conversation_history else None,
                top_k=top_k
            )
            
            generation_time = time.time() - start_time
            
            # Extract results
            generated_answer = rag_results.get('response', '')
            retrieved_context = self.extract_context_from_results(rag_results)
            
            # Create DeepEval test case
            test_case = LLMTestCase(
                input=question,
                actual_output=generated_answer,
                expected_output=reference_answer,
                retrieval_context=[retrieved_context] if retrieved_context else []
            )
            
            # Evaluate with DeepEval metrics
            answer_relevancy_score = 0.0
            faithfulness_score = 0.0
            
            try:
                logger.info("Evaluating Answer Relevancy with DeepEval...")
                self.answer_relevancy_metric.measure(test_case)
                answer_relevancy_score = self.answer_relevancy_metric.score
                logger.info(f"Answer Relevancy Score: {answer_relevancy_score:.3f}")
            except Exception as e:
                logger.error(f"Answer relevancy evaluation failed: {e}")
                logger.warning("Falling back to BERT similarity for answer relevancy")
                answer_relevancy_score = self.calculate_bert_similarity(question, generated_answer)
            
            try:
                if retrieved_context:
                    logger.info("Evaluating Faithfulness with DeepEval...")
                    self.faithfulness_metric.measure(test_case)
                    faithfulness_score = self.faithfulness_metric.score
                    logger.info(f"Faithfulness Score: {faithfulness_score:.3f}")
                else:
                    logger.warning("No context available for faithfulness evaluation")
                    faithfulness_score = 0.0
            except Exception as e:
                logger.error(f"Faithfulness evaluation failed: {e}")
                logger.warning("Setting faithfulness score to 0.0")
                faithfulness_score = 0.0
            
            # Calculate additional metrics
            rouge_l = self.calculate_rouge_l(reference_answer, generated_answer)
            bert_similarity = self.calculate_bert_similarity(reference_answer, generated_answer)
            
            # Calculate performance vs Claude baseline
            performance_vs_claude = {
                'answer_relevancy_gap': answer_relevancy_score - self.claude_baseline['answer_relevancy'],
                'faithfulness_gap': faithfulness_score - self.claude_baseline['faithfulness'],
                'rouge_l_gap': rouge_l - self.claude_baseline['rouge_l'],
                'bert_similarity_gap': bert_similarity - self.claude_baseline['bert_similarity'],
                'generation_time_gap': generation_time - self.claude_baseline['generation_time']
            }
            
            metrics = {
                'question': question,
                'generated_answer': generated_answer,
                'reference_answer': reference_answer,
                'parameters': {
                    'max_tokens': max_tokens,
                    'temperature': temperature,
                    'top_p': top_p,
                    'top_k': top_k
                },
                'answer_relevancy': answer_relevancy_score,
                'faithfulness': faithfulness_score,
                'rouge_l': rouge_l,
                'bert_similarity': bert_similarity,
                'generation_time': generation_time,
                'confidence_level': rag_results.get('confidence_level', 'UNKNOWN'),
                'context_count': rag_results.get('context_count', 0),
                'answer_length': len(generated_answer.split()) if generated_answer else 0,
                'performance_vs_claude': performance_vs_claude
            }
            
            logger.info(f"Metrics - AR: {answer_relevancy_score:.3f}, F: {faithfulness_score:.3f}, R-L: {rouge_l:.3f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating question: {e}")
            return {
                'question': question,
                'error': str(e),
                'parameters': {'max_tokens': max_tokens, 'temperature': temperature, 'top_p': top_p, 'top_k': top_k},
                'generation_time': time.time() - start_time,
                'answer_relevancy': 0.0,
                'faithfulness': 0.0,
                'rouge_l': 0.0,
                'bert_similarity': 0.0,
                'performance_vs_claude': {}
            }
        
        finally:
            # Restore original config
            if original_config and hasattr(self.rag_pipeline.generator, 'model'):
                self.rag_pipeline.generator.model.generation_config = original_config
    
    async def run_evaluation_experiment(self, 
                                      questions_file: str,
                                      output_file: str = None,
                                      max_questions: int = None,
                                      api_key: str = None) -> pd.DataFrame:
        """Run evaluation experiment with Claude comparison"""
        
        # Parameter combinations for Mistral model evaluation
        parameter_combinations = [
            # (max_tokens, temperature, top_p, description)
            # Key configurations of interest
            (4096, 0.7, 0.9, "Claude's Configuration (Primary Test)"),
            (1024, 0.5, 0.8, "Your Target Configuration (Primary Test)"),
            
            # Additional parameter exploration
            (512, 0.1, 0.7, "Conservative short"),
            (512, 0.3, 0.8, "Moderate short"),
            (1024, 0.1, 0.7, "Conservative medium"),
            (1024, 0.3, 0.9, "Moderate medium"),
            (2048, 0.5, 0.8, "Balanced long"),
            (2048, 0.7, 0.9, "Creative long"),
            (4096, 0.3, 0.8, "Conservative max"),
            (4096, 0.9, 0.9, "Creative max"),
        ]
        
        # Load questions
        with open(questions_file, 'r') as f:
            questions_data = json.load(f)
        
        if max_questions:
            questions_data = questions_data[:max_questions]
        
        logger.info(f"Loaded {len(questions_data)} questions")
        
        all_results = []
        
        # Run evaluation for each parameter combination
        for i, (max_tokens, temperature, top_p, description) in enumerate(parameter_combinations):
            logger.info(f"\n=== Parameter Set {i+1}/{len(parameter_combinations)} ===")
            logger.info(f"{description}: max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")
            
            for j, question_data in enumerate(questions_data):
                logger.info(f"Question {j+1}/{len(questions_data)}")
                
                result = await self.evaluate_single_question(
                    question_data, max_tokens, temperature, top_p
                )
                
                result['experiment_id'] = f"exp_{i+1}"
                result['question_id'] = j + 1
                result['description'] = description
                all_results.append(result)
                
                await asyncio.sleep(0.1)
        
        # Convert to DataFrame
        df_results = pd.DataFrame(all_results)
        
        # Save results
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"deepeval_rag_results_{timestamp}.csv"
        
        df_results.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")
        
        # Generate comparison report
        self.generate_claude_comparison_report(df_results, output_file.replace('.csv', '_claude_comparison.txt'))
        
        return df_results
    
    def generate_claude_comparison_report(self, df: pd.DataFrame, output_file: str):
        """Generate performance analysis report for Mistral model"""
        
        with open(output_file, 'w') as f:
            f.write("MISTRAL MODEL PERFORMANCE ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            
            # Claude baseline reference
            f.write("REFERENCE: CLAUDE BASELINE PERFORMANCE\n")
            f.write("-" * 40 + "\n")
            for metric, value in self.claude_baseline.items():
                f.write(f"{metric}: {value:.3f}\n")
            f.write("\n")
            
            # Key configuration performance
            f.write("KEY CONFIGURATIONS PERFORMANCE\n")
            f.write("-" * 35 + "\n")
            
            # Find Claude's config and your target config
            claude_config = df[df['description'].str.contains("Claude's Configuration")]
            target_config = df[df['description'].str.contains("Your Target Configuration")]
            
            if not claude_config.empty:
                f.write("CLAUDE'S CONFIGURATION (4096, 0.7, 0.9) on Mistral:\n")
                f.write(f"  Answer Relevancy: {claude_config['answer_relevancy'].mean():.3f}\n")
                f.write(f"  Faithfulness: {claude_config['faithfulness'].mean():.3f}\n")
                f.write(f"  ROUGE-L: {claude_config['rouge_l'].mean():.3f}\n")
                f.write(f"  BERT Similarity: {claude_config['bert_similarity'].mean():.3f}\n")
                f.write(f"  Generation Time: {claude_config['generation_time'].mean():.3f}s\n\n")
            
            if not target_config.empty:
                f.write("YOUR TARGET CONFIGURATION (1024, 0.5, 0.8) on Mistral:\n")
                f.write(f"  Answer Relevancy: {target_config['answer_relevancy'].mean():.3f}\n")
                f.write(f"  Faithfulness: {target_config['faithfulness'].mean():.3f}\n")
                f.write(f"  ROUGE-L: {target_config['rouge_l'].mean():.3f}\n")
                f.write(f"  BERT Similarity: {target_config['bert_similarity'].mean():.3f}\n")
                f.write(f"  Generation Time: {target_config['generation_time'].mean():.3f}s\n\n")
            
            # All configurations ranking
            f.write("ALL CONFIGURATIONS PERFORMANCE RANKING\n")
            f.write("-" * 42 + "\n")
            
            df_summary = df.groupby(['experiment_id', 'description']).agg({
                'answer_relevancy': 'mean',
                'faithfulness': 'mean',
                'rouge_l': 'mean',
                'bert_similarity': 'mean',
                'generation_time': 'mean'
            }).reset_index()
            
            # Calculate composite score
            df_summary['composite_score'] = (
                0.3 * df_summary['answer_relevancy'] +
                0.3 * df_summary['faithfulness'] +
                0.2 * df_summary['rouge_l'] +
                0.2 * df_summary['bert_similarity']
            )
            
            df_summary = df_summary.sort_values('composite_score', ascending=False)
            
            f.write("Rank | Configuration | Composite Score | Gen Time\n")
            f.write("-" * 55 + "\n")
            
            for i, (_, row) in enumerate(df_summary.iterrows()):
                f.write(f"{i+1:4d} | {row['description'][:25]:25s} | {row['composite_score']:13.3f} | {row['generation_time']:8.2f}s\n")
            
            f.write("\n")
        
        logger.info(f"Mistral performance analysis saved to {output_file}")

def main():
    """Main evaluation function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate RAG with DeepEval vs Claude baseline')
    parser.add_argument('--questions', required=True, help='Questions JSON file')
    parser.add_argument('--api-key', required=True, help='OpenAI API key for DeepEval')
    parser.add_argument('--output', help='Output CSV file')
    parser.add_argument('--max-questions', type=int, help='Max questions to evaluate')
    parser.add_argument('--collection', default='classes', help='Collection type')
    
    args = parser.parse_args()
    
    async def run_evaluation():
        evaluator = DeepEvalRAGEvaluator(openai_api_key=args.api_key)
        evaluator.rag_pipeline.set_collection_type(args.collection)
        
        results_df = await evaluator.run_evaluation_experiment(
            questions_file=args.questions,
            output_file=args.output,
            max_questions=args.max_questions
        )
        
        print(f"\nMISTRAL MODEL EVALUATION COMPLETED!")
        print(f"Results shape: {results_df.shape}")
        
        # Show key configuration performance
        claude_config = results_df[results_df['description'].str.contains("Claude's Configuration")]
        target_config = results_df[results_df['description'].str.contains("Your Target Configuration")]
        
        print(f"\nKEY CONFIGURATION PERFORMANCE:")
        if not claude_config.empty:
            print(f"Claude's Config (4096, 0.7, 0.9):")
            print(f"  Answer Relevancy: {claude_config['answer_relevancy'].mean():.3f}")
            print(f"  ROUGE-L: {claude_config['rouge_l'].mean():.3f}")
            print(f"  Generation Time: {claude_config['generation_time'].mean():.2f}s")
        
        if not target_config.empty:
            print(f"Your Target (1024, 0.5, 0.8):")
            print(f"  Answer Relevancy: {target_config['answer_relevancy'].mean():.3f}")
            print(f"  ROUGE-L: {target_config['rouge_l'].mean():.3f}")
            print(f"  Generation Time: {target_config['generation_time'].mean():.2f}s")
    
    asyncio.run(run_evaluation())

if __name__ == "__main__":
    main()
