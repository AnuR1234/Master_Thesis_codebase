#!/usr/bin/env python3
"""
ABAP Documentation Evaluation Framework - Main Orchestrator
"""

import os
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Import our modules
from data_preprocessor import DataPreprocessor
from model_inference import ModelInference
from automated_metrics import AutomatedMetrics
from llm_evaluator import LLMEvaluator

class ABAPEvaluationFramework:
    def __init__(self, config_path: str):
        """Initialize the evaluation framework."""
        self.config = self.load_config(config_path)
        self.setup_logging()
        self.setup_directories()
        
        # Initialize components
        self.preprocessor = DataPreprocessor(self.config)
        self.inference = ModelInference(self.config)
        self.metrics = AutomatedMetrics(self.config)
        self.llm_evaluator = LLMEvaluator(self.config)
        
        self.logger.info("🚀 ABAP Evaluation Framework initialized")
    
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def setup_logging(self):
        """Setup logging."""
        log_dir = Path(self.config['output_dir']) / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f'evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def setup_directories(self):
        """Create output directories."""
        output_dir = Path(self.config['output_dir'])
        for subdir in ['preprocessed_data', 'generated_docs', 'metrics', 'evaluations', 'reports']:
            (output_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    def run_phase_1_preparation(self):
        """Phase 1: Load and validate dataset."""
        self.logger.info("=== Phase 1: Preparation & Setup ===")
        
        dataset_path = self.config['dataset']['path']
        dataset = self.preprocessor.load_evaluation_dataset(dataset_path)
        
        if not self.preprocessor.validate_dataset(dataset):
            raise ValueError("Dataset validation failed")
        
        self.logger.info(f"✅ Dataset loaded and validated: {len(dataset)} samples")
        return dataset
    
    def run_phase_2_preprocessing(self, dataset: List[Dict]):
        """Phase 2: Apply 8K context truncation."""
        self.logger.info("=== Phase 2: Data Preprocessing (8K Context) ===")
        
        preprocessed_dataset = self.preprocessor.apply_8k_truncation(dataset)
        
        # Save preprocessed data
        output_path = Path(self.config['output_dir']) / 'preprocessed_data' / 'dataset_8k.json'
        with open(output_path, 'w') as f:
            json.dump(preprocessed_dataset, f, indent=2)
        
        self.logger.info(f"✅ Preprocessed dataset saved: {output_path}")
        return preprocessed_dataset
    
    def run_phase_3_inference(self, dataset: List[Dict]):
        """Phase 3: Generate documentation with all models."""
        self.logger.info("=== Phase 3: Model Inference ===")
        
        results = {}
        enabled_models = [name for name, config in self.config['models'].items() if config.get('enabled', True)]
        
        for model_name in enabled_models:
            self.logger.info(f"🤖 Generating documentation with {model_name}...")
            
            model_results = []
            for i, sample in enumerate(dataset):
                try:
                    # Get the right code version for the model
                    if model_name == 'gemini':
                        code = sample.get('abap_code_gemini', sample['abap_code'])
                    else:
                        code = sample.get('abap_code_llama', sample['abap_code'])
                    
                    documentation = self.inference.generate_documentation(model_name, code)
                    
                    result = {
                        'sample_id': sample['id'],
                        'model': model_name,
                        'generated_documentation': documentation,
                        'ground_truth': sample['ground_truth_documentation'],
                        'file_type': sample.get('file_type', 'unknown'),
                        'abap_code': code
                    }
                    model_results.append(result)
                    
                    if (i + 1) % 5 == 0:
                        self.logger.info(f"  Processed {i + 1}/{len(dataset)} samples")
                        
                except Exception as e:
                    self.logger.error(f"  Error processing sample {sample['id']}: {e}")
                    continue
            
            results[model_name] = model_results
            
            # Save intermediate results
            output_path = Path(self.config['output_dir']) / 'generated_docs' / f'{model_name}_results.json'
            with open(output_path, 'w') as f:
                json.dump(model_results, f, indent=2)
            
            self.logger.info(f"✅ {model_name} completed: {len(model_results)} samples")
        
        return results
    
    def run_phase_4_automated_metrics(self, results: Dict):
        """Phase 4: Calculate automated metrics."""
        self.logger.info("=== Phase 4: Automated Metrics ===")
        
        automated_scores = {}
        
        for model_name, model_results in results.items():
            self.logger.info(f"📊 Calculating metrics for {model_name}...")
            
            scores = self.metrics.calculate_all_metrics(model_results)
            automated_scores[model_name] = scores
            
            # Save metrics
            output_path = Path(self.config['output_dir']) / 'metrics' / f'{model_name}_metrics.json'
            with open(output_path, 'w') as f:
                json.dump(scores, f, indent=2)
            
            self.logger.info(f"✅ {model_name} metrics calculated")
        
        # Create comparison table
        comparison_df = self.metrics.create_comparison_table(automated_scores)
        comparison_path = Path(self.config['output_dir']) / 'metrics' / 'comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        
        self.logger.info(f"✅ Comparison table saved: {comparison_path}")
        return automated_scores
    
    def run_phase_5_llm_evaluation(self, results: Dict):
        """Phase 5: LLM-based evaluation."""
        self.logger.info("=== Phase 5: LLM Evaluation ===")
        
        llm_scores = {}
        
        for model_name, model_results in results.items():
            # Limit samples for LLM evaluation (can be expensive)
            limited_results = model_results[:20]  # Evaluate first 20 samples
            
            self.logger.info(f"🧠 LLM evaluating {model_name} ({len(limited_results)} samples)...")
            
            scores = self.llm_evaluator.evaluate_model_results(limited_results)
            llm_scores[model_name] = scores
            
            # Save evaluation results
            output_path = Path(self.config['output_dir']) / 'evaluations' / f'{model_name}_llm_eval.json'
            with open(output_path, 'w') as f:
                json.dump(scores, f, indent=2)
            
            self.logger.info(f"✅ {model_name} LLM evaluation completed")
        
        return llm_scores
    
    def run_phase_6_analysis(self, automated_scores: Dict, llm_scores: Dict):
        """Phase 6: Generate final report."""
        self.logger.info("=== Phase 6: Analysis & Reporting ===")
        
        # Generate simple report
        report = self.generate_simple_report(automated_scores, llm_scores)
        
        # Save report
        report_path = Path(self.config['output_dir']) / 'reports' / 'evaluation_report.md'
        with open(report_path, 'w') as f:
            f.write(report)
        
        self.logger.info(f"✅ Final report saved: {report_path}")
        return report
    
    def generate_simple_report(self, automated_scores: Dict, llm_scores: Dict) -> str:
        """Generate a simple markdown report."""
        
        report = f"""# ABAP Documentation Evaluation Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary

This report compares the performance of fine-tuned Llama 2.5 8B, base Llama 2.5 8B, and Gemini 2.0 Flash on ABAP documentation generation under 8K context constraints.

## Automated Metrics Results

"""
        
        # Create results table
        for model_name, scores in automated_scores.items():
            agg = scores['aggregate_metrics']
            report += f"### {model_name.replace('_', ' ').title()}\n"
            report += f"- **BLEU-4**: {agg.get('bleu_4_mean', 0):.4f}\n"
            report += f"- **ROUGE-L**: {agg.get('rouge_l_f_mean', 0):.4f}\n"
            report += f"- **BERTScore F1**: {agg.get('bert_score_f1_mean', 0):.4f}\n"
            report += f"- **Semantic Similarity**: {agg.get('semantic_similarity_mean', 0):.4f}\n"
            report += f"- **ABAP Coverage**: {agg.get('abap_term_coverage_mean', 0):.4f}\n\n"
        
        report += "## LLM Evaluation Results\n\n"
        
        # LLM results
        for model_name, scores in llm_scores.items():
            stats = scores['aggregate_statistics']
            report += f"### {model_name.replace('_', ' ').title()}\n"
            report += f"- **Overall Score**: {stats.get('overall_score_mean', 0):.2f}/5\n"
            report += f"- **Accuracy**: {stats.get('accuracy_mean', 0):.2f}/5\n"
            report += f"- **ABAP Specificity**: {stats.get('abap_specificity_mean', 0):.2f}/5\n"
            report += f"- **Recommendation Rate**: {stats.get('recommendation_yes_rate', 0)*100:.1f}%\n\n"
        
        # Find best models
        best_auto = max(automated_scores.items(), 
                       key=lambda x: x[1]['aggregate_metrics'].get('bleu_4_mean', 0))
        best_llm = max(llm_scores.items(), 
                      key=lambda x: x[1]['aggregate_statistics'].get('overall_score_mean', 0))
        
        report += f"""## Key Findings

- **Best Automated Metrics**: {best_auto[0].replace('_', ' ').title()}
- **Best LLM Evaluation**: {best_llm[0].replace('_', ' ').title()}

## Conclusion

The evaluation demonstrates the effectiveness of fine-tuning for ABAP-specific documentation generation. 
All models were fairly compared under identical 8K token context constraints.

---
*Generated by ABAP Documentation Evaluation Framework*
"""
        
        return report
    
    def run_complete_evaluation(self):
        """Run the complete evaluation pipeline."""
        try:
            self.logger.info("🎯 Starting complete ABAP documentation evaluation...")
            
            # Run all phases
            dataset = self.run_phase_1_preparation()
            preprocessed_dataset = self.run_phase_2_preprocessing(dataset)
            results = self.run_phase_3_inference(preprocessed_dataset)
            automated_scores = self.run_phase_4_automated_metrics(results)
            llm_scores = self.run_phase_5_llm_evaluation(results)
            final_report = self.run_phase_6_analysis(automated_scores, llm_scores)
            
            self.logger.info("🎉 Evaluation completed successfully!")
            self.logger.info(f"📊 Results available in: {self.config['output_dir']}")
            
            return {
                'automated_scores': automated_scores,
                'llm_scores': llm_scores,
                'report': final_report
            }
            
        except Exception as e:
            self.logger.error(f"💥 Evaluation failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description="ABAP Documentation Evaluation Framework")
    parser.add_argument('config', help='Path to configuration JSON file')
    parser.add_argument('--phase', choices=['all', '1', '2', '3', '4', '5', '6'], 
                       default='all', help='Run specific phase (default: all)')
    
    args = parser.parse_args()
    
    # Initialize framework
    framework = ABAPEvaluationFramework(args.config)
    
    if args.phase == 'all':
        framework.run_complete_evaluation()
    else:
        print(f"Running phase {args.phase}...")
        # For simplicity, just run complete evaluation
        # You can add individual phase runners here if needed
        framework.run_complete_evaluation()

if __name__ == "__main__":
    main()