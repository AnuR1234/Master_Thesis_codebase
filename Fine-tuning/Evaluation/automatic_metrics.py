#!/usr/bin/env python3
"""
Automated Metrics Calculator
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List
from collections import defaultdict

# Try to import evaluation libraries
try:
    from rouge_score import rouge_scorer
    from bert_score import score as bert_score
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from nltk.tokenize import word_tokenize
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        
except ImportError as e:
    print(f"Some metrics libraries not available: {e}")
    print("Install with: pip install rouge-score bert-score sentence-transformers nltk scikit-learn")

class AutomatedMetrics:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        try:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.smoothing = SmoothingFunction()
            self.logger.info("✅ Automated metrics initialized")
        except Exception as e:
            self.logger.error(f"Error initializing metrics: {e}")
            raise
    
    def calculate_bleu_score(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate BLEU scores."""
        try:
            ref_tokens = word_tokenize(reference.lower())
            hyp_tokens = word_tokenize(hypothesis.lower())
            
            return {
                'bleu_1': sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=self.smoothing.method1),
                'bleu_2': sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoothing.method1),
                'bleu_4': sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoothing.method1)
            }
        except Exception as e:
            self.logger.error(f"Error calculating BLEU: {e}")
            return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_4': 0.0}
    
    def calculate_rouge_scores(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate ROUGE scores."""
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return {
                'rouge_1_f': scores['rouge1'].fmeasure,
                'rouge_2_f': scores['rouge2'].fmeasure,
                'rouge_l_f': scores['rougeL'].fmeasure,
            }
        except Exception as e:
            self.logger.error(f"Error calculating ROUGE: {e}")
            return {'rouge_1_f': 0.0, 'rouge_2_f': 0.0, 'rouge_l_f': 0.0}
    
    def calculate_bert_score(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate BERTScore."""
        try:
            P, R, F1 = bert_score([hypothesis], [reference], lang='en', verbose=False)
            return {
                'bert_score_p': float(P[0]),
                'bert_score_r': float(R[0]),
                'bert_score_f1': float(F1[0])
            }
        except Exception as e:
            self.logger.error(f"Error calculating BERTScore: {e}")
            return {'bert_score_p': 0.0, 'bert_score_r': 0.0, 'bert_score_f1': 0.0}
    
    def calculate_semantic_similarity(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate semantic similarity."""
        try:
            ref_embedding = self.sentence_model.encode([reference])
            hyp_embedding = self.sentence_model.encode([hypothesis])
            similarity = cosine_similarity(ref_embedding, hyp_embedding)[0][0]
            return {'semantic_similarity': float(similarity)}
        except Exception as e:
            self.logger.error(f"Error calculating semantic similarity: {e}")
            return {'semantic_similarity': 0.0}
    
    def calculate_abap_metrics(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate ABAP-specific metrics."""
        try:
            # Extract ABAP terms
            abap_keywords = {'CLASS', 'METHOD', 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 
                           'FORM', 'FUNCTION', 'BAPI', 'ABAP', 'SAP', 'TABLE'}
            
            ref_words = set(reference.upper().split())
            hyp_words = set(hypothesis.upper().split())
            
            ref_abap_terms = ref_words.intersection(abap_keywords)
            hyp_abap_terms = hyp_words.intersection(abap_keywords)
            
            # Calculate coverage
            if ref_abap_terms:
                coverage = len(ref_abap_terms.intersection(hyp_abap_terms)) / len(ref_abap_terms)
            else:
                coverage = 1.0
            
            # Calculate precision  
            if hyp_abap_terms:
                precision = len(ref_abap_terms.intersection(hyp_abap_terms)) / len(hyp_abap_terms)
            else:
                precision = 0.0
            
            return {
                'abap_term_coverage': coverage,
                'abap_term_precision': precision
            }
        except Exception as e:
            self.logger.error(f"Error calculating ABAP metrics: {e}")
            return {'abap_term_coverage': 0.0, 'abap_term_precision': 0.0}
    
    def calculate_sample_metrics(self, sample: Dict) -> Dict[str, float]:
        """Calculate all metrics for one sample."""
        reference = sample['ground_truth']
        hypothesis = sample['generated_documentation']
        
        all_metrics = {}
        
        # Calculate all metric types
        all_metrics.update(self.calculate_bleu_score(reference, hypothesis))
        all_metrics.update(self.calculate_rouge_scores(reference, hypothesis))
        all_metrics.update(self.calculate_bert_score(reference, hypothesis))
        all_metrics.update(self.calculate_semantic_similarity(reference, hypothesis))
        all_metrics.update(self.calculate_abap_metrics(reference, hypothesis))
        
        return all_metrics
    
    def calculate_all_metrics(self, model_results: List[Dict]) -> Dict:
        """Calculate metrics for all samples."""
        self.logger.info(f"Calculating metrics for {len(model_results)} samples...")
        
        all_sample_metrics = []
        
        for i, sample in enumerate(model_results):
            try:
                sample_metrics = self.calculate_sample_metrics(sample)
                sample_metrics['sample_id'] = sample['sample_id']
                all_sample_metrics.append(sample_metrics)
                
                if (i + 1) % 10 == 0:
                    self.logger.info(f"Processed {i + 1}/{len(model_results)} samples")
                    
            except Exception as e:
                self.logger.error(f"Error processing sample {sample.get('sample_id', i)}: {e}")
                continue
        
        # Calculate aggregates
        aggregate_metrics = self.calculate_aggregate_metrics(all_sample_metrics)
        
        return {
            'individual_metrics': all_sample_metrics,
            'aggregate_metrics': aggregate_metrics,
            'num_samples': len(all_sample_metrics)
        }
    
    def calculate_aggregate_metrics(self, sample_metrics: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate statistics."""
        if not sample_metrics:
            return {}
        
        metric_names = [key for key in sample_metrics[0].keys() if key != 'sample_id']
        aggregate = {}
        
        for metric in metric_names:
            values = [sample[metric] for sample in sample_metrics if metric in sample]
            
            if values:
                aggregate[f'{metric}_mean'] = np.mean(values)
                aggregate[f'{metric}_std'] = np.std(values)
                aggregate[f'{metric}_min'] = np.min(values)
                aggregate[f'{metric}_max'] = np.max(values)
        
        return aggregate
    
    def create_comparison_table(self, all_model_scores: Dict) -> pd.DataFrame:
        """Create comparison table across models."""
        comparison_data = []
        
        # Get metrics from first model
        first_model = list(all_model_scores.keys())[0]
        metrics = [key for key in all_model_scores[first_model]['aggregate_metrics'].keys() 
                  if key.endswith('_mean')]
        
        for metric in metrics:
            metric_base = metric.replace('_mean', '')
            row = {'metric': metric_base}
            
            for model_name, scores in all_model_scores.items():
                mean_val = scores['aggregate_metrics'].get(metric, 0.0)
                std_val = scores['aggregate_metrics'].get(f'{metric_base}_std', 0.0)
                row[f'{model_name}_mean'] = mean_val
                row[f'{model_name}_std'] = std_val
                row[f'{model_name}_formatted'] = f"{mean_val:.4f} ± {std_val:.4f}"
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)