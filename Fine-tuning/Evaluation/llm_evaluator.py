#!/usr/bin/env python3
"""
LLM-Based Human Evaluator using Grok
"""

import json
import logging
import time
import random
from typing import Dict, List
from openai import OpenAI

class LLMEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Setup evaluation criteria
        self.evaluation_criteria = {
            'accuracy': 'Is the documentation factually correct with respect to the ABAP code?',
            'completeness': 'Does it cover all essential elements (purpose, parameters, key logic)?',
            'clarity_readability': 'Is it easy to understand, grammatically correct, and well-structured?',
            'abap_specificity': 'Does it correctly use ABAP terms and understand common patterns?',
            'relevance_utility': 'Is it helpful for a developer trying to understand this code?',
            'conciseness': 'Is it appropriately concise without unnecessary fluff?',
            'hallucinations': 'Does it avoid inventing non-existent code details? (5=no hallucinations)'
        }
        
        # Setup LLM client
        self.setup_llm_client()
    
    def setup_llm_client(self):
        """Setup the LLM client (Grok via X.AI)."""
        evaluator_config = self.config.get('llm_evaluator', {})
        
        if evaluator_config.get('provider') == 'xai':
            # Grok via X.AI API
            self.client = OpenAI(
                api_key=evaluator_config['api_key'],
                base_url="https://api.x.ai/v1"
            )
            self.model_name = evaluator_config.get('model', 'grok-beta')
        else:
            # Default to OpenAI
            self.client = OpenAI(api_key=evaluator_config['api_key'])
            self.model_name = evaluator_config.get('model', 'gpt-4o')
        
        self.logger.info(f"✅ LLM evaluator setup: {self.model_name}")
    
    def create_evaluation_prompt(self, abap_code: str, generated_doc: str, 
                               ground_truth_doc: str, file_type: str) -> str:
        """Create evaluation prompt for the LLM."""
        
        criteria_text = ""
        for criterion, description in self.evaluation_criteria.items():
            criteria_text += f"- **{criterion}**: {description}\n"
        
        prompt = f"""You are an expert SAP ABAP developer reviewing auto-generated documentation.

## Evaluation Criteria (Rate each 1-5):
{criteria_text}

## ABAP Code ({file_type}):
```abap
{abap_code[:2000]}{'...' if len(abap_code) > 2000 else ''}
```

## Generated Documentation:
{generated_doc[:1500]}{'...' if len(generated_doc) > 1500 else ''}

## Reference Documentation:
{ground_truth_doc[:1500]}{'...' if len(ground_truth_doc) > 1500 else ''}

Rate each criterion 1-5 and respond in JSON format:

{{
    "accuracy": {{"score": <1-5>, "justification": "Brief explanation"}},
    "completeness": {{"score": <1-5>, "justification": "Brief explanation"}},
    "clarity_readability": {{"score": <1-5>, "justification": "Brief explanation"}},
    "abap_specificity": {{"score": <1-5>, "justification": "Brief explanation"}},
    "relevance_utility": {{"score": <1-5>, "justification": "Brief explanation"}},
    "conciseness": {{"score": <1-5>, "justification": "Brief explanation"}},
    "hallucinations": {{"score": <1-5>, "justification": "Brief explanation"}},
    "overall_assessment": "Overall quality assessment",
    "recommendation": "Yes/No/With modifications"
}}

Be objective and consistent. Focus on technical accuracy and practical utility."""

        return prompt
    
    def parse_llm_response(self, response_text: str) -> Dict:
        """Parse LLM response."""
        try:
            # Try to extract JSON
            import re
            
            # Look for JSON block
            json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try to find JSON without markdown
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    json_text = response_text[json_start:json_end]
                else:
                    json_text = response_text
            
            evaluation = json.loads(json_text)
            
            # Validate and fix scores
            for criterion in self.evaluation_criteria.keys():
                if criterion not in evaluation:
                    evaluation[criterion] = {"score": 3, "justification": "Not provided"}
                elif not isinstance(evaluation[criterion], dict):
                    score = evaluation[criterion] if isinstance(evaluation[criterion], (int, float)) else 3
                    evaluation[criterion] = {"score": score, "justification": "Converted"}
                
                # Ensure score is valid
                if not isinstance(evaluation[criterion]['score'], (int, float)) or evaluation[criterion]['score'] < 1 or evaluation[criterion]['score'] > 5:
                    evaluation[criterion]['score'] = 3
            
            return evaluation
            
        except Exception as e:
            self.logger.error(f"Failed to parse LLM response: {e}")
            return self.get_default_evaluation()
    
    def get_default_evaluation(self) -> Dict:
        """Get default evaluation when parsing fails."""
        evaluation = {}
        for criterion in self.evaluation_criteria.keys():
            evaluation[criterion] = {"score": 3, "justification": "Evaluation failed"}
        
        evaluation.update({
            "overall_assessment": "Evaluation failed due to parsing error",
            "recommendation": "Unable to determine"
        })
        
        return evaluation
    
    def evaluate_single_sample(self, sample: Dict) -> Dict:
        """Evaluate one sample."""
        try:
            prompt = self.create_evaluation_prompt(
                abap_code=sample.get('abap_code', ''),
                generated_doc=sample['generated_documentation'],
                ground_truth_doc=sample['ground_truth'],
                file_type=sample.get('file_type', 'unknown')
            )
            
            # Call LLM with retry
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are an expert SAP ABAP developer and documentation reviewer."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=1500
                    )
                    
                    response_text = response.choices[0].message.content
                    evaluation = self.parse_llm_response(response_text)
                    
                    # Add metadata
                    evaluation['sample_id'] = sample['sample_id']
                    evaluation['model'] = sample['model']
                    evaluation['file_type'] = sample.get('file_type', 'unknown')
                    
                    # Calculate overall score
                    scores = [evaluation[criterion]['score'] for criterion in self.evaluation_criteria.keys()]
                    evaluation['overall_score'] = sum(scores) / len(scores)
                    
                    return evaluation
                    
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        self.logger.warning(f"LLM call attempt {attempt + 1} failed, retrying in {wait_time:.1f}s: {e}")
                        time.sleep(wait_time)
                    else:
                        raise
                        
        except Exception as e:
            self.logger.error(f"Error evaluating sample {sample.get('sample_id', 'unknown')}: {e}")
            
            # Return default
            evaluation = self.get_default_evaluation()
            evaluation['sample_id'] = sample['sample_id']
            evaluation['model'] = sample['model']
            evaluation['file_type'] = sample.get('file_type', 'unknown')
            evaluation['overall_score'] = 3.0
            
            return evaluation
    
    def evaluate_model_results(self, model_results: List[Dict]) -> Dict:
        """Evaluate all results from a model."""
        self.logger.info(f"Starting LLM evaluation for {len(model_results)} samples...")
        
        evaluations = []
        
        for i, sample in enumerate(model_results):
            try:
                evaluation = self.evaluate_single_sample(sample)
                evaluations.append(evaluation)
                
                # Add delay to avoid rate limiting
                time.sleep(random.uniform(0.5, 2.0))
                
                if (i + 1) % 5 == 0:
                    self.logger.info(f"Evaluated {i + 1}/{len(model_results)} samples")
                    
            except Exception as e:
                self.logger.error(f"Failed to evaluate sample {sample.get('sample_id', i)}: {e}")
                continue
        
        # Calculate aggregates
        aggregate_stats = self.calculate_aggregate_statistics(evaluations)
        
        return {
            'individual_evaluations': evaluations,
            'aggregate_statistics': aggregate_stats,
            'num_evaluated': len(evaluations)
        }
    
    def calculate_aggregate_statistics(self, evaluations: List[Dict]) -> Dict[str, float]:
        """Calculate aggregate statistics."""
        if not evaluations:
            return {}
        
        stats = {}
        
        # Calculate stats for each criterion
        for criterion in self.evaluation_criteria.keys():
            scores = [eval_data[criterion]['score'] for eval_data in evaluations 
                     if criterion in eval_data and 'score' in eval_data[criterion]]
            
            if scores:
                stats[f'{criterion}_mean'] = sum(scores) / len(scores)
                stats[f'{criterion}_std'] = (sum((x - stats[f'{criterion}_mean']) ** 2 for x in scores) / len(scores)) ** 0.5
                stats[f'{criterion}_min'] = min(scores)
                stats[f'{criterion}_max'] = max(scores)
        
        # Overall score stats
        overall_scores = [eval_data['overall_score'] for eval_data in evaluations 
                         if 'overall_score' in eval_data]
        
        if overall_scores:
            stats['overall_score_mean'] = sum(overall_scores) / len(overall_scores)
            stats['overall_score_std'] = (sum((x - stats['overall_score_mean']) ** 2 for x in overall_scores) / len(overall_scores)) ** 0.5
        
        # Recommendation stats
        recommendations = [eval_data.get('recommendation', 'Unable to determine') for eval_data in evaluations]
        yes_count = recommendations.count('Yes')
        total_recs = len([r for r in recommendations if r != 'Unable to determine'])
        
        if total_recs > 0:
            stats['recommendation_yes_rate'] = yes_count / total_recs
        
        return stats