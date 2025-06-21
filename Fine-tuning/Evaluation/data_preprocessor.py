#!/usr/bin/env python3
"""
Data Preprocessor for ABAP Evaluation Framework
"""

import json
import logging
from typing import Dict, List
from pathlib import Path

class DataPreprocessor:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.max_tokens = config.get('max_context_tokens', 8000)
        
    def load_evaluation_dataset(self, dataset_path: str) -> List[Dict]:
        """Load evaluation dataset from JSON file."""
        self.logger.info(f"Loading dataset from: {dataset_path}")
        
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            self.logger.info(f"✅ Dataset loaded: {len(dataset)} samples")
            return dataset
            
        except FileNotFoundError:
            self.logger.error(f"Dataset file not found: {dataset_path}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Invalid JSON in dataset file: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def validate_dataset(self, dataset: List[Dict]) -> bool:
        """Validate dataset structure and content."""
        self.logger.info("Validating dataset...")
        
        if not dataset:
            self.logger.error("Dataset is empty")
            return False
        
        required_fields = ['id', 'abap_code', 'ground_truth_documentation']
        
        for i, sample in enumerate(dataset):
            # Check required fields
            for field in required_fields:
                if field not in sample:
                    self.logger.error(f"Sample {i}: Missing required field '{field}'")
                    return False
            
            # Check field content
            if not sample['abap_code'].strip():
                self.logger.error(f"Sample {i}: Empty ABAP code")
                return False
                
            if not sample['ground_truth_documentation'].strip():
                self.logger.error(f"Sample {i}: Empty documentation")
                return False
        
        self.logger.info(f"✅ Dataset validation passed: {len(dataset)} samples")
        return True
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Rough estimate: 1 token ≈ 4 characters for English text
        # ABAP code might be different, but this is a reasonable approximation
        return len(text) // 4
    
    def truncate_text(self, text: str, max_tokens: int) -> str:
        """Truncate text to approximately max_tokens."""
        estimated_tokens = self.estimate_tokens(text)
        
        if estimated_tokens <= max_tokens:
            return text
        
        # Calculate truncation point
        truncation_ratio = max_tokens / estimated_tokens
        truncation_point = int(len(text) * truncation_ratio)
        
        # Find a good breaking point (try to break at newlines)
        truncated = text[:truncation_point]
        last_newline = truncated.rfind('\n')
        
        if last_newline > truncation_point * 0.8:  # If newline is reasonably close
            truncated = truncated[:last_newline]
        
        return truncated
    
    def apply_8k_truncation(self, dataset: List[Dict]) -> List[Dict]:
        """Apply 8K context truncation for fair comparison."""
        self.logger.info("Applying 8K context truncation...")
        
        truncated_dataset = []
        
        for sample in dataset:
            truncated_sample = sample.copy()
            
            # Reserve tokens for documentation (roughly 30% of context)
            code_token_limit = int(self.max_tokens * 0.7)
            
            # Truncate ABAP code if needed
            original_code = sample['abap_code']
            truncated_code = self.truncate_text(original_code, code_token_limit)
            
            # Create different versions for different models
            truncated_sample['abap_code'] = truncated_code
            truncated_sample['abap_code_llama'] = truncated_code
            truncated_sample['abap_code_gemini'] = truncated_code
            
            # Track truncation
            original_tokens = self.estimate_tokens(original_code)
            truncated_tokens = self.estimate_tokens(truncated_code)
            
            truncated_sample['truncation_info'] = {
                'original_tokens': original_tokens,
                'truncated_tokens': truncated_tokens,
                'was_truncated': truncated_tokens < original_tokens,
                'truncation_ratio': truncated_tokens / original_tokens if original_tokens > 0 else 1.0
            }
            
            truncated_dataset.append(truncated_sample)
        
        # Log truncation statistics
        truncated_count = sum(1 for sample in truncated_dataset 
                            if sample['truncation_info']['was_truncated'])
        
        self.logger.info(f"✅ 8K truncation applied:")
        self.logger.info(f"  - Total samples: {len(truncated_dataset)}")
        self.logger.info(f"  - Truncated samples: {truncated_count}")
        self.logger.info(f"  - Truncation rate: {truncated_count/len(truncated_dataset)*100:.1f}%")
        
        return truncated_dataset