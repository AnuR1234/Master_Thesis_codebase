#!/usr/bin/env python3
"""
Model Inference for ABAP Documentation Generation
"""

import torch
import re
import logging
import time
import random
from typing import Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import google.generativeai as genai

# ABAP Documentation Templates (Your exact specifications)
CLASS_TEMPLATE = """## Overview
[Brief description of the class purpose and functionality]
## Class Definition
| Aspect | Details |
|--------|---------| 
| Scope | [PUBLIC/PROTECTED/PRIVATE] |
| Type | [FINAL/ABSTRACT] |
| Superclass | [Parent class name] |
| Create Permission | [PUBLIC/PROTECTED/PRIVATE] |
| Friends | [List of friend classes] |
## Implementation Overview
[General description of the implementation approach]
## Method Dependencies
| Method | Calls | Called By |
|--------|-------|-----------|
| [Method name] | [Methods called by this method] | [Methods calling this method] |
## Redefined Methods
| Method | Source | Implementation Details |
|--------|--------|----------------------|
| [Method name] | [Original class] | [Description of implementation] |
## Database Tables Used
| Table Name | Purpose | Key Fields |
|------------|---------|------------|
| [Table name] | [Description of usage] | [Key fields used] |
## Critical Sections
| Section | Methods Involved | Purpose | Considerations |
|---------|-----------------|---------|----------------|
| [Section name] | [List of methods] | [Purpose] | [Special considerations] |
## Method Implementation Details
#### [Method name]
- Logic Flow: [Step-by-step description of the method logic]
- Error Handling: [Description of error handling in the method]
- Dependencies: [Methods or functions this method depends on]
- Key Variables: [Important variables and their purposes]
"""

TEST_CLASS_TEMPLATE = """## Overview
[Brief description of the test class purpose and what it tests]
## Class Definition
| Aspect | Details |
|--------|---------| 
| Scope | [PUBLIC/PROTECTED/PRIVATE] |
| Type | [FINAL] |
| Risk Level | [HARMLESS/DANGEROUS/CRITICAL] |
| Duration | [SHORT/MEDIUM/LONG] |
| Test For | [Class or component being tested] |
## Test Methods
| Method | Purpose |
|--------|---------| 
| [Test method name] | [Description of what this test verifies] |
## Test Data
| Variable | Type | Purpose |
|----------|------|---------|
| [Variable name] | [Data type] | [Purpose of this test data] |
## Mocks and Test Doubles
| Mock Object | Real Object | Purpose |
|-------------|------------|---------|
| [Mock name] | [Real object name] | [Purpose of this mock] |
## Test Implementation Overview
[General description of the testing approach]
## Test Method Dependencies
| Method | Calls | Setup Requirements |
|--------|-------|-------------------|
| [Method name] | [Methods called by this test] | [Required setup for this test] |
## Test-Injection Points
| Injection Point | Purpose | Methods Using |
|-----------------|---------|---------------|
| [Injection name] | [Purpose of this injection] | [Methods using this injection] |
## Method Implementation Details
#### [Method name]
- Test Scenario: [Description of the test scenario]
- Assertions: [List of assertions made in this test]
- Dependencies: [Methods or functions this test depends on]
- Key Variables: [Important variables and their purposes]
"""

REPORT_TEMPLATE = """# Technical Documentation: [REPORT_NAME]
## Overview
[Brief description of the report's purpose and functionality]
## Program Flow
1. **INITIALIZATION**: [Description of initialization steps]
2. **START-OF-SELECTION**: [Description of main program steps]
   - [Sub-step]
   - [Sub-step]
## Data Structures
- [Structure/Table name]: [Description]
## Selection Screen
- [Parameter/Select-option name]: [Description and default value]
## Main Processing Logic
1. [Processing step 1]
2. [Processing step 2]
## Subroutines (Forms)
[List of forms if any, or "No forms are defined in this program"]
## Function Module Usage
[Detailed documentation of function modules used]
## Database Interaction
[Documentation of database tables accessed and operations performed]
"""

FUNCTION_TEMPLATE = """# Technical Documentation: [FUNCTION_NAME]
## Overview
[Brief description of the function's purpose]
## Function Module Definition
| Aspect | Details |
|--------|---------| 
| Function Group | [Group name] |
| Processing Type | [RFC/BAPI/Regular] |
## Interface
### Importing Parameters
| Parameter | Type | Optional | Description |
|-----------|------|----------|-------------|
| [Parameter name] | [Data type] | [Yes/No] | [Description] |
### Exporting Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| [Parameter name] | [Data type] | [Description] |
### Changing Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| [Parameter name] | [Data type] | [Description] |
### Tables Parameters
| Parameter | Type | Description |
|-----------|------|-------------|
| [Parameter name] | [Data type] | [Description] |
### Exceptions
| Exception | Description | Triggered When |
|-----------|-------------|---------------|
| [Exception name] | [Description] | [Conditions for triggering] |
## Implementation Overview
[General description of the implementation approach]
## Processing Logic
1. [Processing step 1]
2. [Processing step 2]
## Database Interaction
[Documentation of database tables accessed and operations performed]
"""

class ModelInference:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.tokenizer = None
        
        # Setup models
        self.setup_models()
    
    def setup_models(self):
        """Initialize all models."""
        try:
            # Setup Gemini
            if self.config['models']['gemini']['enabled']:
                genai.configure(api_key=self.config['models']['gemini']['api_key'])
                self.models['gemini'] = genai.GenerativeModel(
                    self.config['models']['gemini']['model_name']
                )
                self.logger.info("✅ Gemini model initialized")
            
            # Setup Llama models
            if (self.config['models']['base_llama']['enabled'] or 
                self.config['models']['fine_tuned_llama']['enabled']):
                self.setup_llama_models()
                
        except Exception as e:
            self.logger.error(f"Error setting up models: {e}")
            raise
    
    def setup_llama_models(self):
        """Setup Llama models."""
        try:
            base_path = self.config['models']['base_llama']['model_path']
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(base_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load base model if enabled
            if self.config['models']['base_llama']['enabled']:
                self.logger.info("Loading base Llama model...")
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    base_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True
                )
                self.models['base_llama'] = self.base_model
                self.logger.info("✅ Base Llama model loaded")
            
            # Load fine-tuned model if enabled
            if self.config['models']['fine_tuned_llama']['enabled']:
                adapter_path = self.config['models']['fine_tuned_llama']['adapter_path']
                self.logger.info("Loading fine-tuned model with LoRA...")
                
                # If base model wasn't loaded, load it now
                if not hasattr(self, 'base_model'):
                    self.base_model = AutoModelForCausalLM.from_pretrained(
                        base_path,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                
                self.fine_tuned_model = PeftModel.from_pretrained(self.base_model, adapter_path)
                self.fine_tuned_model.eval()
                self.models['fine_tuned_llama'] = self.fine_tuned_model
                self.logger.info("✅ Fine-tuned Llama model loaded")
                
        except Exception as e:
            self.logger.error(f"Error setting up Llama models: {e}")
            raise
    
    def detect_abap_type(self, code: str) -> str:
        """Detect ABAP file type."""
        if re.search(r'(?i)CLASS\s+\w+\s+DEFINITION.*?\bFOR\s+TESTING\b', code, re.DOTALL):
            return "test_class"
        elif re.search(r'(?i)CLASS\s+[\w/]+\s+DEFINITION', code):
            return "class"
        elif re.search(r'(?i)REPORT\s+[\w/]+', code):
            return "report"
        elif re.search(r'(?i)FUNCTION\s+[\w/]+', code):
            return "function"
        return "unknown"
    
    def create_prompt(self, code: str) -> str:
        """Create prompt for documentation generation using the correct template."""
        file_type = self.detect_abap_type(code)
        
        # Select the appropriate template
        if file_type == "test_class":
            template = TEST_CLASS_TEMPLATE
            description = "test class"
        elif file_type in ["class", "final_class", "abstract_class"]:
            template = CLASS_TEMPLATE
            description = "class"
        elif file_type == "report":
            template = REPORT_TEMPLATE
            description = "report"
        elif file_type == "function":
            template = FUNCTION_TEMPLATE
            description = "function module"
        else:
            template = CLASS_TEMPLATE  # Default fallback
            description = "ABAP code"
        
        instruction = f"""Generate comprehensive technical documentation for the following SAP ABAP {description} code. 
Use EXACTLY the template format provided below. Fill in all sections appropriately based on the code analysis.

Template to follow:
{template}

ABAP Code to document:
```abap
{code}
```

Generate the documentation following the exact template structure:"""
        
        return instruction
    
    def generate_with_llama(self, model, code: str) -> str:
        """Generate documentation using Llama models."""
        try:
            # Create prompt
            instruction = self.create_prompt(code)
            
            # Format with Llama template
            prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an SAP ABAP documentation assistant. Generate accurate and comprehensive documentation.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{instruction}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=7000)
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=2000,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # Decode
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract response
            if "<|start_header_id|>assistant<|end_header_id|>" in generated_text:
                response = generated_text.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
            else:
                response = generated_text.split(prompt)[-1].strip() if prompt in generated_text else generated_text
            
            # Clean up
            response = re.sub(r'<\|[^>]+\|>', '', response).strip()
            
            return response if response else "Error: Could not generate documentation"
            
        except Exception as e:
            self.logger.error(f"Error generating with Llama: {e}")
            return f"Error generating documentation: {e}"
    
    def generate_with_gemini(self, code: str) -> str:
        """Generate documentation using Gemini."""
        try:
            instruction = self.create_prompt(code)
            
            # Add retry logic for API calls
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.models['gemini'].generate_content(
                        instruction,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.7,
                            max_output_tokens=2000,
                        )
                    )
                    return response.text
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = (2 ** attempt) + random.uniform(0, 1)
                        self.logger.warning(f"Gemini API attempt {attempt + 1} failed, retrying in {wait_time:.1f}s: {e}")
                        time.sleep(wait_time)
                    else:
                        raise
                        
        except Exception as e:
            self.logger.error(f"Error generating with Gemini: {e}")
            return f"Error generating with Gemini: {e}"
    
    def generate_documentation(self, model_name: str, code: str) -> str:
        """Generate documentation using specified model."""
        try:
            if model_name == 'gemini':
                return self.generate_with_gemini(code)
            elif model_name == 'base_llama':
                return self.generate_with_llama(self.models['base_llama'], code)
            elif model_name == 'fine_tuned_llama':
                return self.generate_with_llama(self.models['fine_tuned_llama'], code)
            else:
                raise ValueError(f"Unknown model: {model_name}")
        except Exception as e:
            self.logger.error(f"Error in generate_documentation for {model_name}: {e}")
            return f"Error: {e}"