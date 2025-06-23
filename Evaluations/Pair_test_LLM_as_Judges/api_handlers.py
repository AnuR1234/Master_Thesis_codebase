#!/usr/bin/env python3
"""
API Handlers for LLM Judge Models
Handles real API calls to Gemini, Grok-3, and Qwen
Updated with current model names as of June 2025
"""

import json
import time
from typing import Dict, Optional
import requests

class RealJudgeAPIs:
    """Real API implementations for judge models"""
    
    def __init__(self, gemini_api_key: Optional[str] = None, grok_api_key: Optional[str] = None, 
                 qwen_api_provider: str = "together", qwen_api_key: Optional[str] = None):
        """
        Initialize with API keys
        
        Args:
            gemini_api_key: Your Google AI API key
            grok_api_key: Your xAI API key  
            qwen_api_provider: "together", "huggingface", or "custom"
            qwen_api_key: API key for chosen Qwen provider
        """
        self.gemini_api_key = gemini_api_key
        self.grok_api_key = grok_api_key
        self.qwen_api_provider = qwen_api_provider
        self.qwen_api_key = qwen_api_key
        
        # Initialize Gemini with available models (try best to fallback)
        if gemini_api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=gemini_api_key)
                
                # Try models in order of preference
                model_preferences = [
                    'gemini-1.5-pro-002',           # Latest stable 1.5 Pro
                    'gemini-1.5-pro',               # Stable 1.5 Pro
                    'gemini-2.0-flash',             # Latest 2.0 Flash
                    'gemini-1.5-flash-002',         # Latest stable Flash
                    'gemini-1.5-flash',             # Stable Flash
                    'gemini-2.5-pro-preview-05-06'  # Preview 2.5 Pro (if available)
                ]
                
                self.gemini_model = None
                for model_name in model_preferences:
                    try:
                        self.gemini_model = genai.GenerativeModel(model_name)
                        # Test the model with a simple query
                        test_response = self.gemini_model.generate_content("Test")
                        print(f"✓ {model_name} initialized successfully")
                        break
                    except Exception as e:
                        print(f"⚠️  {model_name} failed: {str(e)[:100]}")
                        continue
                
                if not self.gemini_model:
                    print("❌ All Gemini models failed to initialize")
                    
            except ImportError:
                print("❌ google-generativeai not installed. Run: pip install google-generativeai")
                self.gemini_model = None
            except Exception as e:
                print(f"❌ Gemini setup failed: {e}")
                self.gemini_model = None
        else:
            self.gemini_model = None
        
        # Test other connections
        if grok_api_key:
            print("✓ Grok-3 API key provided")
        if qwen_api_key:
            print(f"✓ Qwen API key provided for {qwen_api_provider}")
    
    def call_gemini_2_5_pro(self, prompt: str, retry_count: int = 3) -> Dict:
        """Call Gemini 2.5 Pro (or 1.5 Pro fallback) with enhanced ABAP code evaluation"""
        
        if not self.gemini_model:
            return self._fallback_response("gemini-2.5-pro", "API not initialized")
        
        for attempt in range(retry_count):
            try:
                import google.generativeai as genai
                
                # Configure for optimal ABAP code evaluation
                generation_config = genai.types.GenerationConfig(
                    max_output_tokens=2000,
                    temperature=0.1,  # Low temperature for consistent technical evaluation
                    top_p=0.8,
                    top_k=40
                )
                
                safety_settings = [
                    {
                        "category": "HARM_CATEGORY_HARASSMENT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_HATE_SPEECH", 
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                        "threshold": "BLOCK_NONE"
                    },
                    {
                        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                        "threshold": "BLOCK_NONE"
                    }
                ]
                
                response = self.gemini_model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                # Extract JSON from response
                response_text = response.text.strip()
                
                # Clean up response if needed
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                
                return json.loads(response_text)
                
            except json.JSONDecodeError as e:
                print(f"❌ Gemini JSON parsing error (attempt {attempt + 1}): {e}")
                if attempt == retry_count - 1:
                    return self._fallback_response("gemini-2.5-pro", f"JSON parsing failed: {e}")
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Gemini API error (attempt {attempt + 1}): {e}")
                if attempt == retry_count - 1:
                    return self._fallback_response("gemini-2.5-pro", f"API error: {e}")
                time.sleep(2)
        
        return self._fallback_response("gemini-2.5-pro", "Max retries exceeded")
    
    def call_grok_3(self, prompt: str, retry_count: int = 3) -> Dict:
        """Call Grok-3 via xAI API"""
        
        if not self.grok_api_key:
            return self._fallback_response("grok-3", "API key not provided")
        
        headers = {
            "Authorization": f"Bearer {self.grok_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "grok-3",  # Use Grok 3 specifically
            "messages": [
                {
                    "role": "system", 
                    "content": "You are an expert ABAP developer and technical evaluator. Provide precise, structured analysis in valid JSON format."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1,
            "stream": False
        }
        
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    content = response_data["choices"][0]["message"]["content"]
                    
                    # Clean up response
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    
                    return json.loads(content)
                    
                else:
                    print(f"❌ Grok API error {response.status_code}: {response.text}")
                    if attempt == retry_count - 1:
                        return self._fallback_response("grok-3", f"HTTP {response.status_code}")
                    
            except json.JSONDecodeError as e:
                print(f"❌ Grok JSON parsing error (attempt {attempt + 1}): {e}")
                if attempt == retry_count - 1:
                    return self._fallback_response("grok-3", f"JSON parsing failed: {e}")
                    
            except Exception as e:
                print(f"❌ Grok API error (attempt {attempt + 1}): {e}")
                if attempt == retry_count - 1:
                    return self._fallback_response("grok-3", f"API error: {e}")
            
            time.sleep(2)
        
        return self._fallback_response("grok-3", "Max retries exceeded")
    
    def call_qwen_2_5_72b(self, prompt: str, retry_count: int = 3) -> Dict:
        """Call Qwen 2.5-Coder-32B-Instruct (Open Source Code-Specialized Model)"""
        
        if self.qwen_api_provider == "together":
            return self._call_qwen_together(prompt, retry_count)
        elif self.qwen_api_provider == "huggingface":
            return self._call_qwen_huggingface(prompt, retry_count)
        elif self.qwen_api_provider == "custom":
            return self._call_qwen_custom(prompt, retry_count)
        else:
            return self._fallback_response("qwen-2.5-coder-32b", "Invalid provider")
    
    def _call_qwen_together(self, prompt: str, retry_count: int) -> Dict:
        """Call Qwen 2.5-Coder-32B via Together AI (Code-specialized model)"""
        
        if not self.qwen_api_key:
            return self._fallback_response("qwen-2.5-coder-32b", "Together AI API key not provided")
        
        headers = {
            "Authorization": f"Bearer {self.qwen_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "Qwen/Qwen2.5-Coder-32B-Instruct",  # Code-specialized model
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert ABAP developer and code analyst. Analyze the given responses with technical precision focusing on code quality, ABAP best practices, and return valid JSON."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1,
            "top_p": 0.8,
            "repetition_penalty": 1.1
        }
        
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    "https://api.together.xyz/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    content = response_data["choices"][0]["message"]["content"]
                    
                    # Clean and parse JSON
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    
                    return json.loads(content)
                    
                else:
                    print(f"❌ Together AI error {response.status_code}: {response.text}")
                    
            except Exception as e:
                print(f"❌ Qwen (Together) error (attempt {attempt + 1}): {e}")
                if attempt == retry_count - 1:
                    return self._fallback_response("qwen-2.5-coder-32b", f"Together AI error: {e}")
            
            time.sleep(2)
        
        return self._fallback_response("qwen-2.5-coder-32b", "Together AI max retries exceeded")
    
    def _call_qwen_huggingface(self, prompt: str, retry_count: int) -> Dict:
        """Call Qwen 2.5-Coder-32B via Hugging Face Inference API"""
        
        if not self.qwen_api_key:
            return self._fallback_response("qwen-2.5-coder-32b", "HF API key not provided")
        
        headers = {
            "Authorization": f"Bearer {self.qwen_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 2000,
                "temperature": 0.1,
                "top_p": 0.8,
                "return_full_text": False
            }
        }
        
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-Coder-32B-Instruct",
                    headers=headers,
                    json=data,
                    timeout=120
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    
                    # Handle different HF response formats
                    if isinstance(response_data, list) and len(response_data) > 0:
                        content = response_data[0].get("generated_text", "")
                    else:
                        content = response_data.get("generated_text", "")
                    
                    # Clean and parse JSON
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    
                    return json.loads(content)
                    
                else:
                    print(f"❌ Hugging Face error {response.status_code}: {response.text}")
                    
            except Exception as e:
                print(f"❌ Qwen (HF) error (attempt {attempt + 1}): {e}")
                if attempt == retry_count - 1:
                    return self._fallback_response("qwen-2.5-coder-32b", f"HF error: {e}")
            
            time.sleep(3)  # HF might need longer delays
        
        return self._fallback_response("qwen-2.5-coder-32b", "HF max retries exceeded")
    
    def _call_qwen_custom(self, prompt: str, retry_count: int) -> Dict:
        """Call Qwen via custom deployment"""
        
        # Example for custom deployment - modify as needed
        custom_endpoint = "https://your-qwen-deployment.com/v1/chat/completions"
        
        headers = {
            "Authorization": f"Bearer {self.qwen_api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "qwen2.5-coder-32b-instruct",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        for attempt in range(retry_count):
            try:
                response = requests.post(
                    custom_endpoint,
                    headers=headers,
                    json=data,
                    timeout=60
                )
                
                if response.status_code == 200:
                    response_data = response.json()
                    content = response_data["choices"][0]["message"]["content"]
                    
                    # Clean and parse JSON
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.endswith("```"):
                        content = content[:-3]
                    
                    return json.loads(content)
                    
            except Exception as e:
                print(f"❌ Qwen (Custom) error (attempt {attempt + 1}): {e}")
                if attempt == retry_count - 1:
                    return self._fallback_response("qwen-2.5-coder-32b", f"Custom deployment error: {e}")
            
            time.sleep(2)
        
        return self._fallback_response("qwen-2.5-coder-32b", "Custom deployment max retries exceeded")
    
    def _fallback_response(self, model_name: str, error_message: str) -> Dict:
        """Generate fallback response when API calls fail"""
        
        print(f"⚠️  Using fallback for {model_name}: {error_message}")
        
        return {
            "winner": "Tie",  # Conservative choice when API fails
            "confidence": 0.5,
            "scores": {
                "A": {
                    "technical_accuracy": 7,
                    "sap_domain_knowledge": 7,
                    "completeness": 7,
                    "clarity": 7,
                    "practical_value": 7
                },
                "B": {
                    "technical_accuracy": 7,
                    "sap_domain_knowledge": 7,
                    "completeness": 7,
                    "clarity": 7,
                    "practical_value": 7
                }
            },
            "reasoning": f"API error for {model_name}: {error_message}. Using neutral fallback scores.",
            "question_type_assessment": f"API unavailable for proper assessment."
        }

# Test function to verify API connections
def test_apis():
    """Test all API connections"""
    
    print("Testing API connections...")
    
    # Initialize APIs (replace with your actual keys)
    apis = RealJudgeAPIs(
        gemini_api_key=None,  # Add your key here
        grok_api_key=None,    # Add your key here
        qwen_api_provider="together",
        qwen_api_key=None     # Add your key here
    )
    
    test_prompt = """
    Test prompt for API validation.
    
    Please respond with valid JSON:
    {
        "winner": "A",
        "confidence": 0.8,
        "scores": {
            "A": {"technical_accuracy": 8, "sap_domain_knowledge": 7, "completeness": 8, "clarity": 8, "practical_value": 7},
            "B": {"technical_accuracy": 7, "sap_domain_knowledge": 6, "completeness": 7, "clarity": 7, "practical_value": 6}
        },
        "reasoning": "Test response for API validation",
        "question_type_assessment": "Test assessment"
    }
    """
    
    # Test each API
    if apis.gemini_api_key:
        print("\nTesting Gemini...")
        result = apis.call_gemini_2_5_pro(test_prompt)
        print(f"Result: {result.get('winner', 'Failed')}")
    
    if apis.grok_api_key:
        print("\nTesting Grok...")
        result = apis.call_grok_3(test_prompt)
        print(f"Result: {result.get('winner', 'Failed')}")
    
    if apis.qwen_api_key:
        print(f"\nTesting Qwen ({apis.qwen_api_provider})...")
        result = apis.call_qwen_2_5_72b(test_prompt)
        print(f"Result: {result.get('winner', 'Failed')}")

if __name__ == "__main__":
    test_apis()