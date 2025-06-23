#!/usr/bin/env python3
"""
Gemini Model Checker
Check what models are available with your API key
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_available_models():
    """Check what Gemini models are available with your API key"""
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("❌ No GEMINI_API_KEY found in environment")
        return
    
    try:
        import google.generativeai as genai
        print("✅ google-generativeai library imported successfully")
    except ImportError:
        print("❌ google-generativeai not installed")
        print("Install with: pip install google-generativeai")
        return
    
    try:
        # Configure with your API key
        genai.configure(api_key=api_key)
        print(f"✅ Configured with API key: {api_key[:12]}...")
        
        # List all available models
        print(f"\n{'='*60}")
        print(f"AVAILABLE GEMINI MODELS")
        print(f"{'='*60}")
        
        models = genai.list_models()
        available_models = []
        
        for model in models:
            model_name = model.name
            supported_methods = model.supported_generation_methods
            
            print(f"\n📋 Model: {model_name}")
            print(f"   Supported methods: {supported_methods}")
            
            # Check if it supports generateContent
            if 'generateContent' in supported_methods:
                # Extract just the model name part
                simple_name = model_name.replace('models/', '')
                available_models.append(simple_name)
                print(f"   ✅ Supports generateContent - Can be used!")
                print(f"   💡 Use this name in code: '{simple_name}'")
            else:
                print(f"   ❌ Does not support generateContent")
        
        print(f"\n{'='*60}")
        print(f"SUMMARY")
        print(f"{'='*60}")
        print(f"Models that support generateContent ({len(available_models)} found):")
        
        for model in available_models:
            print(f"  - {model}")
        
        # Test a model
        if available_models:
            print(f"\n{'='*60}")
            print(f"TESTING FIRST AVAILABLE MODEL")
            print(f"{'='*60}")
            
            test_model_name = available_models[0]
            print(f"Testing model: {test_model_name}")
            
            try:
                test_model = genai.GenerativeModel(test_model_name)
                test_response = test_model.generate_content("Hello, respond with just 'Test successful!'")
                print(f"✅ Test successful!")
                print(f"   Response: {test_response.text}")
                
                print(f"\n💡 RECOMMENDED CODE UPDATE:")
                print(f"   Replace 'gemini-2.5-pro' with '{test_model_name}' in your code")
                
            except Exception as e:
                print(f"❌ Test failed: {e}")
        
        return available_models
        
    except Exception as e:
        print(f"❌ Error checking models: {e}")
        print(f"\nPossible issues:")
        print(f"1. Invalid API key")
        print(f"2. API key doesn't have access to Gemini models")
        print(f"3. Regional restrictions")
        print(f"4. Billing/quota issues")
        return []

def suggest_fallback_models():
    """Suggest common fallback model names to try"""
    
    fallback_models = [
        "gemini-2.5-flash-preview-05-20",
        "gemini-1.5-flash", 
        "gemini-1.0-pro",
        "gemini-pro",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash-latest"
    ]
    
    print(f"\n{'='*60}")
    print(f"COMMON FALLBACK MODELS TO TRY")
    print(f"{'='*60}")
    
    for model in fallback_models:
        print(f"  - {model}")
    
    print(f"\nTo test these manually, update your api_handlers.py:")
    print(f"Replace: self.gemini_model = genai.GenerativeModel('gemini-2.5-flash-preview-05-20')")
    print(f"With:    self.gemini_model = genai.GenerativeModel('MODEL_NAME_HERE')")

if __name__ == "__main__":
    print("🔍 GEMINI MODEL AVAILABILITY CHECKER")
    print("="*60)
    
    available_models = check_available_models()
    
    if not available_models:
        print(f"\n⚠️  No compatible models found!")
        suggest_fallback_models()
        
        print(f"\n🔧 TROUBLESHOOTING STEPS:")
        print(f"1. Verify your API key is correct")
        print(f"2. Check if billing is enabled on your Google Cloud/AI Studio account")
        print(f"3. Try the fallback models listed above")
        print(f"4. Check regional availability")
    else:
        print(f"\n🎉 SUCCESS! You have {len(available_models)} compatible models available.")
        print(f"Update your code to use one of the available models.")