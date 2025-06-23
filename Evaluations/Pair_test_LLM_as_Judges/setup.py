#!/usr/bin/env python3
"""
Quick Setup Script - Skips package installation for existing environments
"""

import os
import sys
import json
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ["results", "data", "logs"]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created: {directory}/")
    
    return True

def create_env_file():
    """Create .env file with API key templates"""
    print("\n🔐 Creating .env file...")
    
    env_content = """# ABAP RAG Evaluation System - API Keys
# Add your actual API keys below

# Gemini API Key (from Google AI Studio: https://makersuite.google.com/app/apikey)
GEMINI_API_KEY=your-gemini-api-key-here

# Grok API Key (from xAI Console: https://console.x.ai/)
GROK_API_KEY=your-grok-api-key-here

# Qwen API Key (Together AI: https://api.together.xyz/)
QWEN_API_KEY=your-together-ai-key-here

# Evaluation settings
DEFAULT_POSITION_SWITCHES=3
DEFAULT_QUESTIONS_PER_TYPE=3
OUTPUT_DIRECTORY=results
"""
    
    with open(".env", 'w') as f:
        f.write(env_content)
    
    print("✅ Created: .env")
    print("   ⚠️  IMPORTANT: Add your actual API keys to this file!")
    
    return True

def create_sample_data():
    """Create sample dataset files for testing"""
    print("\n📄 Creating sample dataset files...")
    
    sample_open_source = [
        {
            "Question": "What method in the class is responsible for converting a table of XML text lines to an XSTRING value?",
            "reference_context": "The class defines the following methods:\n- `XML_TO_XSTRING`: Converts a table of XML text lines to an XSTRING value.\n- `XSTRING_TO_XML`: Converts an XSTRING value to a table of XML text lines.",
            "RAG_response_1": "The method responsible for converting XML text lines to XSTRING is XML_TO_XSTRING. It concatenates the lines and uses SCMS_STRING_TO_XSTRING function module for conversion.",
            "Question_type": "Simple"
        },
        {
            "Question": "How does the ABAP exception handling work in this class?",
            "reference_context": "The class uses structured exception handling with TRY-CATCH blocks and raises custom exceptions of type /ltb/cx_bas_downup with specific text IDs.",
            "RAG_response_1": "The class uses TRY-CATCH blocks for exception handling and raises custom exceptions of type /ltb/cx_bas_downup.",
            "Question_type": "Complex"
        },
        {
            "Question": "What is the purpose of SCMS_STRING_TO_XSTRING function module?",
            "reference_context": "SCMS_STRING_TO_XSTRING is a standard ABAP function module used to convert string data to XSTRING format for binary processing and file operations.",
            "RAG_response_1": "SCMS_STRING_TO_XSTRING converts string to XSTRING format for binary data processing.",
            "Question_type": "Intermediate"
        }
    ]
    
    sample_closed_source = [
        {
            "Question": "What method in the class is responsible for converting a table of XML text lines to an XSTRING value?",
            "reference_context": "The class defines the following methods:\n- `XML_TO_XSTRING`: Converts a table of XML text lines to an XSTRING value.\n- `XSTRING_TO_XML`: Converts an XSTRING value to a table of XML text lines.",
            "RAG_response_2": "The method responsible for converting a table of XML text lines to an XSTRING value is `XML_TO_XSTRING`. This method takes a table of XML text lines as input parameter `it_xml_tab` and returns an XSTRING value. The method concatenates all lines with CR/LF characters using `cl_abap_char_utilities=>cr_lf` and calls the ABAP function module `SCMS_STRING_TO_XSTRING` for conversion. It includes proper exception handling with `/ltb/cx_bas_downup` for error cases.",
            "Question_type": "Simple"
        },
        {
            "Question": "How does the ABAP exception handling work in this class?",
            "reference_context": "The class uses structured exception handling with TRY-CATCH blocks and raises custom exceptions of type /ltb/cx_bas_downup with specific text IDs.",
            "RAG_response_2": "The class implements comprehensive ABAP exception handling using structured TRY-CATCH-CLEANUP blocks. It raises custom exceptions of type `/ltb/cx_bas_downup` with specific text IDs for different error scenarios. The exception handling includes checking `sy-subrc` after function module calls and proper error propagation up the call stack. Each exception includes contextual error messages for debugging and user feedback.",
            "Question_type": "Complex"
        },
        {
            "Question": "What is the purpose of SCMS_STRING_TO_XSTRING function module?",
            "reference_context": "SCMS_STRING_TO_XSTRING is a standard ABAP function module used to convert string data to XSTRING format for binary processing and file operations.",
            "RAG_response_2": "The ABAP function module SCMS_STRING_TO_XSTRING is a standard SAP utility that converts string data to XSTRING (hexadecimal string) format. This conversion is essential for binary data processing, file operations, and data transmission. The function handles character encoding and ensures proper binary representation suitable for download operations and external system integration.",
            "Question_type": "Intermediate"
        }
    ]
    
    # Save sample files
    with open("data/sample_open_source.json", 'w') as f:
        json.dump(sample_open_source, f, indent=2)
    
    with open("data/sample_closed_source.json", 'w') as f:
        json.dump(sample_closed_source, f, indent=2)
    
    print("✅ Created: data/sample_open_source.json")
    print("✅ Created: data/sample_closed_source.json")
    print("   Use these files to test the system")
    
    return True

def create_quick_readme():
    """Create a quick start README"""
    print("\n📖 Creating Quick Start guide...")
    
    readme_content = """# ABAP RAG Evaluation - Quick Start

## 🚀 You're All Set! 

Your environment is ready. Here's what to do next:

### 1. Add API Keys
Edit the `.env` file and add your actual API keys:
```
GEMINI_API_KEY=your-actual-key-here
GROK_API_KEY=your-actual-key-here  
QWEN_API_KEY=your-actual-key-here
```

### 2. Test the System
```bash
# Test with sample data (no API keys needed)
python3 run_evaluation.py data/sample_open_source.json data/sample_closed_source.json --simulation
```

### 3. Analyze Your Dataset
```bash
# Check your dataset structure  
python3 dataset_analyzer.py your_open_source.json your_closed_source.json
```

### 4. Run Real Evaluation
```bash
# With balanced dataset
python3 run_evaluation.py your_open_source.json your_closed_source.json --balance 3
```

## 📁 Your Dataset Format

Two JSON files with matching questions:

**open_source_responses.json:**
```json
[
  {
    "Question": "What method converts XML to XSTRING?",
    "reference_context": "Documentation...", 
    "RAG_response_1": "Response from open-source RAG...",
    "Question_type": "Simple"
  }
]
```

**closed_source_responses.json:**
```json
[
  {
    "Question": "What method converts XML to XSTRING?",
    "reference_context": "Documentation...",
    "RAG_response_2": "Response from closed-source RAG...", 
    "Question_type": "Simple"
  }
]
```

## 🔑 Getting API Keys

- **Gemini**: https://makersuite.google.com/app/apikey
- **Grok**: https://console.x.ai/
- **Qwen**: https://api.together.xyz/

## 📊 Results

Check the `results/` folder for:
- Detailed JSON results
- Summary report for thesis
- CSV data for statistical analysis

## 🆘 Help

Run with `--help` for options:
```bash
python3 run_evaluation.py --help
```
"""
    
    with open("QUICK_START.md", 'w') as f:
        f.write(readme_content)
    
    print("✅ Created: QUICK_START.md")
    
    return True

def test_imports():
    """Test if required packages are available"""
    print("\n🧪 Testing package imports...")
    
    try:
        import json
        print("✅ json - OK")
        
        import pandas as pd
        print("✅ pandas - OK")
        
        import requests
        print("✅ requests - OK")
        
        # Optional packages
        try:
            import google.generativeai as genai
            print("✅ google-generativeai - OK")
        except ImportError:
            print("⚠️  google-generativeai - Not installed (install: pip install google-generativeai)")
        
        print("✅ Core packages available!")
        return True
        
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("   Install with: pip install pandas requests")
        return False

def main():
    """Main setup function for existing environments"""
    print("🚀 ABAP RAG Evaluation System - Quick Setup")
    print("=" * 50)
    print("(Skipping package installation - using existing environment)")
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Testing package imports", test_imports),
        ("Creating directories", create_directories),
        ("Creating .env file", create_env_file),
        ("Creating sample data", create_sample_data),
        ("Creating Quick Start guide", create_quick_readme)
    ]
    
    for step_name, step_func in steps:
        print(f"\n{step_name}...")
        if not step_func():
            print(f"❌ Setup failed at: {step_name}")
            return 1
    
    print(f"\n{'='*50}")
    print("🎉 QUICK SETUP COMPLETE!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Edit .env file and add your API keys")
    print("2. Test: python3 run_evaluation.py data/sample_open_source.json data/sample_closed_source.json --simulation")
    print("3. Run real evaluation with your data")
    print("\nSee QUICK_START.md for detailed instructions!")
    
    return 0

if __name__ == "__main__":
    exit(main())