# ABAP RAG Evaluation - Quick Start

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
