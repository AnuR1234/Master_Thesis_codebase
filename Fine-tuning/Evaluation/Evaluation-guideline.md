# ABAP Documentation Evaluation Framework

This repository contains the evaluation framework for the language models I used in my master's thesis using ABAP documentation generation. The evaluation will ultimately compare a fine-tuned Meta-Llama-3.1-8B-Instruct language model against a base Meta-Llama-3.1-8B-Instruct and Gemini 2.0 Flash language models while keeping the context length constrained to a fair 8K.

## Overview

The framework supports a full evaluation pipeline about how well the language models can be employed to generate technical documentation for SAP ABAP code. The evaluation addresses a growing need for automated documentation generation and usage in enterprise software development, specifically in legacy code bases like SAP ABAP.

## Important Scripts

### Primary Orchestrators
- **`main_eval_test.py`** - Primary evaluation pipeline
- **`main_evaluator.py`** - Alternative orchestrator

### Core Scripts
- **`data_preprocessor.py`** - loads in dataset and handles 8K context truncation
- **`model_inference.py`** - generates documentation across 3 different language models
- **`automated_metrics.py`** - creates traditional NLP metrics including BLEU, ROUGE, and BERTScore
- **`enhanced_abap_metrics.py`** - ABAP-specific factual accuracy metrics
- **`pdsqi_abap_evaluator.py`** - PDSQI-9 quality assessment
- **`llm_evaluator.py`** - LLM based evaluation using Grok

## How to Run the Evaluation

### Prerequisites

1. **Install Required Libraries**
   ```bash
   pip install torch transformers peft google-generativeai openai
   pip install rouge-score bert-score sentence-transformers nltk scikit-learn
   pip install numpy pandas
   ```

2. **Download NLTK Data**
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('punkt_tab')
   ```
## 🤗 Pre-trained Model

The fine-tuned weights are available on Hugging Face:

**[Anu123/llama3-8b-lora-finetune](https://huggingface.co/Anu123/llama3-8b-lora-finetune)**

3. **Prepare Your Models**
   - Download base Meta-Llama-3.1-8B-Instruct **([https://huggingface.co/Anu123/llama3-8b-lora-finetune](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct))**
   - Ensure your fine-tuned LoRA adapter is available
   - Get API keys for Gemini and Grok

### Configuration Setup

1. **Create a Configuration File** (`config.json`):
   ```json
   {
     "dataset": {
       "path": "path/to/your/evaluation_dataset.json"
     },
     "output_dir": "./evaluation_results",
     "max_context_tokens": 8000,
     "models": {
       "base_llama": {
         "enabled": true,
         "model_path": "path/to/llama-2.5-8b"
       },
       "fine_tuned_llama": {
         "enabled": true,
         "adapter_path": "path/to/lora/adapter"
       },
       "gemini": {
         "enabled": true,
         "api_key": "your-gemini-api-key",
         "model_name": "gemini-2.0-flash"
       }
     },
     "llm_evaluator": {
       "provider": "xai",
       "api_key": "your-grok-api-key",
       "model": "grok-beta"
     }
   }
   ```

2. **Prepare Your Dataset** (`evaluation_dataset.json`):
   ```json
   [
     {
       "id": "sample_001",
       "abap_code": "CLASS zcl_example DEFINITION...",
       "ground_truth_documentation": "## Overview\nThis class...",
       "file_type": "class"
     }
   ]
   ```

### Running the Complete Evaluation

1. **Full Pipeline Execution**
   ```bash
   python main_eval_test.py config.json
   ```

   This runs all 6 phases automatically:
   - Phase 1: Dataset loading and validation
   - Phase 2: 8K context truncation preprocessing
   - Phase 3: Documentation generation with all models
   - Phase 4: Automated metrics calculation
   - Phase 5: LLM-based evaluation (using Grok)
   - Phase 6: Final report generation

2. **Run Specific Phases** (if needed)
   ```bash
   # Run only preprocessing
   python main_eval_test.py config.json --phase 2
   
   # Run only inference
   python main_eval_test.py config.json --phase 3
   ```

3. **Quick Test Mode**
   For testing with a small dataset (≤10 samples), the framework automatically limits LLM evaluations to 3 samples per model to save costs.

### Output Structure

After running, you'll find results in the `output_dir` specified in your config:

```
evaluation_results/
├── logs/
│   └── evaluation_YYYYMMDD_HHMMSS.log
├── preprocessed_data/
│   └── dataset_8k.json
├── generated_docs/
│   ├── base_llama_results.json
│   ├── fine_tuned_llama_results.json
│   └── gemini_results.json
├── metrics/
│   ├── base_llama_metrics.json
│   ├── fine_tuned_llama_metrics.json
│   ├── gemini_metrics.json
│   └── comparison.csv
├── evaluations/
│   ├── base_llama_llm_eval.json
│   ├── fine_tuned_llama_llm_eval.json
│   └── gemini_llm_eval.json
└── reports/
    └── evaluation_report.md
```

### Monitoring Progress

The framework provides detailed logging:
- Real-time console output showing progress
- Detailed log files in the `logs/` directory
- Progress indicators every 5 samples during inference
- Automatic retry logic for API failures

### Customization Options

1. **Adjust Token Limits**
   Modify `max_context_tokens` in config.json (default: 8000)

2. **Control LLM Evaluation Sample Size**
   Edit line in `main_eval_test.py`:
   ```python
   limited_results = model_results[:20]  # Change 20 to your desired number
   ```

3. **Add Custom Metrics**
   Extend `automated_metrics.py` or `enhanced_abap_metrics.py`

4. **Modify Templates**
   Edit documentation templates in `model_inference.py`

### Troubleshooting

1. **Out of Memory Errors**
   - Reduce batch size in model generation
   - Use CPU offloading for large models
   - Process fewer samples at once

2. **API Rate Limits**
   - Increase retry delays in `model_inference.py` and `llm_evaluator.py`
   - Reduce concurrent API calls

3. **Missing Dependencies**
   - Framework falls back to mock implementations if libraries are missing
   - Check logs for "Mock implementations" warnings

## Evaluation Strategy

### Multi-dimensional Evaluation
The framework uses three separate evaluations considered complimentary:

- **Automated Metrics** - BLEU, ROUGE, BERTScore, semantic similarity, and ABAP/technical term coverage
- **Enhanced Metrics** - Factual accuracy checks and template compliance
- **PDSQI-9 Assessment** - Nine different aspects of quality based on a review of available healthcare AI evaluation literature¹
- **LLM-As-Judge** - Guideline by professional expert evaluated utilizing Grok across the seven quality aspects

### PDSQI-9 Quality Aspects
Following the established framework in *Automating Evaluation of AI Text Generation in Healthcare*¹, the evaluation looks at:
- **Accuracy, Completeness, and Hallucinations** (reverse scored)
- **Clarity, Organization, and Usefulness**
- **Omissions** (reverse scored), **Coherence, and Conciseness**

### Fair Comparison Design
All models will be assessed against the same 8K token constraint, to ensure a fair comparison. The framework handles class, test class, report, and function module ABAP files in terms of specific documentation templates.

---

¹ *Reference: "Automating Evaluation of AI Text Generation in Healthcare with a Large Language Model (LLM)-as-a-Judge" - This work adapts the PDSQI-9 framework for technical documentation evaluation in the ABAP domain.*
