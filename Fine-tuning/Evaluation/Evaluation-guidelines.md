# ABAP Documentation Evaluation Framework

This is the evaluation framework I built for my master's thesis to compare how well different AI models can generate ABAP documentation.

## What This Does

I'm comparing three models to see which one writes better ABAP documentation:
- **Fine-tuned Llama 2.5 8B** (my custom trained model)
- **Base Llama 2.5 8B** (vanilla model)
- **Gemini 2.0 Flash** (Google's model)

All models get the same 8K token limit to keep things fair.

## How It Works

The framework runs through 6 phases:
1. Load and validate the ABAP dataset
2. Apply 8K context truncation to all samples
3. Generate documentation with each model
4. Calculate automated metrics (BLEU, ROUGE, etc.)
5. Get LLM-based quality scores using Grok
6. Create comparison reports

## Key Files

- `main_evaluator.py` - Runs the whole evaluation pipeline
- `model_inference.py` - Handles documentation generation for all 3 models
- `automated_metrics.py` - Calculates traditional NLP metrics
- `enhanced_abap_metrics.py` - My custom ABAP-specific metrics
- `pdsqi_abap_evaluator.py` - 9-dimensional quality assessment
- `llm_evaluator.py` - LLM-based human-like evaluation

## What Gets Evaluated

**Traditional Metrics:** BLEU, ROUGE, BERTScore, semantic similarity

**ABAP-Specific Stuff:** How accurate are the method names? Does it hallucinate non-existent code? Does it follow the template structure?

**Quality Assessment (PDSQI-9):** 9 different aspects like accuracy, completeness, clarity, usefulness, etc. on a 1-5 scale

## Setup

Install the required packages:
```bash
pip install torch transformers peft sentence-transformers bert-score rouge-score
pip install scikit-learn nltk pandas numpy google-generativeai openai
```

Create a config file with your model paths and API keys, then run:
```bash
python main_evaluator.py config.json
```

## Templates

The models generate structured documentation using templates I created for different ABAP file types:
- Classes (methods, dependencies, database tables)
- Test classes (test methods, assertions, mocks)
- Reports (program flow, selection screen)
- Function modules (interface, parameters)

## Results

Everything gets saved to an `evaluation_results/` folder with:
- Generated documentation from each model
- Calculated metrics and comparisons
- LLM evaluation scores
- Final comparison report

## Notes

This was built for my thesis research on whether fine-tuning actually helps with domain-specific documentation. The framework tries to be as fair as possible by using the same context limits and evaluation criteria for all models.

The code has fallbacks for retry logic for API calls, so it should be pretty robust even if some dependencies aren't available.
