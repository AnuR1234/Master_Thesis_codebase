# ABAP Documentation Evaluation Framework

This repository contains the evaluation framework for the language models I used in my master's thesis using ABAP documentation generation. The evaluation will ultimately compare a fine-tuned Meta-Llama-3.1-8B-Instruct language model against a base Meta-Llama-3.1-8B-Instruct and Gemini 2.0 Flash language models while keeping the context length constrained to a fair 8K.

## 🤗 Pre-trained Model

The fine-tuned weights are available on Hugging Face:

**[Anu123/llama3-8b-lora-finetune](https://huggingface.co/Anu123/llama3-8b-lora-finetune)**

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
