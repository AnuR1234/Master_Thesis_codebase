# ABAP RAG Evaluation System - Complete Experiment Guide

## 📋 Overview

This experiment implements an LLM-as-Judge methodology to compare the performance of open-source vs closed-source RAG (Retrieval-Augmented Generation) systems for ABAP (SAP's programming language) technical questions. The system uses multiple state-of-the-art language models as impartial judges to evaluate response quality.

### Key Features

* Blind evaluation with position switching to eliminate bias
* Multiple judge models (Gemini, Grok-3, Qwen) for consensus
* Comprehensive scoring across 5 technical dimensions
* Statistical analysis with detailed reporting and visualizations
* Question-type stratification for nuanced performance insights

## 📀 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Two JSON Files                     │
│  ┌─────────────────────┐    ┌─────────────────────────┐    │
│  │ Open-Source RAG     │    │ Closed-Source RAG       │    │
│  │ Responses           │    │ Responses               │    │
│  └─────────────────────┘    └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Dataset Analyzer                           │
│  • Validates matching questions                              │
│  • Analyzes question type distribution                       │
│  • Recommends balancing strategies                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Main Evaluator                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ For each question:                                   │   │
│  │   1. Randomly assign responses to positions A/B     │   │
│  │   2. Send to each judge model                       │   │
│  │   3. Repeat with position switching (3x default)    │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Judge Models                              │
│  ┌──────────┐    ┌──────────┐    ┌────────────────────┐    │
│  │ Gemini   │    │ Grok-3   │    │ Qwen-2.5-Coder-32B │    │
│  │ 1.5 Pro  │    │          │    │                    │    │
│  └──────────┘    └──────────┘    └────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Results & Analysis                        │
│  • JSON: Complete evaluation data                           │
│  • CSV: Statistical analysis export                         │
│  • Markdown: Human-readable report                          │
│  • Plots: Performance visualizations                        │
└─────────────────────────────────────────────────────────────┘
```

## ⚙️ Setup Instructions

### 1. Environment Setup

```bash
# Clone or download the experiment files
cd abap-rag-evaluation
python3 setup.py
```

Creates:

* `results/`, `data/`, `.env`, `QUICK_START.md`

### 2. Install Dependencies

```bash
pip install pandas requests python-dotenv google-generativeai matplotlib seaborn
```

### 3. Configure API Keys

Edit `.env`:

```bash
GEMINI_API_KEY=your-gemini-key
GROK_API_KEY=your-grok-key
QWEN_API_KEY=your-together-key
```

### 4. Verify Gemini Models

```bash
python3 gemini_checker.py
```

## 📊 Dataset Format

Each JSON must include:

* `Question`, `reference_context`, `Question_type`
* Responses (`RAG_response_1`, `RAG_response_2`, `response`, `answer`, etc.)

**Example**:

```json
{
  "Question": "What method converts XML to XSTRING?",
  "reference_context": "The class defines XML_TO_XSTRING method...",
  "RAG_response_1": "The method XML_TO_XSTRING converts...",
  "Question_type": "Simple"
}
```

## 🚀 Running the Experiment

### Step 1: Analyze Your Dataset

```bash
python3 dataset_analyzer.py path/to/open_source.json path/to/closed_source.json
```

### Step 2: Test with Sample Data

```bash
python3 run_evaluation.py data/sample_open_source.json data/sample_closed_source.json --simulation
```

### Step 3: Run Full Evaluation

```bash
python3 run_evaluation.py open.json closed.json --balance 3 --enhanced --plots
```

### Options

* `--balance N`: Balance dataset
* `--switches N`: Position switches
* `--enhanced`: Enhanced summary
* `--plots`: Visualizations
* `--simulation`: No API calls
* `--output-dir`: Custom output path

## 📊 Evaluation Process

1. **Blind Evaluation**: Judges see random A/B with no source info
2. **Position Switching**: Repeat 3x per judge
3. **Scoring Dimensions**:

   * Technical Accuracy
   * SAP Domain Knowledge
   * Completeness
   * Clarity
   * Practical Value

**Prompt Template**:

```
[EVALUATION TASK] ...
[QUESTION CLASSIFICATION] {Simple|Complex|...}
[REFERENCE CONTEXT] ...
[QUESTION] ...
[RESPONSE A] ...
[RESPONSE B] ...
[EVALUATION CRITERIA]
...
[OUTPUT FORMAT] {
  "winner": "A" | "B" | "Tie",
  "confidence": float,
  "scores": {...},
  "reasoning": "...",
  "question_type_assessment": "..."
}
```

## 🔍 Understanding the Results

**Outputs**:

* JSON: `abap_rag_evaluation_TIMESTAMP.json`
* CSV: `evaluation_data_TIMESTAMP.csv`
* Markdown: `enhanced_summary_TIMESTAMP.md`
* Plots: pie/bar charts in `plots/`

### Interpreting Metrics

* Win Rates, Tie Rates
* Confidence Levels:

  * ⨝ 0.8: High
  * 0.6-0.8: Moderate
  * <0.6: Low
* Dimensions:

  * Technical Accuracy
  * SAP Knowledge
  * Completeness
  * Clarity
  * Practical Use

## 🔎 Advanced Usage

* Add judges in `main_evaluator.py`
* Modify `api_handlers.py` to add custom models
* Add new `Question_type` labels as needed

## 🛡️ Troubleshooting

* API errors: Check `.env` and permissions
* JSON errors: Check file formats and fields
* Low confidence: Review ambiguous questions or context quality

## 📘️ Best Practices

* Balance dataset for fairness
* Use all judges for reliability
* Include plots and enhanced report in thesis

## 🌟 Expected Outcomes

* Quantify open vs closed source performance
* Stratified performance by question type
* Validated, reproducible evaluation method

## ✅ Quick Start Checklist

* [x] `python3 setup.py`
* [x] Install deps
* [x] Add API keys
* [x] Test with simulation mode
* [x] Analyze real dataset
* [x] Run full evaluation
* [x] Review `results/`

## 📚 Further Information

* **Methodology**: LLM-as-Judge, position switching
* **Rigor**: Thesis-level reproducibility
* **Extensible**: Add new judges, criteria, metrics
