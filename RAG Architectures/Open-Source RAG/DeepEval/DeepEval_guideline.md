# RAG Evaluation with DeepEval - Quick Guide

## Overview
Evaluate RAG system performance using DeepEval metrics to assess retrieval quality, generation accuracy, and safety. This guide assumes you have already created the golden dataset following the process detailed in **Golden_Dataset_creation_and_evaluation_guidelines.md**.

## DeepEval Metrics

### By Component
- **Retriever**: Contextual Relevancy, Contextual Recall
- **Reranker**: Contextual Precision  
- **Generator**: Answer Relevancy, Faithfulness, Hallucination
- **Safety**: Bias, Toxicity

### Score Interpretation
- **0.9+**: Excellent
- **0.7-0.9**: Good
- **0.5-0.7**: Fair
- **<0.5**: Poor

## Running the Evaluation

### Setup
```bash
# Install dependencies
pip install deepeval langchain-openai pandas numpy python-dotenv tqdm

# Create .env file
AZURE_OPENAI_KEY=your-key
AZURE_API_VERSION=2024-08-01-preview
AZURE_DEPLOYMENT=your-deployment
AZURE_ENDPOINT=https://your-endpoint.openai.azure.com/
```

### Required Dataset Format
The evaluation requires a merged dataset containing golden questions and RAG responses. The dataset creation process is detailed in **Golden_Dataset_creation_and_evaluation_guidelines.md**. The final merged dataset should have this structure:
```json
{
  "question": "What is the purpose of the method?",
  "golden_answer": "The expected answer...",
  "rag_response": "The RAG system's response...",
  "reference_context": "The retrieved context..."
}
```

### Run Scripts

**Full Evaluation**:
```bash
python comprehensive_thesis_eval_final_new_metrics_with_API_costs.py
```

**Simple Questions Only**:
```bash
python eval_setup_for_simple_qns_only_v2.py
```

### Configuration
```python
# In script - modify these:
DATASET_PATH = "/path/to/merged_classes.json"
num_questions = 39  # or None for all
selected_metrics = ['contextual_relevancy', 'contextual_recall', ...]
```

## Output Files

```
results/
├── enhanced_results_*.json     # Detailed scores
├── enhanced_metrics_*.json     # Aggregated metrics
├── api_usage_*.json           # Cost tracking
└── enhanced_report_*.txt      # Summary report
```

## Key Features

### Context Optimization
- **Simple Questions**: Extracts 2-3 focused chunks with key information
- **Complex Questions**: Maintains comprehensive context (600-800 chars)

### API Cost Tracking
- Tracks tokens used per model
- Calculates costs: $5/1M input tokens, $15/1M output tokens
- Reports total cost and cost per question

### Error Handling
- 3 retry attempts per metric
- 180-second timeout
- Graceful failure logging

## Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| API Timeout | Increase timeout to 300s |
| Low Scores | Check context optimization, verify dataset quality |
| High Costs | Reduce num_questions, use simple questions script |

## Results Summary
The evaluation provides:
- Individual metric scores per question
- Component-wise performance analysis
- Question type breakdown
- API usage and cost summary
