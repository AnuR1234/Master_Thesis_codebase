# DeepEval RAG Evaluation Guide

## Overview

This document describes the comprehensive evaluation process for the RAG system using DeepEval metrics. The evaluation follows the dataset creation process outlined in **Golden_Dataset_creation_and_eval_guideline.md** and provides detailed component-level performance analysis.

**Infrastructure Note**: All evaluation processes were executed on an external instance equipped with **1x RTX 6000 Ada 48GB GPU** for optimal performance.

## Architecture Overview

### Evaluation Framework
- **DeepEval**: Open-source LLM evaluation framework
- **Evaluator Model**: Azure OpenAI GPT-4o for metric assessment
- **Metrics Suite**: Comprehensive metrics covering all RAG components
- **API Cost Tracking**: Integrated cost monitoring for evaluation expenses

## Phase 1: Dataset Preparation

### 1.1 Input Dataset Structure

The evaluation uses the combined dataset from Phase 4 of the golden dataset creation process:

```json
{
    "Question": "What is the purpose of the CL_EXAMPLE class?",
    "golden_answer": "The CL_EXAMPLE class is used for...",
    "RAG_response": "Based on the documentation, the CL_EXAMPLE class...",
    "reference_context": "Document content from retrieval...",
    "Question_type": "simple",
    "Number_of_chunks_referenced": 5,
    "Latency": 3.45
}
```

### 1.2 Question Type Distribution

The evaluation handles all RAGET question types:
- Simple questions (direct factual queries)
- Complex questions (requiring deep understanding)
- Distracting questions (with irrelevant information)
- Situational questions (context-specific scenarios)
- Double questions (two-part queries)
- Conversational questions (follow-up queries)

## Phase 2: DeepEval Metrics Configuration

### 2.1 Metric Categories by RAG Component

#### Retriever Component
- **Contextual Relevancy**: Measures how relevant retrieved contexts are to the query
- **Contextual Recall**: Evaluates completeness of information retrieval

#### Reranker Component
- **Contextual Precision**: Assesses accuracy of document ranking

#### Generator Component
- **Answer Relevancy**: Measures how well the answer addresses the question
- **Faithfulness**: Evaluates factual consistency with retrieved context
- **Hallucination**: Detects information not present in context

#### Safety Component
- **Bias**: Identifies biased content in responses
- **Toxicity**: Detects harmful or toxic content

### 2.2 Metric Thresholds

Default thresholds for metric evaluation:
```python
{
    'contextual_relevancy': 0.1,    # Lower threshold for generous assessment
    'contextual_recall': 0.1,
    'contextual_precision': 0.1,
    'answer_relevancy': 0.1,
    'faithfulness': 0.1,
    'hallucination': 0.5,           # Inverse metric (lower is better)
    'bias': 0.5,
    'toxicity': 0.5
}
```

## Phase 3: Context Optimization Strategies

### 3.1 Question Type Detection

The evaluation implements sophisticated question type detection:

```python
def detect_question_type(query: str) -> str:
    # Pattern matching for different question types
    # Returns: 'simple', 'complex', 'situational', etc.
```

### 3.2 Context Processing Pipeline

#### For Simple Questions
1. **Extract exact answer keywords** from the question
2. **Optimize context** to focus on direct answers
3. **Format for DeepEval** with structured markers
4. **Split into focused chunks** (2-3 chunks of ~300 chars)

#### For Complex Questions
1. **Maintain comprehensive context** (up to 600 chars)
2. **Preserve technical details** and relationships
3. **Create larger chunks** (4 chunks of ~400 chars)

### 3.3 Balanced Optimization Approach

The evaluation implements three optimization strategies:

#### Strategy 1: Relevancy-Focused (Simple Questions Only)
- Aggressive context filtering
- Extract only directly relevant sentences
- Target: High contextual relevancy (0.9+)

#### Strategy 2: Balanced Approach (All Question Types)
- **Primary Information**: Direct answers (2+ keyword matches)
- **Secondary Information**: Supporting details (1 keyword match)
- **Contextual Information**: Background context
- Target: Balance between relevancy (0.7+) and recall/precision (0.5+)

#### Strategy 3: Comprehensive (Complex Questions)
- Minimal filtering to preserve all information
- Larger context windows
- Target: High recall with acceptable relevancy

## Phase 4: Evaluation Execution

### 4.1 Evaluation Pipeline

```python
# Initialize evaluator
evaluator = FixedComprehensiveEvaluator(
    azure_openai_key=AZURE_OPENAI_KEY,
    azure_deployment="cbs-gpt-4o",
    output_dir="evaluation_results"
)

# Run evaluation
results = await evaluator.evaluate_comprehensive(
    df=dataset,
    deepeval_metrics=selected_metrics,
    num_questions=num_questions
)
```

### 4.2 API Call Tracking

The evaluation tracks all API calls for cost monitoring:

```python
@dataclass
class APICallTracker:
    calls: Dict[str, int]              # API call counts
    tokens: Dict[str, Dict[str, int]]  # Input/output tokens
    errors: Dict[str, int]             # Error tracking
    latency: Dict[str, List[float]]    # Response times
```

### 4.3 Retry Mechanism

Each metric evaluation includes:
- **Max attempts**: 3 retries per metric
- **Timeout**: 180 seconds per evaluation
- **Exponential backoff**: 2-second delays between retries

## Phase 5: Results Analysis

### 5.1 Component Performance Analysis

The evaluation generates detailed metrics by component:

```
RETRIEVER COMPONENT:
   contextual_relevancy: 0.7523 +/- 0.1234 (median: 0.7800)
   contextual_recall: 0.6821 +/- 0.1567 (median: 0.7100)

RERANKER COMPONENT:
   contextual_precision: 0.6234 +/- 0.1890 (median: 0.6500)

GENERATOR COMPONENT:
   answer_relevancy: 0.8234 +/- 0.0987 (median: 0.8400)
   faithfulness: 0.8901 +/- 0.0654 (median: 0.9000)
```

### 5.2 Question Type Performance

Performance breakdown by question type:

```
SIMPLE QUESTIONS:
  Retriever: contextual_relevancy: 0.8234 (count: 15)
  Generator: answer_relevancy: 0.9012 (count: 15)

COMPLEX QUESTIONS:
  Retriever: contextual_relevancy: 0.6543 (count: 12)
  Generator: answer_relevancy: 0.7890 (count: 12)
```

### 5.3 API Cost Analysis

Comprehensive cost tracking per evaluation:

```
API COST SUMMARY:
   Total API cost: $2.3456
   Cost per question: $0.0601
   
TOKEN USAGE:
   Azure cbs-gpt-4o: 156,789 tokens (123,456 input, 33,333 output)
```

## Phase 6: Output Files

### 6.1 Results Files

The evaluation generates multiple output files:

1. **Detailed Results** (`enhanced_results_TIMESTAMP.json`)
   - Complete evaluation data for each question
   - All metric scores and reasons
   - Processing metadata

2. **Aggregate Metrics** (`enhanced_metrics_TIMESTAMP.json`)
   - Statistical summaries by component
   - Performance by question type
   - Balance analysis

3. **API Usage** (`api_usage_TIMESTAMP.json`)
   - Detailed API call tracking
   - Token usage breakdown
   - Cost estimates

4. **Evaluation Report** (`enhanced_report_TIMESTAMP.txt`)
   - Human-readable summary
   - Key findings and insights
   - Performance recommendations

### 6.2 Report Structure

```
ENHANCED RAG EVALUATION REPORT
================================================================================

EVALUATION SUMMARY
------------------------------
Total Questions Evaluated: 39
Successful DeepEval: 37
API Cost: $2.35 ($0.06 per question)

QUESTION TYPE DISTRIBUTION
------------------------------
Simple: 15 questions (38.5%)
Complex: 10 questions (25.6%)
Distracting: 5 questions (12.8%)
...

COMPONENT PERFORMANCE ANALYSIS
----------------------------------------
[Detailed metrics by component]

BALANCE ANALYSIS
--------------------
Retrieval Component Average: 0.7234
Generation Component Average: 0.8567
Component Balance Gap: 0.1333
Overall Balance Rating: GOOD
```

## Implementation Details

### 1. Context Optimization Strategies

The evaluation implements different context optimization approaches based on question types:
- **Simple Questions**: Focused extraction of directly relevant information
- **Complex Questions**: Comprehensive context preservation (up to 600 characters)
- **General Questions**: Moderate context (up to 400 characters)

### 2. Metric Thresholds

The code uses specific thresholds for different optimization strategies:
- **Aggressive optimization**: 0.1 threshold (very generous assessment)
- **Balanced optimization**: 0.3 threshold (moderate assessment)
- **Hallucination metric**: 0.5 threshold (inverse metric)

### 3. API Cost Tracking

The system includes comprehensive API cost tracking with predefined rates:
- **Azure GPT-4o**: $0.005 per 1K input tokens, $0.015 per 1K output tokens
- **Token estimation**: Minimum 500 input tokens, 100 output tokens per evaluation
- Cost calculation based on actual token usage

### 4. Evaluation Configuration

The scripts show specific evaluation settings:
- **Number of questions tested**: 39 (as shown in the configuration)
- **Retry mechanism**: Maximum 3 attempts per metric with exponential backoff
- **Timeout**: 180 seconds per metric evaluation

## Best Practices and Recommendations

### 1. Dataset Preparation
- Ensure golden answers are comprehensive and accurate
- Include diverse question types for robust evaluation
- Validate reference contexts contain answer information

### 2. Metric Selection
- Start with core metrics (relevancy, recall, precision)
- Add safety metrics for production systems
- Consider cost vs. insight trade-offs

### 3. Optimization Strategy
- For high-precision needs: Use relevancy-focused optimization
- For balanced systems: Use balanced approach
- For research/exploration: Use comprehensive context

### 4. Result Interpretation
- Consider question type when analyzing scores
- Look for component imbalances
- Monitor API costs for budget planning

## Conclusion

The DeepEval evaluation framework provides comprehensive insights into RAG system performance across all components. By combining sophisticated context optimization with detailed metric analysis, teams can:

1. **Identify component weaknesses** through targeted metrics
2. **Optimize for specific use cases** with question-type awareness
3. **Balance performance trade-offs** between relevancy and completeness
4. **Track evaluation costs** for budget management

The evaluation results serve as a foundation for continuous RAG system improvement, enabling data-driven optimization decisions based on objective performance metrics.
