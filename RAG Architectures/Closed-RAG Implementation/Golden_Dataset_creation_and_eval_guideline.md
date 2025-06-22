# RAG Golden Dataset Creation and Evaluation Process

## Overview

This document describes the end-to-end process of creating a golden dataset for evaluating a Retrieval-Augmented Generation (RAG) system designed for SAP ABAP code documentation. The process includes dataset generation, quality filtering, and evaluation pipeline execution.

**Infrastructure Note**: All processes were executed on an external instance equipped with **1x RTX 6000 Ada 48GB GPU** to handle the computationally intensive LLM operations.

## Architecture Overview

The RAG system consists of several key components:

- **Vector Database**: Qdrant for storing and retrieving document embeddings
- **Embedding Models**: OpenAI embeddings for dense vectors, BM25 for sparse vectors
- **LLM**: Claude 3.7 Sonnet via Amazon Bedrock for response generation
- **Query Enhancement**: Automatic query rewriting and decomposition
- **Reranker**: Cross-encoder model for improving retrieval relevance

## Phase 1: Golden Dataset Generation

### 1.1 Dataset Structure

The golden dataset follows the RAGET (Retrieval-Augmented Generation Evaluation Toolkit) methodology, generating six distinct question types:

1. **Simple Questions**: Direct, straightforward queries about SAP ABAP concepts
2. **Complex Questions**: Advanced queries requiring deep understanding and paraphrasing
3. **Distracting Questions**: Queries with irrelevant information to test focus
4. **Situational Questions**: Context-specific queries with user scenarios
5. **Double Questions**: Queries asking for two distinct pieces of information
6. **Conversational Questions**: Follow-up questions building on previous context

### 1.2 Generation Process

The dataset generation script (`golden_dataset.py`) implements the following workflow:

```python
# Key parameters
questions_per_type = 15  # Number of questions for each type
collection_types = ["classes", "reports"]  # SAP ABAP documentation types
```

#### Document Pool Creation
1. **Retrieval**: Extract documents from Qdrant collections
2. **Quality Filtering**: Only documents with >100 characters of content
3. **Topic Extraction**: Identify main topics from document titles

#### Question Generation Pipeline
For each question type:
1. Select random document from the pool
2. Generate question using Claude 3.7 Sonnet with type-specific prompts
3. Generate reference answer based on the document context
4. Ensure uniqueness (>70% dissimilarity threshold)
5. Retry mechanism for failed generations (max 3 attempts)

### 1.3 Output Format

Each generated question includes:
```json
{
    "question": "What is the purpose of the CL_EXAMPLE class?",
    "reference_context": "Document content from Qdrant...",
    "reference_answer": "Generated comprehensive answer...",
    "conversation_history": [],
    "question_type": "simple"
}
```

## Phase 2: Dataset Quality Filtering

### 2.1 Critique Process

The generated dataset undergoes quality assessment using Gemini 2.0 Flash (`rag_evaluation_dataset_critic.py`) based on three criteria:

1. **Groundedness** (1-5 scale)
   - Score 5: Context provides all necessary information
   - Score 1: Context insufficient to answer the question

2. **Relevance** (1-5 scale)
   - Score 5: Clearly related to SAP ABAP domain and useful
   - Score 1: Unrelated to SAP ABAP domain

3. **Stand-Alone** (1-5 scale)
   - Score 5: Question is entirely self-contained
   - Score 1: Highly dependent on external context

### 2.2 Filtering Criteria

Questions must achieve a minimum score of 4 in all three categories to pass the critique. This ensures:
- Questions can be answered from the provided context
- Questions are relevant to SAP ABAP practitioners
- Questions are understandable without additional context

### 2.3 Critique Results

Example output from the critique process:
```
===== EVALUATION SUMMARY =====
Total questions processed: 90
Questions that passed critique: 72 (80.0%)
Questions that failed critique: 18 (20.0%)

Failure reasons:
  Failed groundedness: 8 (8.9%)
  Failed relevance: 3 (3.3%)
  Failed standalone: 7 (7.8%)
=============================
```

## Phase 3: RAG Evaluation Pipeline

### 3.1 Evaluation Setup

The evaluation script (`RAG_running_golden_DS_eval.py`) processes the filtered golden dataset through the complete RAG pipeline:

```python
# Key evaluation parameters
use_hybrid = True          # Use both dense and sparse vectors
use_reranker = True        # Apply reranking to results
top_k = 5                  # Number of context documents
enable_query_enhancement = True  # Apply query rewriting/decomposition
```

### 3.2 Processing Workflow

For each question in the golden dataset:

1. **Query Enhancement** (conditional)
   - Disabled for "simple" questions
   - Enabled for all other question types
   - Includes query rewriting and decomposition

2. **Document Retrieval**
   - Hybrid search using dense (OpenAI) and sparse (BM25) embeddings
   - Retrieval from appropriate Qdrant collection

3. **Reranking**
   - Cross-encoder model scores query-document pairs
   - Selects top-k most relevant documents

4. **Response Generation**
   - Claude 3.7 Sonnet generates answer using retrieved context
   - Confidence level assessment (HIGH/MEDIUM/LOW)

5. **Latency Measurement**
   - End-to-end processing time recorded for each query

### 3.3 Output Structure

The evaluation produces a JSON file with the following structure for each question:

```json
{
    "Question": "Original question from golden dataset",
    "expected_response": null,
    "retrieved_docs": [
        {
            "id": 1,
            "title": "Document title",
            "filename": "source_file.abap",
            "score": 0.8234,
            "text": "Document content...",
            "code_snippet": "ABAP code..."
        }
    ],
    "response": "Generated answer from Claude",
    "question_type": "simple",
    "latency": 3.45
}
```

## Phase 4: Results Analysis

### 4.1 Performance Metrics

The evaluation captures several key metrics:

- **Retrieval Quality**: Number and relevance scores of retrieved documents
- **Response Quality**: Assessed through confidence levels
- **System Performance**: Query latency measurements
- **Query Enhancement Impact**: Comparison of enhanced vs. non-enhanced queries

### 4.2 Data Combination

The `combine_golden_RAG_dataset.py` script merges:
- Original golden dataset questions
- RAG system responses
- Performance metrics

This creates a comprehensive evaluation dataset for further analysis.

## Phase 5: DeepEval Comprehensive Evaluation

The combined dataset undergoes comprehensive evaluation using DeepEval metrics to assess RAG component performance. This process is detailed in the companion document: **DeepEval_RAG_Evaluation_Guide.md**

## File Structure

```
project/
├── golden_dataset.py                    # Dataset generation
├── rag_evaluation_dataset_critic.py     # Quality filtering
├── RAG_running_golden_DS_eval.py        # Evaluation pipeline
├── combine_golden_RAG_dataset.py        # Results merger
├── pipeline.py                          # Core RAG pipeline
├── retriever.py                         # Document retrieval
├── generator.py                         # Response generation
├── query_enhancer.py                    # Query enhancement
├── config.py                            # Configuration
├── claude.env                           # Environment variables
└── stream_lit.py                        # Web interface
```

## Usage Instructions

### 1. Generate Golden Dataset
```bash
python golden_dataset.py
# Output: raget_evaluation_dataset_10_classes.json
#         raget_evaluation_dataset_10_reports.json
```

### 2. Filter Dataset Quality
```bash
python rag_evaluation_dataset_critic.py \
    --input raget_evaluation_dataset_10_reports.json \
    --output refined_raget_evaluation_dataset_10_reports.json \
    --min-score 4
```

### 3. Run RAG Evaluation
```bash
python RAG_running_golden_DS_eval.py
# Input: refined_raget_evaluation_dataset_10_reports.json
# Output: rag_evaluation_results_reports.json
```

### 4. Combine Results (Optional)
```bash
python combine_golden_RAG_dataset.py \
    golden_dataset.json \
    rag_results.json \
    combined_output.json
```

## Key Insights

1. **Question Type Distribution**: The RAGET methodology ensures comprehensive coverage of different query patterns users might employ.

2. **Quality Filtering Impact**: ~80% of generated questions pass the quality criteria, ensuring high-quality evaluation data.

3. **Query Enhancement**: Sophisticated queries benefit from automatic enhancement, while simple queries are processed as-is to preserve baseline performance metrics.

4. **Infrastructure Requirements**: The GPU-accelerated instance (RTX 6000 Ada 48GB) was essential for:
   - Parallel processing of multiple LLM calls
   - Fast embedding generation
   - Efficient reranking operations

## Future Enhancements

1. **Automated Evaluation Metrics**: Implement RAGAS metrics for systematic quality assessment
2. **Multi-Collection Evaluation**: Cross-collection query handling
3. **Conversation Flow Testing**: Extended multi-turn conversation scenarios
4. **Performance Optimization**: Query caching and batch processing improvements

## Conclusion

This comprehensive pipeline provides a robust framework for evaluating RAG systems on domain-specific documentation. The combination of systematic dataset generation, quality filtering, and thorough evaluation ensures reliable assessment of the RAG system's capabilities in handling SAP ABAP documentation queries.
