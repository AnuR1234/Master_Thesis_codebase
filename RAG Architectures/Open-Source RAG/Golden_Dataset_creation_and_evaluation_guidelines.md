# Open Source RAG Golden Dataset Creation and Evaluation Process

## Overview

This document describes the end-to-end process of creating a golden dataset and evaluating an open-source Retrieval-Augmented Generation (RAG) system for SAP ABAP code documentation. The system utilizes open-source models and frameworks while maintaining enterprise-grade performance.

**Infrastructure Note**: All processes were executed on an external instance equipped with **1x RTX 6000 Ada 48GB GPU** to handle the computationally intensive operations.

## System Architecture

### Core Components

- **Vector Database**: Qdrant with dual vector storage (dense + sparse)
- **Embedding Model**: intfloat/e5-large-v2 (open-source, 1024 dimensions)
- **Sparse Embeddings**: Qdrant/bm25 for keyword matching
- **LLM**: Mistral-7B-Instruct-v0.3 (open-source, locally hosted)
- **Reranker**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **Framework**: Custom pipeline with anti-hallucination controls

### Key Differences from Cloud-Based Solution

1. **Local Model Hosting**: All models run on-device (RTX 6000 Ada)
2. **Open-Source Stack**: No dependency on commercial APIs
3. **Enhanced Memory Management**: Optimized for 48GB VRAM
4. **Anti-Hallucination Controls**: Strict document adherence mechanisms

## Phase 1: Environment Setup

### 1.1 Model Preparation

The system requires downloading and caching models locally:

```python
# Model paths configuration
MODELS_DIR = "/home/user/rag_models"
MODEL_PATHS = {
    "mistralai/Mistral-7B-Instruct-v0.3": "/home/user/rag_models/mistralai_Mistral-7B-Instruct-v0.3",
    "intfloat/e5-large-v2": "/home/user/rag_models/intfloat_e5-large-v2"
}
```

### 1.2 RTX 6000 Ada Optimization

```python
# GPU memory configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096,expandable_segments:True"
RTX_6000_OPTIMIZATIONS = {
    "max_memory_gb": 45,
    "embedding_batch_size": 32,
    "generation_batch_size": 8,
    "max_sequence_length": 8192,
    "attention_implementation": "eager"  # FlashAttention disabled for compatibility
}
```

## Phase 2: Collection Configuration

### 2.1 Dual Collection Setup

```python
COLLECTIONS = {
    "classes": {
        "name": "SAP_ABAP_CODE_DOCUMENTATION_E5_BM25_CLASSES_TEST",
        "description": "SAP ABAP Classes Documentation"
    },
    "reports": {
        "name": "SAP_ABAP_CODE_DOCUMENTATION_E5_BM25_REPORTS_TEST",
        "description": "SAP ABAP Reports Documentation"
    }
}
```

### 2.2 Vector Configuration

- **Dense Vectors**: E5 embeddings (1024 dimensions) with "query:" prefix
- **Sparse Vectors**: BM25 for keyword-based retrieval
- **Hybrid Search**: RRF (Reciprocal Rank Fusion) combining both approaches

## Phase 3: Golden Dataset Creation

### 3.1 Dataset Generation Process

The golden dataset creation follows the same RAGET methodology as the cloud-based system but with open-source model adaptations:

```python
# Using Mistral-7B for dataset generation
class RAGETQuestionGenerator:
    def __init__(self, collection_type, document_pool, top_topic_names):
        self.generator = EnhancedRAGGenerator()  # Uses Mistral-7B
```

### 3.2 Question Type Distribution

Same six question types as the cloud-based system:
- Simple questions (direct factual queries)
- Complex questions (requiring deep understanding)
- Distracting questions (with irrelevant information)
- Situational questions (context-specific scenarios)
- Double questions (two-part queries)
- Conversational questions (follow-up queries)

### 3.3 Open-Source Generation Challenges

Key adaptations for open-source models:
- **Lower temperature settings**: 0.05 for strict adherence
- **Reduced top_p**: 0.8 to minimize hallucination
- **Strict prompting**: Enhanced anti-hallucination instructions

## Phase 4: Anti-Hallucination Pipeline

### 4.1 Enhanced System Prompts

```python
STRICT_DOCUMENT_ADHERENCE_PROMPT = """You are an expert SAP ABAP documentation assistant. You must follow these CRITICAL rules:

ABSOLUTE REQUIREMENTS:
1. Start with "Confidence: [HIGH/MEDIUM/LOW]"
2. Use ONLY information from the provided documents
3. Never add information from your general knowledge
4. If information is not in the documents, explicitly state "This information is not available in the provided documentation"
5. Quote exact phrases when possible using [Document X: "exact quote"]

FORBIDDEN ACTIONS:
- Adding parameters, methods, or functionality not mentioned in the documents
- Describing general SAP concepts not present in the provided context
- Making assumptions about code behavior beyond what's documented
"""
```

### 4.2 Query Enhancement Disabled

Unlike some implementations, query enhancement is **explicitly disabled** to prevent retrieval interference:

```python
# In pipeline.py
self.query_enhancement_enabled = False
logger.info("FIXED: Query enhancement DISABLED to prevent retrieval interference")
```

## Phase 5: Retrieval Implementation

### 5.1 E5 Embedding Configuration

```python
class E5EmbeddingModel:
    def encode(self, texts, batch_size=16):
        # CRITICAL: Use correct E5 prefix for queries
        formatted_texts = []
        for text in texts:
            prefix = "query: "  # E5 requires this prefix
            formatted_text = f"{prefix}{text}"
            formatted_texts.append(formatted_text)
```

### 5.2 Hybrid Retrieval Strategy

```python
def _perform_hybrid_retrieval(self, query, dense_vec, collection_name, limit):
    # Generous prefetch limits for comprehensive retrieval
    prefetch_limit = limit * 8
    
    prefetch = [
        models.Prefetch(
            query=sparse_vector,
            using="BM25",
            limit=prefetch_limit,
        ),
        models.Prefetch(
            query=dense_vec,
            using="E5",
            limit=prefetch_limit,
        ),
    ]
    
    # RRF fusion for balanced results
    response = self.client.query_points(
        collection_name=collection_name,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit
    )
```

## Phase 6: Response Generation

### 6.1 Mistral-7B Configuration

```python
GENERATION_CONFIG = {
    "max_new_tokens": 2048,      # Controlled length
    "temperature": 0.9,         
    "top_p": 0.7,               # Reduced creativity
    "top_k": 30,                # Focused token selection
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,   # Prevent repetition
}
```

### 6.2 Context Formatting

```python
def _format_context_with_strict_boundaries(self, context_results):
    # Include up to 5 documents for comprehensive context
    max_docs = min(5, len(context_results))
    
    for i, (result, score) in enumerate(context_results[:max_docs]):
        # Clear document boundaries
        content_parts.append(f"Document {i+1}: Title: {title}")
        
        # Clean code preservation
        if code_snippet:
            clean_code = self._extract_comprehensive_code(code_snippet)
            content_parts.append(f"Code Snippet:\n\n\n{clean_code}")
```

## Phase 7: Evaluation Process

### 7.1 Enhanced RAG Evaluator

```python
class EnhancedRAGEvaluator:
    def __init__(self, golden_dataset_path, output_path):
        self.pipeline = EnhancedRAGPipeline()
        self.enable_retries = True
        self.max_retries = 2
```

### 7.2 Evaluation Metrics

The evaluation captures:
- **Response Quality**: Via confidence levels (HIGH/MEDIUM/LOW)
- **Retrieval Effectiveness**: Number and relevance of retrieved documents
- **Latency**: End-to-end processing time
- **Hallucination Detection**: Risk assessment (HIGH/MEDIUM/LOW)

### 7.3 Memory Management

Critical for RTX 6000 Ada optimization:
```python
def clear_gpu_memory(self):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

## Phase 8: Results Processing

### 8.1 Output Format

```json
{
    "Question": "What is the purpose of the escape_special_characters method?",
    "response": "Based on the provided documentation...",
    "question_type": "simple",
    "latency": 3.45,
    "confidence_level": "HIGH",
    "retrieved_docs": [...],
    "hallucination_check": {
        "risk_level": "LOW",
        "suspicious_phrases": [],
        "undocumented_parameters": []
    }
}
```

### 8.2 Dataset Combination

The `combine_golden_RAG_dataset.py` script merges:
- Golden dataset questions
- RAG system responses
- Retrieved context (from RAG evaluation)
- Performance metrics

## Performance Characteristics

### RTX 6000 Ada Utilization

1. **Memory Usage**:
   - Mistral-7B: ~15GB VRAM
   - E5 Embeddings: ~2GB VRAM
   - Reranker: ~1GB VRAM
   - Available for batch processing: ~25GB

2. **Processing Speed**:
   - Embedding generation: 32 texts/batch
   - LLM generation: 8 sequences/batch
   - Average latency: 3-5 seconds per query

3. **Optimization Strategies**:
   - No FlashAttention (compatibility)
   - FP16 precision throughout
   - Aggressive memory cleanup between queries

## Key Differences from Cloud Implementation

1. **Model Selection**:
   - Cloud: Claude 3.7 Sonnet
   - Open-Source: Mistral-7B-Instruct-v0.3

2. **Embedding Approach**:
   - Both use dense+sparse, but open-source uses E5 with specific prefixing

3. **Cost Structure**:
   - Cloud: Per-token API costs
   - Open-Source: One-time GPU investment

4. **Latency Profile**:
   - Cloud: Network + API processing
   - Open-Source: Pure GPU computation

5. **Scalability**:
   - Cloud: Unlimited concurrent requests
   - Open-Source: Limited by GPU memory

## Anti-Hallucination Mechanisms

### 1. Strict Prompt Engineering
- Explicit forbidding of external knowledge
- Required confidence declarations
- Mandatory document citations

### 2. Generation Control
- Constrained token selection
- No beam search to reduce creativity

### 3. Post-Processing Validation
- Hallucination risk scoring
- Undocumented entity detection
- Confidence adjustment based on risk

### 4. Query Processing
- No query enhancement/rewriting
- Direct document retrieval
- Strict context boundaries

## How to Run the Scripts

### Prerequisites

1. **Hardware Requirements**:
   - GPU: RTX 6000 Ada 48GB (or similar high-memory GPU)
   - RAM: 32GB+ recommended
   - Storage: 100GB+ for models and datasets

2. **Software Requirements**:
   ```bash
   # Python 3.8+
   python --version
   
   # CUDA toolkit (11.8 or 12.1 recommended)
   nvcc --version
   
   # PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Environment Setup**:
   ```bash
   # Clone the repository
   git clone <repository_url>
   cd rag_pipeline_opensource
   
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

### Step 1: Download Models

First, download all required models locally:

```bash
# Download the LLM (Mistral-7B)
python model_downloader.py --model mistralai/Mistral-7B-Instruct-v0.3 --token YOUR_HF_TOKEN

# Download the embedding model
python model_downloader.py --embedding

# List downloaded models
python model_downloader.py --list
```

**Note**: You'll need a Hugging Face token for gated models like Mistral. Get one from https://huggingface.co/settings/tokens

### Step 2: Start Qdrant Vector Database

```bash
# Using Docker
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant

# Or using binary
./qdrant --config-path config/config.yaml
```

### Step 3: Generate Golden Dataset

```bash
# Generate golden dataset for all collections
python golden_dataset.py

# This will create:
# - raget_evaluation_dataset_10_classes.json
# - raget_evaluation_dataset_10_reports.json
```

**Configuration options**:
```python
# In golden_dataset.py, adjust:
questions_per_type = 15  # Number of questions per type
collection_types = ["classes", "reports"]  # Collections to process
```

### Step 4: Filter Dataset Quality (Optional)

```bash
# Run quality filtering with Gemini (if you have API access)
python rag_evaluation_dataset_critic.py \
    --input raget_evaluation_dataset_10_reports.json \
    --output refined_raget_evaluation_dataset_10_reports.json \
    --min-score 4 \
    --max-questions 100
```

### Step 5: Run RAG Evaluation

```bash
# Basic evaluation
python RAG_run_on_Golden_ds.py \
    --dataset refined_raget_evaluation_dataset_10_reports.json \
    --output rag_evaluation_results_reports.json \
    --collection reports

# Full evaluation with options
python RAG_run_on_Golden_ds.py \
    --dataset golden_dataset.json \
    --output results.json \
    --collection classes \
    --limit 50 \
    --start 0 \
    --max-retries 2 \
    --save-interval 10
```

**Command-line options**:
- `--dataset`: Path to golden dataset
- `--output`: Output file for results
- `--collection`: Collection type (classes/reports)
- `--limit`: Number of questions to process
- `--start`: Starting index (for resuming)
- `--no-retries`: Disable retry mechanism
- `--max-retries`: Maximum retries per question
- `--save-interval`: Save results every N questions

### Step 6: Process Retrieved Documents

```bash
# Extract and format retrieved documents
python combine_retrieved_docs.py rag_evaluation_results.json processed_results.json
```

### Step 7: Combine Datasets

```bash
# Merge golden dataset with RAG responses
python combine_golden_RAG_dataset.py \
    golden_dataset.json \
    rag_evaluation_results.json \
    merged_output.json
```

### Step 8: Run DeepEval Analysis (Optional)

```bash
# Run comprehensive evaluation with DeepEval
python comprehensive_thesis_eval_final_new_metrics_with_API_costs.py

# For simple questions only
python eval_setup_for_simple_qns_only_v2.py
```

### Step 9: Extract Simple Questions (Optional)

```bash
# Filter only simple questions for focused evaluation
python only_simle_qns.py
```

### Running the Streamlit Interface

For interactive testing:

```bash
# Start the Streamlit app
streamlit run streamlit_app_RAG.py

# The app will be available at http://localhost:8501
```

### Monitoring and Debugging

1. **Check GPU Usage**:
   ```bash
   # Monitor GPU memory and utilization
   watch -n 1 nvidia-smi
   ```

2. **View Logs**:
   ```bash
   # Application logs
   tail -f rag_pipeline.log
   
   # Evaluation logs
   tail -f rag_evaluation.log
   ```

3. **Memory Management**:
   If you encounter OOM errors, adjust batch sizes in `config.py`:
   ```python
   RTX_6000_OPTIMIZATIONS = {
       "embedding_batch_size": 16,  # Reduce if OOM
       "generation_batch_size": 4,   # Reduce if OOM
   }
   ```

### Common Issues and Solutions

1. **CUDA Out of Memory**:
   ```python
   # In config.py, reduce batch sizes
   BATCH_SIZE = 8  # Instead of 16
   
   # Enable memory cleanup
   UNLOAD_UNUSED_MODELS = True
   ```

2. **Model Download Failures**:
   ```bash
   # Clean and retry
   python model_downloader.py --model MODEL_NAME --clean --force --token TOKEN
   ```

3. **Slow Generation**:
   ```python
   # In generator.py, adjust generation config
   "max_new_tokens": 512,  # Reduce from 1024
   ```

4. **Collection Not Found**:
   ```bash
   # Verify Qdrant collections
   curl http://localhost:6333/collections
   ```

### Production Deployment Tips

1. **Use Process Manager**:
   ```bash
   # Install PM2
   npm install -g pm2
   
   # Start services
   pm2 start streamlit_app_RAG.py --interpreter python
   pm2 start qdrant
   ```

2. **Set Environment Variables**:
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Edit with your settings
   nano .env
   ```

3. **Enable GPU Persistence**:
   ```bash
   # For NVIDIA GPUs
   sudo nvidia-smi -pm 1
   ```

## Conclusion

The open-source RAG implementation demonstrates that enterprise-grade document Q&A systems can be built entirely with open-source components. The key success factors are:

1. **Proper GPU utilization** (RTX 6000 Ada 48GB)
2. **Strict anti-hallucination controls** at every pipeline stage
3. **Hybrid retrieval** combining semantic and keyword search
4. **Careful prompt engineering** for open-source LLMs
5. **Comprehensive evaluation** using golden datasets

The system achieves comparable quality to cloud-based solutions while maintaining complete data privacy and control over the infrastructure.
