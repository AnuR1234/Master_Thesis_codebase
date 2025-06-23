# Open-Source RAG Implementation
Enterprise-grade RAG pipeline using open-source models for SAP ABAP documentation retrieval with anti-hallucination mechanisms.

## Architecture
- **LLM**: Mistral-7B-Instruct-v0.3 (local inference)
- **Embeddings**: E5-Large-v2 (1024-dimensional dense vectors)
- **Vector DB**: Qdrant with hybrid search (dense + sparse BM25)
- **Reranker**: Cross-encoder (ms-marco-MiniLM-L-6-v2)

## Prerequisites
- NVIDIA GPU with 12GB+ VRAM
- Python 3.8+
- CUDA 11.8+ with PyTorch 2.0+
- Docker (for Qdrant)

## Quick Setup
1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Start Qdrant**
```bash
docker run -p 6333:6333 qdrant/qdrant
```

3. **Launch application**
```bash
streamlit run streamlit_app_RAG.py
```

## Core Components
| File | Purpose |
|------|---------|
| `pipeline.py` | RAG orchestrator with anti-hallucination |
| `retriever.py` | Hybrid document retrieval |
| `generator.py` | Local LLM text generation |
| `embedding.py` | E5 embeddings with BM25 support |
| `query_enhancer.py` | Query rewriting and decomposition |
| `config.py` | System configuration and prompts |
| `streamlit_app_RAG.py` | Main web interface |
| `run_system.py` | System management and testing suite |
| `rtx6000_memory_manager.py` |(helper module) GPU memory optimization |
| `memory_cleanup.py` |(helper module) Memory utilities and cleanup |
| `timing_utils.py` |(helper module) Performance timing and monitoring |
| `kill_process.py` |(helper module) GPU process management |
| `model_downloader.py` |(helper module) To downlaod the Open-Source Models Locally |

## Key Features
- ✅ **Anti-hallucination protection** with confidence scoring
- ✅ **Query enhancement** (rewriting/decomposition)
- ✅ **Hybrid retrieval** (dense + sparse vectors)
- ✅ **Local inference** (no API costs)
- ✅ **Multi-collection support** (Classes/Reports)

## Configuration
Key settings in `config.py`:
- `LLM_TEMPERATURE`: 0.7 (conservative for accuracy)
- `DEFAULT_TOP_K`: 5 (retrieved contexts)
- `DENSE_VECTOR_WEIGHT` / `SPARSE_VECTOR_WEIGHT`: 0.7/0.3

## Usage
**Web Interface**: `http://localhost:8501`

**Programmatic**:
```python
from pipeline import EnhancedRAGPipeline
result = await pipeline.process_query("query text")
```

## Performance
- **Memory**: 15-25GB VRAM
- **Context window**: 2048 tokens
- **Zero API costs** (fully local)
