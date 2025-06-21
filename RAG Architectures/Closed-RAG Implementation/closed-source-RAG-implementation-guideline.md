# Closed-Source RAG Implementation

Enterprise-grade RAG pipeline using commercial APIs for SAP ABAP documentation retrieval.

## Architecture

- **LLM**: Amazon Bedrock (Claude 3.7 Sonnet)
- **Embeddings**: SAP AI Core (OpenAI proxy) with fallback to OpenAI API
- **Vector DB**: Qdrant with hybrid search (dense + sparse BM25)
- **Reranker**: Cross-encoder (ms-marco-MiniLM-L-6-v2)

## Prerequisites

- AWS account with Bedrock access
- SAP AI Core subscription (optional)
- OpenAI API key (fallback embeddings)
- Docker (for Qdrant)
- Python 3.8+

## Quick Setup

1. **Install dependencies**
```bash
pip install -r requirements.txt
```

2. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your API credentials
```

3. **Start Qdrant**
```bash
docker run -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant
```

4. **Configure AWS**
```bash
aws configure  # Enter your AWS credentials
```

5. **Run application**
```bash
streamlit run stream_lit.py
```

## Core Components

| File | Purpose |
|------|---------|
| `pipeline.py` | Main RAG orchestrator |
| `retriever.py` | Hybrid document retrieval |
| `generator.py` | LLM response generation |
| `embedding.py` | Multi-provider embedding management |
| `query_enhancer.py` | Query rewriting and decomposition |
| `stream_lit.py` | Web interface |
| `cli.py` | Command-line interface |

## Key Features

- ✅ Hybrid retrieval (dense + sparse vectors)
- ✅ Query enhancement with rewriting
- ✅ Multi-turn conversations
- ✅ Confidence level assessment
- ✅ Sub-query decomposition
- ✅ Collection management (Classes/Reports)

## Configuration

Key settings in `.env`:
- `BEDROCK_MODEL`: Claude model selection
- `DENSE_VECTOR_WEIGHT` / `SPARSE_VECTOR_WEIGHT`: Hybrid search tuning
- `DEFAULT_TOP_K`: Number of retrieved contexts
- `USE_RERANKER_DEFAULT`: Enable/disable reranking

## Usage Examples

**Web Interface**: `http://localhost:8501`

**CLI Query**:
```bash
python cli.py query -q "How do ABAP classes implement interfaces?"
```

**Batch Processing**:
```bash
python cli.py batch -f queries.txt -o results.json
```

## Cost Optimization

- Adjust `top_k` to reduce context size
- Disable reranker for simple queries
- Use query caching
- Batch similar queries

## Performance

- **Latency**: 2-5 seconds per query
- **Throughput**: ~12-15 queries/minute
- **Context window**: 4096 tokens
- **Embedding dimension**: 1536

## Troubleshooting

Common issues:
- **Bedrock access**: Verify AWS permissions
- **SAP AI Core**: Check OAuth2 credentials
- **Embeddings fallback**: Ensure OpenAI API key is valid
- **Logs**: Check `rag_pipeline.log` for details

---

See `.env.example` for complete configuration options.
