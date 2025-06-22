# Embedding Generation Guide for SAP ABAP Code Documentation

This guide explains how to set up and use the embedding generation scripts for creating a vector database of SAP ABAP code documentation using Qdrant.

## Overview

The repository contains two main embedding generation scripts:
- **`embedding_full_dataset_on_disk_openai_BM25.py`** - Uses OpenAI's text-embedding-ada-002 model
- **`embedding_full_dataset_on_disk_open_source_e5_BM25.py`** - Uses the open-source E5-large-v2 model

Both scripts implement hybrid search capabilities combining dense embeddings with BM25 sparse vectors for improved retrieval performance.

## Prerequisites

### 1. Install Docker
Ensure Docker is installed on your system. You can download it from [Docker's official website](https://www.docker.com/get-started).

### 2. Install and Run Qdrant

Pull the Qdrant Docker image and run it:

```bash
# Pull the Qdrant image
docker pull qdrant/qdrant

# Run Qdrant with persistent storage
docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant
```

This command:
- Exposes port 6333 for HTTP API and 6334 for gRPC
- Creates a local `qdrant_storage` directory for persistent data storage
- The `:z` flag ensures proper SELinux context (important for Linux systems)

### 3. Set Up Python Environment

Create and activate a virtual environment:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate
```

### 4. Install Required Dependencies

```bash
pip install pandas numpy tqdm python-dotenv
pip install qdrant-client
pip install fastembed
pip install sentence-transformers  # For E5 model
pip install requests asyncio        # For OpenAI model
```

For the OpenAI version, you'll also need:
```bash
pip install gen-ai-hub-sdk  # If using SAP AI Core
# OR
pip install openai          # If using OpenAI directly
```

### 5. Create Environment File

Create a `.env` file (rename from `rag.env` if needed) in your project directory:

```env
# For OpenAI version with SAP AI Core
AICORE_AUTH_URL=your_auth_url
AICORE_CLIENT_ID=your_client_id
AICORE_CLIENT_SECRET=your_client_secret
AICORE_RESOURCE_GROUP=default
AICORE_BASE_URL=your_base_url
AICORE_DEPLOYMENT_ID_embed_1=your_deployment_id_1
AICORE_DEPLOYMENT_ID_embed_2=your_deployment_id_2
AICORE_DEPLOYMENT_ID_embed_3=your_deployment_id_3

# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

**Note**: For the E5 open-source model, you don't need API credentials.

## Project Structure

```
embeddings_generation/
├── embedding_full_dataset_on_disk_openai_BM25.py
├── embedding_full_dataset_on_disk_open_source_e5_BM25.py
├── rag.env (or .env)
├── README.md
└── qdrant_storage/ (created by Docker)
```

## Data Preparation

Your CSV file should have the following structure:

| Column | Description | Required |
|--------|-------------|----------|
| `doc` | Full text content of the document | Yes |
| `title` | Document title | Yes |
| `code_snippet` | Code snippet from the document | Yes |
| `length` | Document length | Yes |
| `filename` | Source filename | Yes |

## Usage

### Option 1: Using OpenAI Embeddings

1. Update the CSV path in the script:
```python
csv_path = "/path/to/your/output_chunk_file.csv"
```

2. Update the collection name if needed:
```python
COLLECTION_NAME = "SAP_ABAP_CODE_DOCUMENTATION_OPENAI"
```

3. Run the script:
```bash
python embedding_full_dataset_on_disk_openai_BM25.py
```

### Option 2: Using E5 Open-Source Embeddings

1. Update the CSV path in the script:
```python
csv_path = "/path/to/your/output_chunk_file.csv"
```

2. Update the collection name if needed:
```python
COLLECTION_NAME = "SAP_ABAP_CODE_DOCUMENTATION_E5_BM25_REPORTS"
```

3. Run the script:
```bash
python embedding_full_dataset_on_disk_open_source_e5_BM25.py
```

## Key Features

### 1. Hybrid Search
Both scripts implement hybrid search combining:
- **Dense vectors**: Semantic embeddings (OpenAI ada-002 or E5-large-v2)
- **Sparse vectors**: BM25 for keyword-based search

### 2. Resumable Processing
- Progress is saved to `embedding_progress.json` or `embedding_progress_reports.json`
- If interrupted, the script resumes from the last successful batch
- Prevents duplicate uploads by checking existing IDs

### 3. Batch Processing
- Documents are processed in configurable batches (default: 128 for E5, 300 for OpenAI)
- Automatic pausing between batches to prevent rate limiting
- Progress updates after each batch

### 4. On-Disk Storage
- Vectors stored on disk instead of memory for large datasets
- HNSW index also stored on disk
- Optimized for handling millions of documents

### 5. Error Handling
- Automatic retries with exponential backoff for API errors
- Graceful handling of rate limits (OpenAI version)
- Option to continue or abort on errors

## Configuration Options

### Common Settings
```python
BATCH_SIZE = 128           # Documents per batch
BATCH_PAUSE = 5           # Seconds between batches
PROGRESS_FILE = "embedding_progress.json"
```

### OpenAI-Specific Settings
```python
MAX_CONCURRENT = 10       # Maximum concurrent API calls
OPENAI_VECTOR_SIZE = 1536 # ada-002 embedding dimension
```

### E5-Specific Settings
```python
E5_VECTOR_SIZE = 1024     # E5-large-v2 embedding dimension
```

## Monitoring Progress

The scripts provide detailed progress information:
- Current batch number and total batches
- Documents processed per second
- Embedding generation time
- Upload success/failure status
- Automatic progress saving

## Troubleshooting

### 1. Qdrant Connection Error
- Ensure Qdrant Docker container is running
- Check if ports 6333/6334 are not blocked
- Verify QDRANT_HOST and QDRANT_PORT in your .env file

### 2. Out of Memory Errors
- Reduce BATCH_SIZE
- Ensure on-disk storage is enabled (default in provided scripts)
- Check available disk space

### 3. API Rate Limits (OpenAI)
- The script handles rate limits automatically with retries
- Increase BATCH_PAUSE if needed
- Consider using multiple deployment IDs for load balancing

### 4. Slow Processing
- For E5 model: Consider using GPU if available
- Reduce batch size for faster progress updates
- Check disk I/O performance

## Best Practices

1. **Start with a small test dataset** to verify everything works correctly
2. **Monitor disk space** - Qdrant storage can grow large with millions of documents
3. **Keep the progress file** - It's your safety net for resuming interrupted jobs
4. **Use meaningful collection names** to organize different datasets
5. **Regular backups** of the qdrant_storage directory for production use



## Next Steps

After successful embedding generation:
1. Verify data in Qdrant using the Qdrant dashboard (http://localhost:6333/dashboard)
2. Test search queries using the Qdrant API
3. Implement your retrieval pipeline using the generated embeddings
4. Fine-tune hybrid search weights for optimal performance

## Support

For issues or questions:
1. Check the console output for detailed error messages
2. Review the progress file for the last successful batch
3. Ensure all dependencies are correctly installed
4. Verify your CSV file format matches the expected schema
