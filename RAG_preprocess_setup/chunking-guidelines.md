# ABAP Documentation Chunking Framework

This preprocessing component of my master's thesis breaks down ABAP documentation into semantically coherent chunks for evaluation pipeline processing.

## Core Scripts

### Main Components
- **`chunking_main.py`** - Primary orchestrator with batch processing
- **`instance_manager.py`** - Manages multiple processing instances and file distribution
- **`doc_chunker.py`** - Core chunker for ABAP documentation and code pairs
- **`base_chunker.py`** - Base chunking class with header-aware splitting
- **`llm_chunker.py`** - LLM-based semantic chunking via SAP AI Core
- **`rename_abap_files.py`** - Utility to rename `.prog.abap` to `.abap` files

## Key Features

- **Multi-instance processing** - Parallel processing across workers with checkpoint coordination
- **Adaptive chunking** - Header-aware, semantic, and rule-based fallback strategies
- **ABAP integration** - Links documentation chunks with corresponding code snippets
- **Robust handling** - Rate limiting, checkpoints, and fallback mechanisms for LLM failures

## Setup

Create a `.env` file (`chunk.env`) with required configurations:

```bash
# SAP AI Core Authentication
AICORE_AUTH_URL=your_auth_url
AICORE_BASE_URL=your_base_url
AICORE_CLIENT_ID=your_client_id
AICORE_CLIENT_SECRET=your_client_secret
AICORE_RESOURCE_GROUP=your_resource_group

# Deployment IDs (for multiple instances)
AICORE_DEPLOYMENT_ID=your_default_deployment_id
AICORE_DEPLOYMENT_ID_1=your_instance_1_deployment_id
AICORE_DEPLOYMENT_ID_2=your_instance_2_deployment_id
AICORE_DEPLOYMENT_ID_3=your_instance_3_deployment_id
```

## Usage

```bash
# Basic chunking
python chunking_main.py --doc-dir /path/to/docs --code-dir /path/to/code

# With LLM semantic chunking
python chunking_main.py --doc-dir /path/to/docs --code-dir /path/to/code --use-llm

# Batch processing with custom settings
python chunking_main.py --batch-size 500 --instances 3 --max-chars 4000
```

## Output

Generates JSON files with structured chunks containing:
- Unique identifiers and navigation links
- Associated ABAP code snippets
- Section metadata and content
- Processing information

This framework prepares ABAP documentation data for the subsequent evaluation phases in my thesis research.
