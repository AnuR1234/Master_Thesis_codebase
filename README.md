# Leveraging Large Language Models and Retrieval Augmented Generation Systems for Automated Documentation and Retrieval of Custom ABAP Code in Legacy SAP Systems

**Master Thesis Project**

## Overview

This repository contains the implementation and evaluation of different Retrieval-Augmented Generation (RAG) architectures for SAP ABAP code documentation, developed as part of my master's thesis. The project explores both closed-source and open-source approaches, evaluates their performance, and includes experiments with fine-tuned language models.

## Repository Structure

```
.
├── README.md                           # This file
├── 1-rag-architectures/               # RAG implementations
│   ├── closed-source-rag/             # Enterprise RAG with commercial APIs
│   │   └── README.md                  # Detailed setup and usage instructions
│   └── open-source-rag/               # Community RAG with open models
│       └── README.md                  # Detailed setup and usage instructions
├── 2-evaluations/                     # Comparative analysis
│   ├── README.md                      # Evaluation methodology and results
│   ├── metrics/                       # Performance metrics
│   └── benchmarks/                    # Test datasets and results
├── 3-fine-tuned-llama/               # LLaMA fine-tuning experiments
│   ├── setup/                         # Fine-tuning configuration
│   │   └── README.md                  # Fine-tuning instructions
│   └── evaluation/                    # Model evaluation results
│       └── README.md                  # Performance analysis
└── 4-rag-preprocessing-setup/        # Preprocessing experiments
    ├── chunking/                      # Document chunking strategies
    │   └── README.md                  # Chunking analysis
    ├── retrieval-strategies/          # Different retrieval approaches
    │   └── README.md                  # Strategy comparison
    └── parameter-tests/               # Hyperparameter optimization
        └── README.md                  # Parameter tuning results
```

## Thesis Objectives

1. **Compare closed-source vs open-source RAG architectures** for enterprise documentation
2. **Evaluate performance metrics** including accuracy, latency, and cost
3. **Explore fine-tuning benefits** using LLaMA models on SAP ABAP data
4. **Optimize preprocessing strategies** for improved retrieval quality

## Key Components

### 1. RAG Architectures

- **Closed-Source RAG**: Leverages Amazon Bedrock (Claude 3), SAP AI Core, and OpenAI embeddings
- **Open-Source RAG**: Utilizes open models via Ollama/HuggingFace and open embeddings

*See respective README files in each subdirectory for detailed implementation details.*

### 2. Evaluations

Comprehensive comparison of both architectures across:
- Response quality (groundedness, relevance, hallucination)
- Performance metrics (latency, throughput)
- Cost analysis
- Scalability considerations

*Detailed results available in `2-evaluations/README.md`*

### 3. Fine-tuned LLaMA

Experiments with fine-tuning LLaMA models on SAP ABAP documentation:
- Custom dataset preparation
- Fine-tuning methodology
- Performance comparison with base models

*Setup instructions in `3-fine-tuned-llama/setup/README.md`*

### 4. RAG Preprocessing Setup

Investigation of optimal preprocessing strategies:
- **Chunking**: Comparison of different chunk sizes and overlap strategies
- **Retrieval Strategies**: Dense vs sparse vs hybrid approaches
- **Parameter Tests**: Optimization of retrieval parameters (top-k, reranking, weights)

*Results documented in respective subdirectories*

## Infrastructure

All experiments were conducted on:
- **Hardware**: 1x RTX 6000 Ada 48GB GPU
- **Platform**: vast.ai cloud instance
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.10+

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sap-abap-rag-thesis.git
cd sap-abap-rag-thesis
```

2. Choose a component to explore:
```bash
# For RAG architectures
cd 1-rag-architectures/closed-source-rag/
# OR
cd 1-rag-architectures/open-source-rag/

# Follow the README.md in each directory
```

## Key Findings

1. **Closed-source RAG** provides superior response quality but at higher operational costs
2. **Open-source RAG** offers competitive performance with full data privacy and control
3. **Fine-tuned LLaMA** models show promising results for domain-specific queries
4. **Hybrid retrieval** with optimized chunking significantly improves retrieval accuracy

*Full analysis available in the evaluations section*

## Technologies Used

- **Vector Database**: Qdrant
- **LLMs**: Claude 3 (Bedrock), LLaMA 2/3, open models
- **Embeddings**: OpenAI, SAP AI Core, Sentence Transformers
- **Frameworks**: LangChain, Streamlit, FastAPI
- **Evaluation**: RAGAS, custom metrics

## Research Contributions

1. Comprehensive comparison framework for enterprise RAG systems
2. Optimization strategies for SAP ABAP documentation retrieval
3. Cost-benefit analysis of commercial vs open-source approaches
4. Fine-tuning methodology for domain-specific technical documentation

## Citation

If you use this work in your research, please cite:
```bibtex
@mastersthesis{yourname2024rag,
  title={Retrieval-Augmented Generation for SAP ABAP Documentation: A Comparative Study},
  author={Your Name},
  year={2024},
  school={Your University}
}
```

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Acknowledgments

- Thesis advisor: [Advisor Name]
- SAP ABAP documentation team
- Open-source community contributors

---

**Note**: Each major component has its own detailed README file with specific setup instructions, usage guidelines, and technical details. Please refer to the respective directories for in-depth information.
