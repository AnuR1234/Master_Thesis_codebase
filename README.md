# Leveraging Large Language Models and Retrieval Augmented Generation Systems for Automated Documentation and Retrieval of Custom ABAP Code in Legacy SAP Systems

**Master Thesis Project**

## Overview

This repository contains the implementation and evaluation of different Retrieval-Augmented Generation (RAG) architectures for SAP ABAP code documentation, developed as part of my master's thesis. The project explores both closed-source and open-source approaches, evaluates their performance, and includes experiments with fine-tuned language models.

## Repository Structure

```
C:.
│   README.md
│
├───.qodo
├───Evaluations
│       .gitkeep
│       Experiment on generation quality of closed and open source RAG systems.pdf
│       List_of_prompts_used.pdf
│       Open_source_vs_closed_RAG_final.pdf
│       pair_wise_evaluation_LLM_as_judge_11_6.pdf
│       Task 2_ Chunking and Embedding Strategies Experimentations .pdf
│       Task 3-comparing different embedding models and retrieval techniques.pdf
│
├───Fine-tuning
│   │   .gitkeep
│   │   Fine-tune-llama-guidance.md
│   │
│   ├───Evaluation
│   │       .gitkeep
│   │       automated_metrics.py
│   │       automatic_metrics.py
│   │       data_preprocessor.py
│   │       enhanced_abap_metrics.py
│   │       Evaluation-guideline.md
│   │       Evaluation-guidelines.md
│   │       evaluation_report.md
│   │       llm_evaluator.py
│   │       main_evaluator.py
│   │       main_eval_test.py
│   │       model_inference.py
│   │       pdsqi_abap_evaluator.py
│   │
│   └───Fine_tune_code
│           .gitkeep
│           convert_llama_tf_to_pt.py
│           deepspeed_config_z3_qlora.yaml
│           run.sh
│           setup_env.sh
│           train_llama3_lora.py
│           utils_llama3.py
│
├───RAG Architectures
│   │   dummy.md
│   │   requirements.txt
│   │
│   ├───Closed-RAG Implementation
│   │   │   .gitkeep
│   │   │   claude.env
│   │   │   cli.py
│   │   │   closed-source-RAG-implementation-guideline.md
│   │   │   combine_golden_RAG_dataset.py
│   │   │   config.py
│   │   │   embedding.py
│   │   │   gemini_context_filter.py
│   │   │   generator.py
│   │   │   golden_dataset.py
│   │   │   Golden_Dataset_creation_and_eval_guideline.md
│   │   │   only_simle_qns.py
│   │   │   pipeline.py
│   │   │   query_enhancer.py
│   │   │   RAG_running_golden_DS_eval.py
│   │   │   retriever.py
│   │   │   stream_lit.py
│   │   │
│   │   └───DeepEval
│   │           .gitkeep
│   │           comprehensive_thesis_eval_final_new_metrics_with_API_costs.py
│   │           DeepEval_RAG_Evaluation_Guide.md
│   │           eval_setup_for_simple_qns_only_v2.py
│   │
│   └───Open-Source RAG
│       │   .gitkeep
│       │   combine_golden_RAG_dataset.py
│       │   combine_retrieved_docs.py
│       │   config.py
│       │   debug_pipeline.py
│       │   embedding.py
│       │   generator.py
│       │   Golden_Dataset_creation_and_evaluation_guidelines.md
│       │   kill_process.py
│       │   memory_cleanup.py
│       │   model_downloader.py
│       │   only_simle_qns.py
│       │   open-RAG-implementation-guidelines.md
│       │   pipeline.py
│       │   query_enhancer.py
│       │   RAG_run_on_Golden_ds.py
│       │   retriever.py
│       │   rtx6000_memory_manager.py
│       │   run_system.py
│       │   streamlit_app_RAG.py
│       │   timing_utils.py
│       │
│       └───DeepEval
│               .gitkeep
│               comprehensive_thesis_eval_final_new_metrics_with_API_costs.py
│               DeepEval_guideline.md
│               eval_setup_for_simple_qns_only_v2.py
│
└───RAG_preprocess_setup
    │   .gitkeep
    │   base_chunker.py
    │   chunking-guidelines.md
    │   chunking_main.py
    │   doc_chunker.py
    │   Experiment2_Exploring_Different_Embeddings_and_Retrieval_Techniques.ipynb
    │   Experiment_3_Optimizing_Generation_Parameters_for_High_Quality_RAG_Responses_in_SAP_ABAP_Code_Documentation.py
    │   instance_manager.py
    │   llm_chunker.py
    │   rename_abap_files.py
    │
    └───Embeddings_generation
            .gitkeep
            embeddings_generation_guidelines.md
            embedding_full_dataset_on_disk_openai_BM25.py
            embedding_full_dataset_on_disk_open_source_e5_BM25.py
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
