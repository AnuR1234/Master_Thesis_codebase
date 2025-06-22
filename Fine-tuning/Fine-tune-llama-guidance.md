# Llama 3.1 Fine-tuning for SAP ABAP Technical Documentation

Fine-tuned Meta-Llama-3.1-8B-Instruct model for generating comprehensive technical documentation from SAP ABAP code using LoRA (Low-Rank Adaptation) with DeepSpeed optimization.

## 🤗 Pre-trained Model

The fine-tuned weights are available on Hugging Face:

**(Anu123/llama3-8b-lora-finetune)**

## Architecture
- **Base Model**: Meta-Llama-3.1-8B-Instruct
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Training Framework**: DeepSpeed with ZeRO Stage 3
- **Quantization**: 4-bit with bitsandbytes
- **Template**: Llama 3 chat format with ABAP-specific instructions

## Training Configuration
- **LoRA Parameters**: r=32, alpha=32, dropout=0.1
- **Batch Size**: 8 per device × 16 gradient accumulation = 128 effective
- **Learning Rate**: 2e-5 with cosine scheduling
- **Max Sequence Length**: 2048 tokens
- **Training Epochs**: 3
- **Optimizer**: AdamW with weight decay 1e-4

## Core Components

| File | Purpose |
|------|---------|
| `train_llama3_lora.py` | Main training script with custom SFTTrainer |
| `utils_llama3.py` | Model setup and dataset creation utilities |
| `deepspeed_config_z3_qlora.yaml` | DeepSpeed ZeRO-3 configuration |
| `run.sh` | Training execution script |
| `setup_env.sh` | Environment setup script |
| `convert_llama_tf_to_pt.py` | Model conversion utility |
| `req.txt` | Python dependencies |

## Prerequisites
- **GPU**: 2× NVIDIA GPUs with 16GB+ VRAM each
- **System RAM**: 64GB+ recommended
- **Python**: 3.8+
- **CUDA**: 11.8+ or 12.x
- **DeepSpeed**: 0.16.7
- **Transformers**: 4.51.3

## Quick Setup

### 1. Environment Setup
```bash
# Make setup script executable and run
chmod +x setup_env.sh
./setup_env.sh

# Activate environment
source llama3_env/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r req.txt
```

### 3. Download Base Model
```bash
python convert_llama_tf_to_pt.py
```

### 4. Prepare Dataset
Ensure your dataset is in JSONL format with fields:
```json
{
  "instruction": "Generate comprehensive technical documentation for the following ABAP code:",
  "code": "CLASS zcl_example DEFINITION...",
  "documentation": "## Overview\nThis class provides..."
}
```

### 5. Start Training
```bash
chmod +x run.sh
./run.sh
```

## Using the Fine-tuned Model

### Download from Hugging Face
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Download the fine-tuned model
model_name = "your-username/llama3.1-8b-abap-docs"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
```

### Generate Documentation
```python
def generate_abap_docs(code):
    prompt = f"""<s>[INST] Generate comprehensive technical documentation for the following ABAP code:

```abap
{code}
```[/INST]"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], 
                               skip_special_tokens=True)
    return response

# Example usage
abap_code = """
CLASS zcl_data_processor DEFINITION
  PUBLIC
  FINAL
  CREATE PUBLIC.
  
  PUBLIC SECTION.
    METHODS: process_data
      IMPORTING iv_input TYPE string
      RETURNING VALUE(rv_result) TYPE string.
ENDCLASS.
"""

documentation = generate_abap_docs(abap_code)
print(documentation)
```

### Upload Custom Model to Hugging Face
```python
from huggingface_hub import HfApi, login

# Login to Hugging Face
login()

# Push model to hub
model.push_to_hub("your-username/your-model-name")
tokenizer.push_to_hub("your-username/your-model-name")
```

## Training Features

### Custom SFTTrainer
- **Metrics Tracking**: Real-time loss, accuracy, and learning rate logging
- **Plot Generation**: Automatic visualization of training progress
- **Memory Optimization**: Efficient gradient checkpointing and caching
- **Evaluation**: Comprehensive validation during training

### DeepSpeed Integration
- **ZeRO Stage 3**: Parameter and optimizer state partitioning
- **CPU Offloading**: Reduced GPU memory usage
- **Mixed Precision**: BF16 training for efficiency
- **Gradient Accumulation**: Large effective batch sizes

### LoRA Configuration
```python
LoraConfig(
    r=32,                    # Rank
    lora_alpha=32,          # Scaling factor
    lora_dropout=0.1,       # Dropout rate
    bias="none",            # No bias adaptation
    task_type="CAUSAL_LM",  # Causal language modeling
    target_modules=[        # Target attention and MLP layers
        "q_proj", "k_proj", "v_proj", "o_proj",
        "down_proj", "up_proj", "gate_proj", "lm_head"
    ]
)
```

## Documentation Template

The model generates structured documentation following this template:

```markdown
## Overview
[Brief description of the class purpose and functionality]

## Class Definition
| Aspect | Details |
|--------|---------|
| Scope | PUBLIC/PROTECTED/PRIVATE |
| Type | FINAL/ABSTRACT |
| Superclass | [Parent class name] |

## Method Implementation Details
#### [Method name]
- **Logic Flow**: Step-by-step description
- **Error Handling**: Exception management
- **Dependencies**: Required methods/classes
- **Key Variables**: Important variables and purposes
```

## Performance Metrics

Training results on ABAP documentation dataset:
- **Training Loss**: Converged to ~1.2
- **Validation Loss**: ~1.4
- **Token Accuracy**: ~85%
- **Training Time**: ~4 hours on 2×RTX 4090
- **Model Size**: ~8B parameters + 67M LoRA parameters

## Advanced Usage

### Custom Training Data
```python
# Prepare your ABAP dataset
def prepare_abap_dataset(code_files, documentation_files):
    dataset = []
    for code, docs in zip(code_files, documentation_files):
        example = {
            "instruction": "Generate comprehensive technical documentation for the following ABAP code:",
            "code": code,
            "documentation": docs
        }
        dataset.append(example)
    return dataset
```

### Evaluation Metrics
```python
# The training script automatically tracks:
# - Training/Validation Loss
# - Token-level Accuracy  
# - Learning Rate Schedule
# - Gradient Norms
# - Memory Usage
```

### Model Deployment
```python
# For production deployment
import torch
from transformers import pipeline

# Create pipeline
doc_generator = pipeline(
    "text-generation",
    model="your-username/llama3.1-8b-abap-docs",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Generate documentation
result = doc_generator(
    prompt,
    max_new_tokens=1024,
    temperature=0.7,
    do_sample=True
)
```

## Troubleshooting

### Common Issues
- **CUDA OOM**: Reduce batch size or enable gradient checkpointing
- **DeepSpeed errors**: Ensure proper GPU configuration and memory
- **Model loading**: Verify Hugging Face authentication
- **Slow training**: Check GPU utilization and adjust batch sizes

### Memory Optimization
```bash
# Set environment variables for memory efficiency
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=1
```

### Monitoring Training
```python
# Training metrics are automatically saved to:
# - output_dir/metrics/training_metrics.csv
# - output_dir/plots/loss_comparison_latest.png
# - output_dir/logs/training.log
```

---

**Note**: Replace `your-username/llama3.1-8b-abap-docs` with your actual Hugging Face repository URL when uploading the model.
