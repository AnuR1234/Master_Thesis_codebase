# direct_load.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

print(f"PyTorch version: {torch.__version__}")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
print("Tokenizer loaded successfully")

# Load the model directly
try:
    print("Loading model in safetensors format...")
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_safetensors=True  # Explicitly use safetensors
    )
    print("Model loaded successfully!")
    
    # Save locally if needed
    print("Saving model to local directory...")
    model.save_pretrained("./llama3-pt-model", safe_serialization=True)
    tokenizer.save_pretrained("./llama3-pt-model")
    print("Model and tokenizer saved successfully to ./llama3-pt-model")
    
except Exception as e:
    print(f"Error loading model: {e}")