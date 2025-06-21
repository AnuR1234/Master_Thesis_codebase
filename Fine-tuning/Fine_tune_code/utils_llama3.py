# utils_llama3.py
import os
import logging
import torch
from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig
)
from peft import LoraConfig, prepare_model_for_kbit_training

logging.basicConfig(
    level=logging.WARN,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Template for ABAP class documentation
CLASS_TEMPLATE = """## Overview
[Brief description of the class purpose and functionality]
## Class Definition
| Aspect | Details |
|--------|---------|
| Scope | [PUBLIC/PROTECTED/PRIVATE] |
| Type | [FINAL/ABSTRACT] |
| Superclass | [Parent class name] |
| Create Permission | [PUBLIC/PROTECTED/PRIVATE] |
| Friends | [List of friend classes] |
## Implementation Overview
[General description of the implementation approach]
## Method Dependencies
| Method | Calls | Called By |
|--------|-------|-----------|
| [Method name] | [Methods called by this method] | [Methods calling this method] |
## Redefined Methods
| Method | Source | Implementation Details |
|--------|--------|----------------------|
| [Method name] | [Original class] | [Description of implementation] |
## Database Tables Used
| Table Name | Purpose | Key Fields |
|------------|---------|------------|
| [Table name] | [Description of usage] | [Key fields used] |
## Critical Sections
| Section | Methods Involved | Purpose | Considerations |
|---------|-----------------|---------|----------------|
| [Section name] | [List of methods] | [Purpose] | [Special considerations] |
## Method Implementation Details
#### [Method name]
- Logic Flow: [Step-by-step description of the method logic]
- Error Handling: [Description of error handling in the method]
- Dependencies: [Methods or functions this method depends on]
- Key Variables: [Important variables and their purposes]
"""

# Llama 3 chat template using the correct format
def format_llama_prompt(instruction, code, documentation=None):
    if documentation:
        # For training examples that include both input and expected output
        prompt = f"<s>[INST] Generate comprehensive technical documentation for the following ABAP code:\n\n```abap\n{code}\n```[/INST] {documentation}</s>"
    else:
        # For inference only (no expected output)
        prompt = f"<s>[INST] Generate comprehensive technical documentation for the following ABAP code:\n\n```abap\n{code}\n```[/INST]"
    
    return prompt

def create_datasets(tokenizer, data_args, training_args):
    """
    Reads a jsonl file, tokenizes everything once (in parallel).
    We KEEP the 'text' column so that SFTTrainer can do its final processing.
    """
    cache_dir = "./cached_tokenized_data_val"
    if os.path.isdir(cache_dir):
        logger.info(f"Loading tokenized dataset from cache: {cache_dir}")
        try:
            ds = load_from_disk(cache_dir)
            train_dataset = ds["train"]
            eval_dataset = ds["test"] if "test" in ds else None
            
            # Apply max_train_samples if specified
            if data_args.max_train_samples is not None:
                train_dataset = train_dataset.select(range(min(len(train_dataset), data_args.max_train_samples)))
                
            # Apply max_eval_samples if specified
            if eval_dataset is not None and data_args.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))
                
            logger.info(f"Train dataset size: {len(train_dataset)}")
            if eval_dataset:
                logger.info(f"Eval dataset size: {len(eval_dataset)}")
            return train_dataset, eval_dataset
        except Exception as e:
            logger.warning(f"Error loading cached dataset: {e}")
            logger.info("Will rebuild dataset from scratch")

    logger.info(f"Loading dataset from: {data_args.dataset_name}")

    try:
        # Check if dataset exists
        if not os.path.exists(data_args.dataset_name):
            logger.error(f"Dataset file not found: {data_args.dataset_name}")
            directory = os.path.dirname(data_args.dataset_name)
            if os.path.exists(directory):
                logger.info(f"Files in directory {directory}:")
                logger.info(str(os.listdir(directory)))
            raise FileNotFoundError(f"Dataset file not found: {data_args.dataset_name}")
            
        dataset = load_dataset("json", data_files=data_args.dataset_name, split="train")
        logger.info(f"Raw dataset size: {len(dataset)}")
        logger.info(f"Dataset columns: {dataset.column_names}")

        # Apply max_train_samples if specified (before processing to save computation)
        if data_args.max_train_samples is not None:
            dataset = dataset.select(range(min(len(dataset), data_args.max_train_samples)))
            logger.info(f"Limited dataset size to {len(dataset)} samples")

        def build_prompt_and_tokenize(example):
            # Check for required fields
            if not example.get("code", ""):
                return {
                    "text": "",
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": []
                }
                
            # Build a Llama-style prompt using Llama 3 chat format
            formatted_text = format_llama_prompt(
                instruction=example.get("instruction", "Generate comprehensive technical documentation for the following ABAP code:"),
                code=example.get("code", ""),
                documentation=example.get("documentation", "")
            )

            # Keep 'text' so SFTTrainer can see it
            example["text"] = formatted_text

            # Tokenize once
            tokenized = tokenizer(
                formatted_text,
                truncation=True,
                max_length=data_args.max_seq_length,
                padding="max_length",
                return_tensors=None,
                return_attention_mask=True
            )

            return {
                "text": formatted_text,  # keep it
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": tokenized["input_ids"].copy()
            }

        def validate_example(example):
            return (
                isinstance(example.get("input_ids"), list)
                and isinstance(example.get("attention_mask"), list)
                and len(example["input_ids"]) == len(example["attention_mask"])
                and len(example["input_ids"]) <= data_args.max_seq_length
                and len(example["input_ids"]) > 0  # Ensure non-empty examples
            )

        # Map in parallel
        processed_dataset = dataset.map(
            build_prompt_and_tokenize,
            remove_columns=dataset.column_names,
            desc="Tokenizing ABAP code examples",
            num_proc=8
        )

        # Check for empty examples before filtering
        logger.info(f"Dataset size before filtering: {len(processed_dataset)}")
        empty_count = sum(1 for ex in processed_dataset if len(ex.get("input_ids", [])) == 0)
        logger.info(f"Empty examples count: {empty_count}")

        processed_dataset = processed_dataset.filter(validate_example, num_proc=8)
        logger.info(f"Validated dataset size: {len(processed_dataset)}")

        # Peek at a few samples to verify
        sample_size = min(3, len(processed_dataset))
        for i in range(sample_size):
            ex = processed_dataset[i]
            logger.info(f"Sample {i+1}:")
            logger.info(f"Input IDs length = {len(ex['input_ids'])}")
            logger.info(f"Attention Mask length = {len(ex['attention_mask'])}")
            # Log a small part of the text to verify content
            logger.info(f"Text preview: {ex['text'][:100]}...")

        # Train/test split
        splits = processed_dataset.train_test_split(
            test_size=data_args.validation_split,
            shuffle=True,
            seed=training_args.seed
        )
        train_dataset = splits["train"]
        eval_dataset = splits["test"]

        # Apply max_eval_samples if specified
        if data_args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(min(len(eval_dataset), data_args.max_eval_samples)))

        logger.info(f"Train dataset size: {len(train_dataset)}, Eval dataset size: {len(eval_dataset)}")

        # Save dataset to disk for faster loading next time
        try:
            ds_dict = DatasetDict({"train": train_dataset, "test": eval_dataset})
            ds_dict.save_to_disk(cache_dir)
            logger.info(f"Tokenized dataset cached to {cache_dir}")
        except Exception as e:
            logger.warning(f"Error saving dataset to cache: {e}")

        return train_dataset, eval_dataset

    except Exception as e:
        logger.error(f"Error creating datasets: {e}")
        raise

def create_and_prepare_model(args, data_args, training_args):
    """Create and prepare the Llama 3 model with LoRA configuration."""
    
    # Set up the torch dtype
    if hasattr(torch, args.torch_dtype):
        torch_dtype = getattr(torch, args.torch_dtype)
    else:
        torch_dtype = torch.bfloat16
        logger.warning(f"Torch dtype {args.torch_dtype} not found, using bfloat16 instead")
    
    # Set up Flash Attention if requested
    attn_implementation = "flash_attention_2" if args.use_flash_attn else "eager"
    
    # Set up quantization config if requested
    if args.use_4bit_quantization:
        logger.info("Setting up 4-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype
        )
    elif args.use_8bit_quantization:
        logger.info("Setting up 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
    else:
        quantization_config = None
    
    # Load the model
    try:
        # Check if the path is a local directory
        if os.path.isdir(args.model_name_or_path):
            logger.info(f"Loading model from local directory: {args.model_name_or_path}")
            config = AutoConfig.from_pretrained(args.model_name_or_path)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                config=config,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                use_safetensors=True
            )
        else:
            # Handle HuggingFace model ID
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype,
                quantization_config=quantization_config,
                use_safetensors=True
            )
        logger.info(f"Model loaded successfully: {type(model).__name__}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Disable cache for training efficiency
    model.config.use_cache = False

    # Enable gradient checkpointing if requested
    if training_args.gradient_checkpointing:
        logger.info("Enabling gradient checkpointing")
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": getattr(training_args, "use_reentrant", False)}
        )
        
        # Apply PEFT preparation if using quantization
        if args.use_4bit_quantization or args.use_8bit_quantization:
            logger.info("Preparing model for k-bit training")
            model = prepare_model_for_kbit_training(model)

    # Load the tokenizer
    logger.info(f"Loading tokenizer from {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        padding_side="left",  # Better for causal LMs
        truncation_side="right",
        use_fast=True,
    )

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        logger.info("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create PEFT config if requested
    peft_config = None
    if args.use_peft_lora:
        logger.info("Setting up LoRA configuration")
        target_modules = (
            args.lora_target_modules.split(",")
            if args.lora_target_modules != "all-linear"
            else args.lora_target_modules
        )
        
        logger.info(f"LoRA target modules: {target_modules}")
        logger.info(f"LoRA parameters: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        
        peft_config = LoraConfig(
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            r=args.lora_r,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules
        )

    return model, peft_config, tokenizer