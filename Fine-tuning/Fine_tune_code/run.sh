#!/bin/bash

# Create the necessary directories
mkdir -p /tmp/llama3-sft-lora-ds

# Run the DeepSpeed training command
deepspeed --master_port=29503 --num_gpus=2 train_llama3_lora.py \
  --seed 200 \
  --model_name_or_path "/workspace/llama3-pt-model" \
  --dataset_name "/workspace/aligned_dataset1.jsonl" \
  --template_type "llama" \
  --splits "train" \
  --max_seq_length 2048 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --learning_rate 2e-5 \
  --max_train_samples 30000 \
  --max_eval_samples 1000 \
  --weight_decay 1e-4 \
  --warmup_ratio 0.15 \
  --lr_scheduler_type "cosine" \
  --lora_r 32 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --output_dir "/tmp/llama3-sft-lora-ds-new" \
  --use_flash_attn False \
  --use_peft_lora True \
  --use_4bit_quantization True \
  --gradient_checkpointing True \
  --bf16 True \
  --validation_split 0.1 \
  --max_grad_norm 1.0 \
  --save_steps 50 \
  --save_total_limit 5 \
  --logging_steps 25 \
  --eval_steps 250 \
  --eval_strategy "steps" \
  --torch_dtype "bfloat16" \
  --use_reentrant False \
  --dataset_text_field "text" \
  --packing False \
  --add_special_tokens True \
  --deepspeed /workspace/fine_tune_llama_v2/deepspeed_config_z3_qlora.yaml