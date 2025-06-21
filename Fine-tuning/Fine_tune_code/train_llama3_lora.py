# train_llama3_lora.py
import os
import json
import logging
import warnings
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import time
from datetime import datetime
from pathlib import Path

from dataclasses import dataclass, field
from typing import Optional, Union, List, Dict
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    set_seed,
    DataCollatorForLanguageModeling,
    TrainerCallback,
    TrainerControl,
    TrainerState
)
from trl import SFTTrainer
from peft import get_peft_model, LoraConfig

from utils_llama3 import create_and_prepare_model, create_datasets

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# 1) Extend TrainingArguments to include "resume_from_checkpoint".
# ------------------------------------------------------------------------------
@dataclass
class ExtendedTrainingArguments(TrainingArguments):
    """
    Extends HF's TrainingArguments with an optional resume_from_checkpoint field.
    - If None, training starts fresh.
    - If True, auto-detect last checkpoint in output_dir.
    - If it's a string path, tries to resume from that specific checkpoint folder.
    """
    resume_from_checkpoint: Union[bool, str, None] = field(
        default=None,
        metadata={"help": "Whether or from which checkpoint to resume. If True, picks the last checkpoint in output_dir."}
    )
    use_reentrant: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use reentrant activation checkpointing"}
    )
    eval_strategy: Optional[str] = field(
        default="no",
        metadata={"help": "Evaluation strategy to adopt during training"}
    )
    eval_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Number of update steps between two evaluations"}
    )

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    template_type: Optional[str] = field(default="llama")
    lora_alpha: Optional[int] = field(default=64)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=32)
    lora_target_modules: Optional[str] = field(
        default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj,lm_head"
    )
    use_flash_attn: Optional[bool] = field(default=False)
    use_peft_lora: Optional[bool] = field(default=True)
    use_8bit_quantization: Optional[bool] = field(default=False)
    use_4bit_quantization: Optional[bool] = field(default=False)
    torch_dtype: Optional[str] = field(default="bfloat16")

@dataclass
class DataTrainingArguments:
    dataset_name: Optional[str] = field(default="dataset.jsonl")
    dataset_text_field: str = field(default="text")  # SFTTrainer uses this
    max_seq_length: Optional[int] = field(default=4096)
    validation_split: Optional[float] = field(default=0.1)
    splits: Optional[str] = field(default="train,test")
    max_eval_samples: Optional[int] = field(default=None)
    max_train_samples: Optional[int] = field(default=None)
    packing: Optional[bool] = field(default=False)
    add_special_tokens: Optional[bool] = field(default=True)
    append_concat_token: Optional[bool] = field(default=False)

class MyDataCollator:
    """
    Customized data collator that removes 'text' from the batch so we don't get a "string to tensor" error,
    but we keep 'text' in the dataset so SFTTrainer's final mapping doesn't fail.
    """
    def __init__(self, tokenizer):
        self._tokenizer = tokenizer
        self._default_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )

    def __call__(self, features):
        for f in features:
            f.pop("text", None)  # remove text
        return self._default_collator(features)

class LossTracker:
    """
    Tracks training metrics and saves them to CSV and plots.
    """
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.metrics_dir = self.output_dir / "metrics"
        self.plots_dir = self.output_dir / "plots"
        
        # Create directories
        self.metrics_dir.mkdir(exist_ok=True, parents=True)
        self.plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize metrics storage
        self.metrics = {
            'step': [],
            'loss': [],
            'eval_loss': [],
            'learning_rate': [],
            'epoch': [],
            'grad_norm': [],
            'mean_token_accuracy': [],
            'eval_mean_token_accuracy': [],
            'num_tokens': [],
            'timestamp': []
        }
        
        # Initialize CSV file
        self.csv_path = self.metrics_dir / "training_metrics.csv"
        self.json_path = self.metrics_dir / "training_metrics.json"
        
        # Log initialization
        logger.info(f"Initialized LossTracker. Metrics will be saved to {self.csv_path}")
        
        # Track step history to avoid duplicates
        self.step_history = set()
        
        # Track last save time to avoid excessive file operations
        self.last_save_time = time.time()

    def add_metrics(self, metrics, step):
        """Add metrics for a training step with improved tracking"""
        # Add current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Print what we're receiving with unique marker for clearer logs
        print(f"LossTracker.add_metrics - Step {step}: {metrics}")
        
        # Check if it's a new step or we're updating an existing one
        if step not in self.metrics['step']:
            # New step - add to history
            self.step_history.add(step)
            self.metrics['step'].append(step)
            self.metrics['timestamp'].append(timestamp)
            
            # Initialize all metrics for this step with None
            for key in self.metrics.keys():
                if key not in ['step', 'timestamp']:
                    self.metrics[key].append(None)
        else:
            # Existing step - update timestamp
            idx = self.metrics['step'].index(step)
            self.metrics['timestamp'][idx] = timestamp
        
        # Find the index for this step
        idx = self.metrics['step'].index(step)
        
        # Update metrics for this step
        for key, value in metrics.items():
            if key not in ['step', 'timestamp']:
                # Add new metric types if they don't exist
                if key not in self.metrics:
                    self.metrics[key] = [None] * len(self.metrics['step'])
                
                # Update the value - prefer non-None values
                if idx < len(self.metrics[key]):
                    # Only update if current value is None or if new value is not None
                    if self.metrics[key][idx] is None or value is not None:
                        self.metrics[key][idx] = value
                else:
                    # Extend list if needed
                    while len(self.metrics[key]) <= idx:
                        self.metrics[key].append(None)
                    self.metrics[key][idx] = value
        
        # Save to disk periodically to avoid excessive I/O
        current_time = time.time()
        if current_time - self.last_save_time > 10:  # Save every 10 seconds
            self.save_metrics()
            self.create_plots()
            self.last_save_time = current_time
            
    def save_metrics(self):
        """Save metrics to CSV and JSON"""
        try:
            # Create DataFrame and save to CSV
            df = pd.DataFrame(self.metrics)
            df.to_csv(self.csv_path, index=False)
            
            # Save to JSON as well
            with open(self.json_path, 'w') as f:
                json.dump(self.metrics, f, indent=2)
                
            logger.info(f"Saved metrics to {self.csv_path} and {self.json_path}")
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")

    def create_plots(self):
        """Create training plots"""
        try:
            # Create timestamp for filenames
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Plot loss (training and validation)
            self._create_loss_comparison_plot(timestamp)
            
            # Plot learning rate
            self._create_lr_plot(timestamp)
            
            # Plot token accuracy (training and validation)
            self._create_accuracy_comparison_plot(timestamp)
            
            # Plot gradient norm
            self._create_grad_norm_plot(timestamp)
            
            # Create combined plot
            self._create_combined_plot(timestamp)
            
            logger.info(f"Created plots in {self.plots_dir}")
        except Exception as e:
            logger.error(f"Error creating plots: {e}")

    def _create_loss_comparison_plot(self, timestamp):
        """Create loss plot comparing training and validation"""
        if not self.metrics['loss'] or len(self.metrics['loss']) < 2:
            return
            
        plt.figure(figsize=(10, 6))
        
        # Plot training loss
        valid_indices = [i for i, x in enumerate(self.metrics['loss']) if x is not None]
        if valid_indices:
            plt.plot(
                [self.metrics['step'][i] for i in valid_indices],
                [self.metrics['loss'][i] for i in valid_indices],
                'b-', label='Training Loss'
            )
        
        # Plot validation loss if available
        eval_valid_indices = [i for i, x in enumerate(self.metrics['eval_loss']) if x is not None]
        if eval_valid_indices:
            plt.plot(
                [self.metrics['step'][i] for i in eval_valid_indices],
                [self.metrics['eval_loss'][i] for i in eval_valid_indices],
                'r-', label='Validation Loss'
            )
            
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training vs Validation Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save both timestamped and latest version
        plt.savefig(self.plots_dir / f"loss_comparison_{timestamp}.png")
        plt.savefig(self.plots_dir / "loss_comparison_latest.png")
        plt.close()

    def _create_lr_plot(self, timestamp):
        """Create learning rate plot"""
        valid_indices = [i for i, x in enumerate(self.metrics['learning_rate']) if x is not None]
        if not valid_indices:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(
            [self.metrics['step'][i] for i in valid_indices],
            [self.metrics['learning_rate'][i] for i in valid_indices],
            'g-'
        )
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        
        # Save both timestamped and latest version
        plt.savefig(self.plots_dir / f"lr_plot_{timestamp}.png")
        plt.savefig(self.plots_dir / "lr_plot_latest.png")
        plt.close()
        
    def _create_accuracy_comparison_plot(self, timestamp):
        """Create token accuracy plot comparing training and validation"""
        train_valid_indices = [i for i, x in enumerate(self.metrics['mean_token_accuracy']) if x is not None]
        eval_valid_indices = [i for i, x in enumerate(self.metrics['eval_mean_token_accuracy']) if x is not None]
        
        if not train_valid_indices and not eval_valid_indices:
            return
            
        plt.figure(figsize=(10, 6))
        
        # Plot training accuracy
        if train_valid_indices:
            plt.plot(
                [self.metrics['step'][i] for i in train_valid_indices],
                [self.metrics['mean_token_accuracy'][i] for i in train_valid_indices],
                'b-', label='Training Accuracy'
            )
        
        # Plot validation accuracy
        if eval_valid_indices:
            plt.plot(
                [self.metrics['step'][i] for i in eval_valid_indices],
                [self.metrics['eval_mean_token_accuracy'][i] for i in eval_valid_indices],
                'r-', label='Validation Accuracy'
            )
            
        plt.xlabel('Steps')
        plt.ylabel('Token Accuracy')
        plt.title('Training vs Validation Token Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save both timestamped and latest version
        plt.savefig(self.plots_dir / f"accuracy_comparison_{timestamp}.png")
        plt.savefig(self.plots_dir / "accuracy_comparison_latest.png")
        plt.close()

    def _create_grad_norm_plot(self, timestamp):
        """Create gradient norm plot"""
        valid_indices = [i for i, x in enumerate(self.metrics['grad_norm']) if x is not None]
        if not valid_indices:
            return
            
        plt.figure(figsize=(10, 6))
        plt.plot(
            [self.metrics['step'][i] for i in valid_indices],
            [self.metrics['grad_norm'][i] for i in valid_indices],
            'm-'
        )
        plt.xlabel('Steps')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm During Training')
        plt.grid(True, alpha=0.3)
        
        # Save both timestamped and latest version
        plt.savefig(self.plots_dir / f"grad_norm_plot_{timestamp}.png")
        plt.savefig(self.plots_dir / "grad_norm_plot_latest.png")
        plt.close()

    def _create_combined_plot(self, timestamp):
        """Create combined metrics plot with multiple y-axes"""
        if len(self.metrics['step']) < 2:
            return

        # Create figure with primary y-axis
        fig, ax1 = plt.subplots(figsize=(12, 8))
        
        # Plot loss on primary y-axis
        color = 'tab:blue'
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss', color=color)
        
        # Plot training loss
        train_valid_indices = [i for i, x in enumerate(self.metrics['loss']) if x is not None]
        if train_valid_indices:
            ax1.plot(
                [self.metrics['step'][i] for i in train_valid_indices],
                [self.metrics['loss'][i] for i in train_valid_indices],
                color=color, label='Train Loss'
            )
        
        # Plot validation loss
        eval_valid_indices = [i for i, x in enumerate(self.metrics['eval_loss']) if x is not None]
        if eval_valid_indices:
            ax1.plot(
                [self.metrics['step'][i] for i in eval_valid_indices],
                [self.metrics['eval_loss'][i] for i in eval_valid_indices],
                color='tab:orange', label='Val Loss'
            )
            
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Create a secondary y-axis for accuracy
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Accuracy', color=color)
        
        # Plot training accuracy
        acc_valid_indices = [i for i, x in enumerate(self.metrics['mean_token_accuracy']) if x is not None]
        if acc_valid_indices:
            ax2.plot(
                [self.metrics['step'][i] for i in acc_valid_indices],
                [self.metrics['mean_token_accuracy'][i] for i in acc_valid_indices],
                color=color, label='Train Acc'
            )
            
        # Plot validation accuracy
        eval_acc_valid_indices = [i for i, x in enumerate(self.metrics['eval_mean_token_accuracy']) if x is not None]
        if eval_acc_valid_indices:
            ax2.plot(
                [self.metrics['step'][i] for i in eval_acc_valid_indices],
                [self.metrics['eval_mean_token_accuracy'][i] for i in eval_acc_valid_indices],
                color='tab:purple', label='Val Acc'
            )
            
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Add third y-axis for gradient norm
        ax3 = ax1.twinx()
        # Offset the third y-axis to the right
        ax3.spines["right"].set_position(("axes", 1.1))
        color = 'tab:green'
        ax3.set_ylabel('Gradient Norm', color=color)
        
        # Plot gradient norm
        grad_valid_indices = [i for i, x in enumerate(self.metrics['grad_norm']) if x is not None]
        if grad_valid_indices:
            ax3.plot(
                [self.metrics['step'][i] for i in grad_valid_indices],
                [self.metrics['grad_norm'][i] for i in grad_valid_indices],
                color=color, linestyle='--', label='Grad Norm'
            )
            
        ax3.tick_params(axis='y', labelcolor=color)
        
        # Add title and grid
        plt.title('Training and Validation Metrics')
        ax1.grid(True, alpha=0.3)
        
        # Add a legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper center')
        
        plt.tight_layout()
        
        # Save both timestamped and latest version
        plt.savefig(self.plots_dir / f"combined_metrics_{timestamp}.png")
        plt.savefig(self.plots_dir / "combined_metrics_latest.png")
        plt.close()

# Define the metrics logging callback
class MetricsLoggingCallback(TrainerCallback):
    """Forces metric logging at regular intervals during training"""
    
    def __init__(self, log_every_n_steps=5):
        self.log_every_n_steps = log_every_n_steps
        
    def on_step_end(self, args, state, control, **kwargs):
        """Called after each step"""
        if state.global_step % self.log_every_n_steps == 0:
            trainer = kwargs.get('trainer', None)
            if trainer and hasattr(trainer, 'loss_tracker'):
                # Get current learning rate
                lr = None
                if hasattr(trainer, 'optimizer') and trainer.optimizer:
                    for param_group in trainer.optimizer.param_groups:
                        lr = param_group['lr']
                        break
                
                # Get current loss
                loss = None
                if hasattr(trainer, 'state') and hasattr(trainer.state, 'log_history') and trainer.state.log_history:
                    for entry in reversed(trainer.state.log_history):
                        if 'loss' in entry:
                            loss = entry['loss']
                            break
                
                # Create metrics dict
                metrics = {
                    'learning_rate': lr,
                    'loss': loss,
                    'step': state.global_step,
                    'epoch': state.epoch
                }
                
                # Log forced metrics
                print(f"[CALLBACK] Logging metrics at step {state.global_step}: {metrics}")
                trainer.loss_tracker.add_metrics(metrics, state.global_step)

class CustomSFTTrainer(SFTTrainer):
    def __init__(self, model, args, train_dataset, eval_dataset=None, processing_class=None, peft_config=None, **kwargs):
        # Initialize loss tracker
        self.loss_tracker = LossTracker(args.output_dir)
        self.eval_step_interval = getattr(args, 'eval_steps', 500)
        
        # Extract potentially problematic parameters from kwargs
        max_seq_length = kwargs.pop('max_seq_length', None)
        
        # Instead of passing args directly, create a compatible config
        from trl import SFTConfig
        
        # Extract known SFTConfig parameters from args
        sft_params = {}
        for param in SFTConfig.__init__.__code__.co_varnames:
            if hasattr(args, param):
                sft_params[param] = getattr(args, param)
        
        # Add critical parameters if missing
        if 'output_dir' not in sft_params and hasattr(args, 'output_dir'):
            sft_params['output_dir'] = args.output_dir
        
        # Store data_args parameters in sft_params if they are supported
        if hasattr(args, '_data_args'):
            data_args = args._data_args
            # Try to add dataset_text_field if supported
            if hasattr(data_args, 'dataset_text_field'):
                try:
                    # Check if this parameter is supported
                    dummy_config = SFTConfig()
                    if hasattr(dummy_config, 'dataset_text_field'):
                        sft_params['dataset_text_field'] = data_args.dataset_text_field
                except:
                    pass
            
            # Try to add packing if supported
            if hasattr(data_args, 'packing'):
                try:
                    # Check if this parameter is supported
                    dummy_config = SFTConfig()
                    if hasattr(dummy_config, 'packing'):
                        sft_params['packing'] = data_args.packing
                except:
                    pass
        
        # Add max_seq_length if supported and provided
        if max_seq_length is not None:
            try:
                dummy_config = SFTConfig()
                if hasattr(dummy_config, 'max_seq_length'):
                    sft_params['max_seq_length'] = max_seq_length
                elif hasattr(dummy_config, 'max_length'):
                    sft_params['max_length'] = max_seq_length
            except:
                pass
        
        # Create SFTConfig
        sft_args = SFTConfig(**sft_params)
        
        # Use SFTTrainer initialization with compatible args
        super().__init__(
            model=model,
            args=sft_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            peft_config=peft_config,
            **kwargs
        )
        
        # Disable caching for training efficiency
        self.model.config.use_cache = False
        
        # Use our custom collator
        if processing_class is not None:
            self.data_collator = MyDataCollator(processing_class)
            
        # Add the metrics logging callback to force regular updates
        self.add_callback(MetricsLoggingCallback(log_every_n_steps=5))
        
    def force_log_metrics(self):
        """Force capture and log metrics directly from trainer state"""
        if not hasattr(self, 'state'):
            print("No trainer state available for metrics capture")
            return
            
        step = self.state.global_step
        metrics = {}
        
        # Get metrics from state
        if hasattr(self.state, 'log_history') and self.state.log_history:
            # Collect unique metrics from all log entries
            for log_entry in self.state.log_history:
                for key, value in log_entry.items():
                    if isinstance(value, (int, float)) or value is None:
                        # Don't overwrite metrics we already have unless this is newer
                        if key not in metrics or (
                            'step' in log_entry and 
                            log_entry.get('step', 0) > metrics.get('_step_for_' + key, 0)
                        ):
                            metrics[key] = value
                            # Track which step this metric came from
                            metrics['_step_for_' + key] = log_entry.get('step', 0)
        
        # Remove internal tracking keys
        clean_metrics = {k: v for k, v in metrics.items() if not k.startswith('_step_for_')}
        
        # Add current loss if available
        if hasattr(self, 'state') and hasattr(self.state, 'global_step'):
            if hasattr(self, 'train_loss'):
                clean_metrics['loss'] = self.train_loss
        
        # Directly capture learning rate
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            for param_group in self.optimizer.param_groups:
                clean_metrics['learning_rate'] = param_group['lr']
                break
        
        print(f"[Step {step}] Force logging metrics: {clean_metrics}")
        
        # Add to our tracker
        if clean_metrics:
            self.loss_tracker.add_metrics(clean_metrics, step)    
            
    # Fix: The log method needs to accept start_time parameter
    def log(self, logs, start_time=None):
        """Override log method to track metrics"""
        # To prevent recursion
        if hasattr(self, "_in_log_call") and self._in_log_call:
            super().log(logs, start_time)
            return
        
        # Set flag to prevent recursion
        self._in_log_call = True
        
        try:
            # Call the parent class log method with both parameters
            super().log(logs, start_time)
            
            # Extract and track metrics
            if logs:
                # Get current step
                step = self.state.global_step
                
                # Ensure all metric names are consistent
                clean_logs = {}
                
                # Record ALL metrics regardless of naming convention
                for key, value in logs.items():
                    if isinstance(value, (int, float)) or value is None:
                        if key.startswith('eval/'):
                            clean_key = 'eval_' + key.split('/', 1)[1]
                            clean_logs[clean_key] = value
                        else:
                            clean_logs[key] = value
                
                # Add values from trainer state if available
                if hasattr(self, 'state') and hasattr(self.state, 'log_history') and self.state.log_history:
                    for log_entry in self.state.log_history[-3:]:  # Check last few entries
                        for key, value in log_entry.items():
                            if isinstance(value, (int, float)) or value is None:
                                if key not in clean_logs:
                                    clean_logs[key] = value
                
                # Print detailed debug info
                print(f"[Step {step}] Captured metrics: {clean_logs}")
                
                # Add metrics to tracker
                self.loss_tracker.add_metrics(clean_logs, step)
        finally:
            # Reset flag
            self._in_log_call = False

    def print_trainable_parameters(self):
        """Print the number of trainable parameters in the model."""
        trainable_params = 0
        all_params = 0
        
        for _, param in self.model.named_parameters():
            all_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                
        logger.info(
            f"Trainable params: {trainable_params:,d} || "
            f"All params: {all_params:,d} || "
            f"Trainable%: {100 * trainable_params / all_params:.2f}%"
        )
            
    def save_model(self, output_dir=None, _internal_call=False):
        """Override save_model to also save metrics and plots"""
        # First, call the parent method to save the model
        super().save_model(output_dir, _internal_call)
        
        # Then, save metrics and create plots
        if hasattr(self, 'loss_tracker'):
            self.loss_tracker.save_metrics()
            self.loss_tracker.create_plots()
            
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        Overridden evaluate method that properly handles the ignore_keys parameter.
        This fixes compatibility with the parent Trainer class.
        """
        # Temporarily disable progress bar just for evaluation
        old_disable_tqdm = self.args.disable_tqdm
        self.args.disable_tqdm = True
        
        # Run evaluation with the proper parameters
        result = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix
        )
        
        # Restore original setting
        self.args.disable_tqdm = old_disable_tqdm
        
        return result

def create_direct_plots(output_dir, train_metrics, eval_metrics):
    """Create direct plots from final metrics"""
    try:
        plots_dir = os.path.join(output_dir, "direct_plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        # Plot training metrics
        if train_metrics:
            plt.figure(figsize=(10, 6))
            
            # Extract all numeric metrics
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)) and key not in ['epoch', 'step']:
                    plt.bar(key, value)
            
            plt.title('Training Metrics')
            plt.ylabel('Value')
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(os.path.join(plots_dir, "train_metrics.png"))
            plt.close()
        
        # Plot eval metrics if available
        if eval_metrics:
            plt.figure(figsize=(10, 6))
            
            # Extract all numeric metrics
            for key, value in eval_metrics.items():
                if isinstance(value, (int, float)) and key not in ['epoch', 'step']:
                    plt.bar(key, value)
            
            plt.title('Evaluation Metrics')
            plt.ylabel('Value')
            plt.grid(axis='y', alpha=0.3)
            plt.savefig(os.path.join(plots_dir, "eval_metrics.png"))
            plt.close()
            
        # Plot comparison if both available
        if train_metrics and eval_metrics:
            # Create a comparison plot for loss
            if 'loss' in train_metrics and 'eval_loss' in eval_metrics:
                plt.figure(figsize=(8, 6))
                plt.bar(['Training Loss', 'Evaluation Loss'], 
                        [train_metrics['loss'], eval_metrics['eval_loss']])
                plt.title('Loss Comparison')
                plt.grid(axis='y', alpha=0.3)
                plt.savefig(os.path.join(plots_dir, "loss_comparison.png"))
                plt.close()
    except Exception as e:
        print(f"Error creating direct plots: {e}")

def main():
    """Main training function"""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, ExtendedTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Create output directory if it doesn't exist
    os.makedirs(training_args.output_dir, exist_ok=True)

    # Set up logging to file
    file_handler = logging.FileHandler(os.path.join(training_args.output_dir, "training.log"))
    file_handler.setLevel(logging.WARN)
    logger.addHandler(file_handler)

    # Log training arguments
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")

    # DEBUG PRINTS
    logger.info("=" * 50)
    logger.info("TRAINING CONFIGURATION SUMMARY:")
    logger.info(f"Batch size per device: {training_args.per_device_train_batch_size}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {training_args.per_device_train_batch_size * torch.cuda.device_count() * training_args.gradient_accumulation_steps}")
    logger.info(f"Logging steps: {training_args.logging_steps}")
    logger.info(f"Evaluation strategy: {training_args.eval_strategy}")
    logger.info(f"Evaluation steps: {training_args.eval_steps}")
    logger.info(f"Save steps: {training_args.save_steps}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Number of epochs: {training_args.num_train_epochs}")
    logger.info(f"Max train samples: {data_args.max_train_samples}")
    logger.info("=" * 50)

    # Ensure training_args has certain settings
    training_args.remove_unused_columns = False
    training_args._data_args = data_args
    
    # Set evaluation steps (default to 5x logging steps)
    if getattr(training_args, 'eval_steps', None) is None:
        training_args.eval_steps = training_args.logging_steps * 5
        logger.info(f"Setting eval_steps to {training_args.eval_steps}")
    
    # Set seed for reproducibility
    set_seed(training_args.seed)

    # Create base model, tokenizer, and peft config
    model, peft_config, tokenizer = create_and_prepare_model(model_args, data_args, training_args)
    
    # Apply LoRA if requested
    if model_args.use_peft_lora and peft_config is not None:
        logger.info("Applying LoRA configuration")
        model = get_peft_model(model, peft_config)
    
    # Create datasets
    train_dataset, eval_dataset = create_datasets(tokenizer, data_args, training_args)

    if len(train_dataset) == 0:
        logger.error("Empty training dataset")
        return

    # Initialize trainer with the original training_args
    trainer = CustomSFTTrainer(
        model=model,
        args=training_args,  # Use the original arguments
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, 
        processing_class=tokenizer,
        peft_config=peft_config,
        max_seq_length=data_args.max_seq_length
    )
    
    # Add custom logging hook to capture intermediate metrics
    class LoggingCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs:
                step = state.global_step
                # Force metrics capture at every logging step
                trainer.loss_tracker.add_metrics(logs, step)
                print(f"[LoggingCallback] Added metrics at step {step}: {logs}")

    # Add the callback to ensure metrics are captured at each logging step
    trainer.add_callback(LoggingCallback())
    
    # Print trainable parameters
    trainer.print_trainable_parameters()

    # Resume from checkpoint if requested
    resume_from_checkpoint = training_args.resume_from_checkpoint
    if resume_from_checkpoint is None:
        logger.info("Starting training from scratch")
        train_result = trainer.train()
    else:
        logger.info(f"Resuming training from {resume_from_checkpoint}")
        train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.force_log_metrics()
    # Save the model
    logger.info("Saving final model")
    trainer.save_model()
    trainer.save_state()

    # Save training results
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    # Run a final evaluation
    if eval_dataset is not None:
        logger.info("Running final evaluation")
        eval_metrics = trainer.evaluate(eval_dataset)
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        trainer.force_log_metrics()
    
    # Final update of plots and metrics
    trainer.loss_tracker.save_metrics()
    trainer.loss_tracker.create_plots()
    create_direct_plots(training_args.output_dir, train_result.metrics, eval_metrics if eval_dataset else None)

    logger.info(f"Training complete! Results and model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()