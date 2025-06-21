#instance_manager.py
import os
import json
import logging
import pandas as pd
import time
from tqdm import tqdm
import multiprocessing as mp
import numpy as np
from doc_chunker import DocumentationChunker
from llm_chunker import SAPLLMMultiDeploymentChunker

def setup_logging(log_file):
    """Configure logging to file and console"""
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler (DEBUG level)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def load_checkpoint(checkpoint_file):
    """Load checkpoint information about processed files"""
    if os.path.exists(checkpoint_file):
        try:
            logging.info(f"Loading checkpoint from: {checkpoint_file}")
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            logging.info(f"Loaded checkpoint with {len(checkpoint_data.get('processed_files', []))} processed files")
            return checkpoint_data
        except Exception as e:
            logging.error(f"Error loading checkpoint from {checkpoint_file}: {e}")
    else:
        logging.info(f"No checkpoint file found at: {checkpoint_file}")
    
    # Return empty checkpoint data if file doesn't exist or has errors
    logging.info("Creating new checkpoint data")
    return {
        "instance_assignments": {},
        "processed_files": []
    }

def save_checkpoint(checkpoint_data, checkpoint_file):
    """Save checkpoint information about processed files"""
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    with open(checkpoint_file, 'w') as f:
        json.dump(checkpoint_data, f, indent=2)

def save_chunks_to_instance_json(df, output_file):
    """Save chunks dataframe to instance-specific JSON file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if os.path.exists(output_file):
        # Append to existing file
        try:
            existing_df = pd.read_json(output_file, orient='records')
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_json(output_file, orient='records', indent=2)
            return combined_df
        except Exception as e:
            logging.error(f"Error appending to {output_file}: {e}")
            # Fallback to overwrite
            df.to_json(output_file, orient='records', indent=2)
            return df
    else:
        # Create new file
        df.to_json(output_file, orient='records', indent=2)
        return df

def distribute_files_to_instances(doc_files, num_instances, checkpoint_file):
    """Distribute files evenly among instances"""
    # Load checkpoint to see if we have existing assignments
    checkpoint = load_checkpoint(checkpoint_file)
    instance_assignments = checkpoint.get("instance_assignments", {})
    processed_files = checkpoint.get("processed_files", [])
    
    logging.info(f"Checkpoint file path: {checkpoint_file}")
    logging.info(f"Found {len(processed_files)} previously processed files in checkpoint")
    
    # Filter out already processed files
    unprocessed_files = [f for f in doc_files if f not in processed_files]
    logging.info(f"Remaining unprocessed files: {len(unprocessed_files)}")
    
    if not unprocessed_files:
        logging.info("All files have been processed.")
        return instance_assignments
    
    # If we already have assignments, use them
    if instance_assignments and all(str(i) in instance_assignments for i in range(num_instances)):
        logging.info(f"Using existing instance assignments from checkpoint with {sum(len(files) for files in instance_assignments.values())} total assigned files")
        # Remove any processed files from assignments
        files_before = sum(len(files) for files in instance_assignments.values())
        for instance_id in instance_assignments:
            instance_assignments[instance_id] = [
                f for f in instance_assignments[instance_id] 
                if f not in processed_files
            ]
        files_after = sum(len(files) for files in instance_assignments.values())
        logging.info(f"Removed {files_before - files_after} already processed files from assignments")
        return instance_assignments
        
    # Create new assignments - distribute files evenly
    logging.info(f"Creating new assignments for {len(unprocessed_files)} files across {num_instances} instances")
    instance_assignments = {str(i): [] for i in range(num_instances)}
    
    # Sort files by estimated size (from filename length as a simple proxy)
    # This helps distribute complex files more evenly
    file_sizes = [(f, os.path.getsize(os.path.join(os.getenv('doc_dir', '.'), f)) if os.path.exists(os.path.join(os.getenv('doc_dir', '.'), f)) else len(f)) 
                 for f in unprocessed_files]
    
    # Sort by descending size to allocate largest files first
    file_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Distribute files using a greedy approach to balance load
    instance_loads = [0] * num_instances
    for file_name, file_size in file_sizes:
        # Find instance with smallest current load
        min_load_instance = instance_loads.index(min(instance_loads))
        # Assign file to this instance
        instance_assignments[str(min_load_instance)].append(file_name)
        # Update the instance's load
        instance_loads[min_load_instance] += file_size
    
    # Log assignments
    for i in range(num_instances):
        logging.info(f"Assigned {len(instance_assignments[str(i)])} files to instance {i} (estimated load: {instance_loads[i]})")
    
    # Save assignments to checkpoint
    checkpoint["instance_assignments"] = instance_assignments
    save_checkpoint(checkpoint, checkpoint_file)
    
    return instance_assignments

def process_instance_files(instance_id, doc_dir, code_dir, output_dir, deployment_config, 
                          max_chars=2000, overlap_chars=200, use_llm=False, checkpoint_file=None):
    """Process all files assigned to a specific instance"""
    # Set up instance-specific logging
    log_file = os.path.join(output_dir, f"instance_{instance_id}.log")
    logger = setup_logging(log_file)
    
    # Load file assignments for this instance
    checkpoint = load_checkpoint(checkpoint_file)
    
    # Get files assigned to this instance
    instance_assignments = checkpoint.get("instance_assignments", {})
    assigned_files = instance_assignments.get(str(instance_id), [])
    processed_files = checkpoint.get("processed_files", [])
    
    if not assigned_files:
        logging.info(f"Instance {instance_id}: No files assigned")
        return {"instance_id": instance_id, "processed_count": 0, "error_count": 0}
    
    # Create the chunker and initialization/connection to LLM if needed
    if use_llm:
        logging.info(f"Instance {instance_id}: Initializing with deployment {deployment_config['deployment_id']}")
        llm_chunker = SAPLLMMultiDeploymentChunker(
            deployment_configs=[deployment_config],
            timeout=120
        )
        doc_chunker = DocumentationChunker(
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            llm_chunkers=llm_chunker,
            use_llm=True
        )
    else:
        doc_chunker = DocumentationChunker(
            max_chars=max_chars,
            overlap_chars=overlap_chars,
            use_llm=False
        )
    
    # Set up instance-specific output file
    instance_output_file = os.path.join(output_dir, f"instance_{instance_id}_chunks.json")
    
    # Track statistics
    processed_count = 0
    error_count = 0
    
    # Process files with improved performance tracking
    start_time = time.time()
    total_files = len(assigned_files)
    batch_save_interval = 5  # Save results every 5 files
    
    accumulated_chunks_df = pd.DataFrame()
    
    for file_index, doc_file in enumerate(assigned_files):
        # Calculate processing rate and ETA
        if file_index > 0:
            elapsed = time.time() - start_time
            files_per_second = file_index / elapsed
            files_remaining = total_files - file_index
            eta_seconds = files_remaining / files_per_second if files_per_second > 0 else 0
            
            # Format ETA nicely
            eta_hours = int(eta_seconds / 3600)
            eta_minutes = int((eta_seconds % 3600) / 60)
            eta_str = f"{eta_hours}h {eta_minutes}m"
            
            logging.info(f"Instance {instance_id}: Processing {file_index}/{total_files} files " +
                         f"({files_per_second:.4f} files/sec, ETA: {eta_str})")
        
        # Skip if already processed
        if doc_file in processed_files:
            logging.info(f"Instance {instance_id}: Skipping already processed file {doc_file}")
            continue
            
        try:
            # Get corresponding code file
            base_name = os.path.splitext(doc_file)[0]
            code_file = f"{base_name}.abap"
            code_path = os.path.join(code_dir, code_file)
            doc_path = os.path.join(doc_dir, doc_file)
            
            # Check if files exist
            if not os.path.exists(doc_path):
                logging.error(f"Instance {instance_id}: Doc file not found: {doc_path}")
                error_count += 1
                continue
                
            if not os.path.exists(code_path):
                logging.error(f"Instance {instance_id}: Code file not found: {code_path}")
                error_count += 1
                continue
            
            # Process the file pair - using the improved parallel section processor
            deployment_id = deployment_config.get('deployment_id') if use_llm else None
            file_start = time.time()
            
            # Use the parallel section processing method for better performance
            chunks_df = doc_chunker.process_file_parallel_sections(
                doc_path, 
                code_path, 
                deployment_id=deployment_id,
                instance_id=instance_id
            )
            
            file_elapsed = time.time() - file_start
            
            if not chunks_df.empty:
                # Accumulate chunks
                if accumulated_chunks_df.empty:
                    accumulated_chunks_df = chunks_df
                else:
                    accumulated_chunks_df = pd.concat([accumulated_chunks_df, chunks_df], ignore_index=True)
                    
                # Save chunks periodically to avoid memory issues
                if file_index % batch_save_interval == 0 or file_index == total_files - 1:
                    if not accumulated_chunks_df.empty:
                        save_chunks_to_instance_json(accumulated_chunks_df, instance_output_file)
                        logging.info(f"Instance {instance_id}: Saved batch of {len(accumulated_chunks_df)} chunks from {file_index-batch_save_interval+1}-{file_index+1}")
                        accumulated_chunks_df = pd.DataFrame()  # Reset accumulated chunks
                
                processed_count += 1
                
                # Update checkpoint
                processed_files.append(doc_file)
                checkpoint["processed_files"] = processed_files
                save_checkpoint(checkpoint, checkpoint_file)
                
                logging.info(f"Instance {instance_id}: Processed {doc_file} in {file_elapsed:.2f}s - generated {len(chunks_df)} chunks")
            else:
                logging.warning(f"Instance {instance_id}: No chunks generated for {doc_file}")
                error_count += 1
                
        except Exception as e:
            logging.error(f"Instance {instance_id}: Error processing {doc_file}: {e}")
            error_count += 1
    
    # Save any remaining accumulated chunks
    if not accumulated_chunks_df.empty:
        save_chunks_to_instance_json(accumulated_chunks_df, instance_output_file)
        logging.info(f"Instance {instance_id}: Saved final batch of {len(accumulated_chunks_df)} chunks")
    
    # Cleanup resources
    if use_llm and hasattr(llm_chunker, 'close'):
        try:
            llm_chunker.close()
        except:
            pass
            
    # Summary stats
    total_elapsed = time.time() - start_time
    avg_time_per_file = total_elapsed / max(processed_count, 1)
    logging.info(f"Instance {instance_id}: Completed processing {processed_count} files in {total_elapsed:.2f}s")
    logging.info(f"Instance {instance_id}: Average time per file: {avg_time_per_file:.2f}s")
    
    return {
        "instance_id": instance_id, 
        "processed_count": processed_count, 
        "error_count": error_count
    }