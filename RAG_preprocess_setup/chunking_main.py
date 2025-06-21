#!/usr/bin/env python3
#chunking_main.py
import os
import argparse
import logging
import pandas as pd
import multiprocessing as mp
from multiprocessing import freeze_support
from dotenv import load_dotenv
from tqdm import tqdm
import json
import time
import sys
import shutil

from instance_manager import (
    setup_logging, 
    distribute_files_to_instances, 
    process_instance_files,
    load_checkpoint,
    save_checkpoint
)

def combine_instance_outputs(output_dir, final_output_file, batch_number=None):
    """Combine outputs from all instances into a single file"""
    instance_files = [f for f in os.listdir(output_dir) if f.startswith("instance_") and f.endswith("_chunks.json")]
    
    if not instance_files:
        logging.warning("No instance output files found to combine")
        return 0
    
    # Create a batch-specific output filename if batch_number is provided
    if batch_number is not None:
        batch_output_file = os.path.join(output_dir, f"all_chunks_{batch_number}.json")
    else:
        batch_output_file = final_output_file
        
    all_dfs = []
    total_chunks = 0
    
    # Use batch loading to reduce memory usage
    for instance_file in instance_files:
        try:
            file_path = os.path.join(output_dir, instance_file)
            # Estimate number of chunks in file by counting lines and dividing by estimated lines per chunk
            file_size = os.path.getsize(file_path)
            chunk_size = 1000  # Process in batches of 1000 chunks
            
            # Use pandas chunking for large files
            if file_size > 100 * 1024 * 1024:  # 100 MB
                chunks_reader = pd.read_json(file_path, orient='records', lines=True, chunksize=chunk_size)
                for chunk_df in chunks_reader:
                    all_dfs.append(chunk_df)
                    total_chunks += len(chunk_df)
                    logging.info(f"Loaded {len(chunk_df)} chunks from {instance_file} (batch)")
            else:
                # For smaller files, load all at once
                instance_df = pd.read_json(file_path, orient='records')
                all_dfs.append(instance_df)
                total_chunks += len(instance_df)
                logging.info(f"Loaded {len(instance_df)} chunks from {instance_file}")
        except Exception as e:
            logging.error(f"Error loading {instance_file}: {e}")
    
    if all_dfs:
        # Combine in batches if there are many chunks
        if len(all_dfs) > 10:
            logging.info(f"Combining {len(all_dfs)} dataframes in batches")
            # Process in batches of 5 dataframes to avoid memory issues
            batch_size = 5
            combined_df = None
            
            for i in range(0, len(all_dfs), batch_size):
                batch = all_dfs[i:i+batch_size]
                batch_df = pd.concat(batch, ignore_index=True)
                
                if combined_df is None:
                    combined_df = batch_df
                else:
                    combined_df = pd.concat([combined_df, batch_df], ignore_index=True)
                    
                logging.info(f"Combined batch {i//batch_size + 1}/{(len(all_dfs)-1)//batch_size + 1}")
        else:
            # Smaller number of dataframes, combine all at once
            combined_df = pd.concat(all_dfs, ignore_index=True)
        
        # Write to JSON file in chunks
        if total_chunks > 50000:
            # Very large output - write in smaller chunks to avoid memory issues
            logging.info(f"Writing {total_chunks} chunks to {batch_output_file} in batches")
            
            with open(batch_output_file, 'w') as f:
                f.write('[\n')  # Start JSON array
                
                chunk_size = 10000
                for i in range(0, len(combined_df), chunk_size):
                    chunk = combined_df.iloc[i:i+chunk_size]
                    chunk_json = chunk.to_json(orient='records', indent=2)
                    # Remove brackets from JSON to allow chunked writing
                    if i == 0:  # First chunk
                        chunk_json = chunk_json.lstrip('[\n') 
                    elif i + chunk_size >= len(combined_df):  # Last chunk
                        chunk_json = chunk_json.rstrip('\n]')
                    else:  # Middle chunks
                        chunk_json = chunk_json.lstrip('[\n').rstrip('\n]')
                    
                    f.write(chunk_json)
                    
                    # Add comma between chunks (except last)
                    if i + chunk_size < len(combined_df):
                        f.write(',\n')
                    
                    logging.info(f"Written batch {i//chunk_size + 1}/{(len(combined_df)-1)//chunk_size + 1}")
                
                f.write('\n]')  # End JSON array
        else:
            # Regular sized output - write at once
            combined_df.to_json(batch_output_file, orient='records', indent=2)
            
        logging.info(f"Combined {total_chunks} chunks into {batch_output_file}")
        
        # Clear instance chunk files after combining to save space
        for instance_file in instance_files:
            try:
                os.remove(os.path.join(output_dir, instance_file))
                logging.info(f"Removed instance file: {instance_file}")
            except Exception as e:
                logging.error(f"Error removing instance file {instance_file}: {e}")
        
        return total_chunks
    
    return 0

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Process ABAP documentation using multiple GPT instances')
    
    # Existing arguments
    parser.add_argument('--doc-dir', type=str, default=r'/home/user/Desktop/DATA/REPORT_from_FORMS_tmp_documentaion123',
                        help='Directory containing documentation (.md) files (default: /root/CLASSES)')
    parser.add_argument('--code-dir', type=str, default=r'/home/user/Desktop/DATA/REPORTS_DEMO_FILES',
                        help='Directory containing ABAP code (.abap) files (default: /root/CLASS_ABAP)')
    parser.add_argument('--output-dir', type=str, default=r'/home/user/Desktop/chunk_output_reports',
                        help='Directory to save chunked output (default: /workspace/code/chunk_output)')
    parser.add_argument('--log-file', type=str, default=r'/home/user/Desktop/chunking_reports.log',
                        help='Path to log file (default: /workspace/code/chunking.log)')
    parser.add_argument('--instances', type=int, default=3,
                        help='Number of GPT instances to use (default: 3)')
    parser.add_argument('--max-chars', type=int, default=4000,  # Increased from 2000
                        help='Maximum characters per chunk (default: 3000)')
    parser.add_argument('--overlap-chars', type=int, default=600,
                        help='Overlap characters between chunks (default: 200)')
    parser.add_argument('--use-llm', action='store_true',
                        help='Use LLM for semantic chunking')
    parser.add_argument('--checkpoint-file', type=str, default=r'/home/user/Desktop/chunking_report_checkpoint.json',
                        help='Path to save/load checkpoint data (default: /workspace/code/chunking_checkpoint.json)')
    parser.add_argument('--fresh-start', action='store_true',
                        help='Ignore existing checkpoint and start fresh')
    
    # New arguments
    parser.add_argument('--use-parallel-sections', action='store_true',
                        help='Use parallel section processing for better performance')
    parser.add_argument('--rate-limit', type=int, default=15,
                        help='Initial rate limit for LLM API calls per minute (default: 15)')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Number of files to process in each batch (default: 1000)')
    parser.add_argument('--continue-batch', type=int, default=None,
                        help='Continue processing from specified batch number')
    
    return parser.parse_args()

def process_batch(batch_number, doc_files, args):
    """Process a specific batch of files"""
    # Create a batch-specific checkpoint file
    batch_checkpoint_file = f"{os.path.splitext(args.checkpoint_file)[0]}_batch_{batch_number}.json"
    
    # Create a fresh checkpoint for this batch if needed
    if not os.path.exists(batch_checkpoint_file) or args.fresh_start:
        # Create a fresh checkpoint for this batch
        checkpoint = {
            "instance_assignments": {},
            "processed_files": [],
            "batch_number": batch_number,
            "batch_total_files": len(doc_files)
        }
        save_checkpoint(checkpoint, batch_checkpoint_file)
        logging.info(f"Created fresh checkpoint for batch {batch_number} with {len(doc_files)} files")
    else:
        # Load existing batch checkpoint
        checkpoint = load_checkpoint(batch_checkpoint_file)
        logging.info(f"Loaded checkpoint for batch {batch_number}: {len(checkpoint.get('processed_files', []))} files already processed")
    
    # Distribute files to instances
    instance_assignments = distribute_files_to_instances(
        doc_files, 
        args.instances, 
        batch_checkpoint_file
    )
    
    # Log assignments
    for instance_id, files in instance_assignments.items():
        logging.info(f"Batch {batch_number} - Instance {instance_id}: {len(files)} files assigned")
    
    # Check if all files in this batch are already processed
    if checkpoint.get("processed_files") and len(checkpoint.get("processed_files")) >= len(doc_files):
        logging.info(f"Batch {batch_number} is already fully processed. Skipping...")
        return True
    
    # Set up deployment configurations
    deployment_configs = []
    
    # Check for deployment ID environment variables (instance-specific)
    for i in range(1, args.instances + 1):
        var_name = f'AICORE_DEPLOYMENT_ID_{i}'
        deployment_id = os.getenv(var_name)
        if deployment_id:
            deployment_configs.append({
                "deployment_id": deployment_id,
                "model_name": "gpt-4o",
                "rate_limit": args.rate_limit
            })
            logging.info(f"Using deployment ID for instance {i-1}: {deployment_id}")
    
    # If no instance-specific deployment IDs found, use default for all instances
    if not deployment_configs and os.getenv('AICORE_DEPLOYMENT_ID'):
        default_id = os.getenv('AICORE_DEPLOYMENT_ID')
        for i in range(args.instances):
            deployment_configs.append({
                "deployment_id": default_id,
                "model_name": "gpt-4o",
                "rate_limit": args.rate_limit
            })
        logging.info(f"Using default deployment ID for all instances: {default_id}")
    
    # Process files in parallel using multiple processes
    start_time = time.time()
    processes = []
    results = []
    
    with mp.Pool(processes=args.instances) as pool:
        # Submit jobs to process instance files
        for i in range(args.instances):
            if args.use_llm and i < len(deployment_configs):
                # Use the corresponding deployment config
                config = deployment_configs[i]
            else:
                config = {"deployment_id": None, "model_name": None}
            
            process = pool.apply_async(
                process_instance_files,
                args=(
                    i,
                    args.doc_dir,
                    args.code_dir,
                    args.output_dir,
                    config,
                    args.max_chars,
                    args.overlap_chars,
                    args.use_llm,
                    batch_checkpoint_file
                )
            )
            processes.append(process)
        
        # Wait for all processes to complete
        for process in processes:
            result = process.get()
            results.append(result)
    
    # Log results
    total_processed = sum(r["processed_count"] for r in results)
    total_errors = sum(r["error_count"] for r in results)
    
    elapsed_time = time.time() - start_time
    logging.info(f"Batch {batch_number} processing completed in {elapsed_time:.2f} seconds")
    logging.info(f"Batch {batch_number} - Total files processed: {total_processed}")
    logging.info(f"Batch {batch_number} - Total errors: {total_errors}")
    
    # Combine outputs from this batch
    batch_output_file = os.path.join(args.output_dir, f"all_chunks_{batch_number}.json")
    total_chunks = combine_instance_outputs(args.output_dir, batch_output_file, batch_number)
    
    logging.info(f"Batch {batch_number} - Combined output saved to {batch_output_file}")
    logging.info(f"Batch {batch_number} - Total chunks generated: {total_chunks}")
    
    # Compute statistics
    try:
        if os.path.exists(batch_output_file):
            combined_df = pd.read_json(batch_output_file, orient='records')
            with_code = 0
            
            for i, row in combined_df.iterrows():
                metadata = json.loads(row['metadata'])
                if metadata.get('code_snippet') and len(metadata['code_snippet'].strip()) > 0:
                    with_code += 1
            
            logging.info(f"Batch {batch_number} - Chunks with code: {with_code}/{total_chunks} ({with_code/total_chunks*100:.1f}%)")
    except Exception as e:
        logging.error(f"Error computing statistics for batch {batch_number}: {e}")
    
    return True

def main():
    # Support for Windows multi-processing
    freeze_support()
    
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logging
    logger = setup_logging(args.log_file)
    
    # Load environment variables from existing chunk.env file
    env_file = 'chunk.env'
    if os.path.exists(env_file):
        load_dotenv(env_file)
        logging.info(f"Loaded environment variables from {env_file}")
    else:
        logging.warning(f"Environment file {env_file} not found, using existing environment variables")
    
    # Verify environment variables for LLM use
    if args.use_llm:
        for key in ['AICORE_AUTH_URL', 'AICORE_BASE_URL', 'AICORE_CLIENT_ID', 
                   'AICORE_CLIENT_SECRET', 'AICORE_RESOURCE_GROUP']:
            if not os.getenv(key):
                logging.error(f"Missing required environment variable: {key}")
                return
    
    # List documentation files
    doc_files = [f for f in os.listdir(args.doc_dir) if f.endswith('.md')]
    if not doc_files:
        logging.error(f"No markdown files found in {args.doc_dir}")
        return
    
    logging.info(f"Found {len(doc_files)} documentation files")
    
    # Determine which batches need to be processed
    start_batch = 1
    if args.continue_batch is not None:
        start_batch = args.continue_batch
        logging.info(f"Continuing from batch {start_batch}")
    
    # Calculate number of batches
    batch_size = args.batch_size
    total_batches = (len(doc_files) + batch_size - 1) // batch_size  # Ceiling division
    
    # Process batches
    for batch_num in range(start_batch, total_batches + 1):
        # Calculate start and end indices for this batch
        start_idx = (batch_num - 1) * batch_size
        end_idx = min(start_idx + batch_size, len(doc_files))
        
        # Get files for this batch
        batch_files = doc_files[start_idx:end_idx]
        
        logging.info(f"===== Starting Batch {batch_num}/{total_batches} with {len(batch_files)} files =====")
        
        # Process this batch
        result = process_batch(batch_num, batch_files, args)
        
        logging.info(f"===== Completed Batch {batch_num}/{total_batches} =====")
        
        # After each batch is complete, save a summary file with batch information
        summary_file = os.path.join(args.output_dir, "batch_summary.json")
        summary = {
            "total_batches": total_batches,
            "batch_size": batch_size,
            "current_batch": batch_num,
            "total_files": len(doc_files),
            "processed_files": batch_num * batch_size if batch_num < total_batches else len(doc_files),
            "remaining_files": len(doc_files) - (batch_num * batch_size if batch_num < total_batches else len(doc_files)),
            "completion_percentage": (batch_num / total_batches) * 100,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logging.info(f"Saved batch summary to {summary_file}")
        logging.info(f"Overall progress: {summary['completion_percentage']:.1f}% complete")
        
        # Optional pause between batches
        if batch_num < total_batches:
            logging.info(f"Waiting 5 seconds before starting next batch...")
            time.sleep(5)
    
    # Final combination of all batches if needed
    logging.info("All batches processed successfully!")
    
    # Success message
    logging.info("===== ABAP Documentation Chunking Completed =====")
    print("Chunking process completed successfully. Check logs for details.")

if __name__ == "__main__":
    print("=== ABAP Documentation Chunker ===")
    print("Starting chunking process with batch processing...")
    main()
    print("Chunking process completed. Check logs for details.")