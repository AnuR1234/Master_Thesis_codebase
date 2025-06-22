#!/usr/bin/env python3
"""
Model Downloader for RAG Pipeline

This script downloads and caches language models locally for use with the RAG generator.
It handles the downloading, verification, and organization of model files.
"""

import os
import logging
import argparse
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the directory where models will be cached locally
# Use an absolute path to make sure the directory is always available
MODELS_DIR = os.path.join(os.path.expanduser("~"), "rag_models")

# Define the default embedding model
DEFAULT_EMBEDDING_MODEL = "intfloat/e5-large-v2"

def setup_model_directory():
    """Create the models directory if it doesn't exist"""
    os.makedirs(MODELS_DIR, exist_ok=True)
    logger.info(f"Model cache directory: {MODELS_DIR}")

def download_model(model_name, force=False, api_token=None):
    """
    Download and cache a model locally
    
    Args:
        model_name: Name of the model to download
        force: If True, redownload even if model exists
        api_token: Hugging Face API token for accessing gated models
        
    Returns:
        Path to the local model directory
    """
    try:
        # Import necessary libraries
        from huggingface_hub import snapshot_download
        from huggingface_hub import login
        
        # If API token is provided, log in with it
        if api_token:
            logger.info(f"Logging in to Hugging Face with provided API token")
            login(token=api_token)
        
        # Create model-specific cache directory
        # Replace slashes in model name with underscores to create a valid directory name
        safe_name = model_name.replace('/', '_')
        model_dir = os.path.join(MODELS_DIR, safe_name)
        
        # Check if model is already downloaded
        if os.path.exists(model_dir) and os.path.isdir(model_dir) and not force:
            logger.info(f"Model {model_name} already exists at {model_dir}")
            # Check if the download is complete by verifying key files
            if os.path.exists(os.path.join(model_dir, "config.json")):
                logger.info(f"Model files appear to be complete")
                return model_dir
            else:
                logger.warning(f"Model directory exists but appears incomplete. Deleting and re-downloading...")
                # Delete the incomplete directory
                shutil.rmtree(model_dir)
        elif force and os.path.exists(model_dir):
            logger.info(f"Force option specified. Removing existing model directory: {model_dir}")
            shutil.rmtree(model_dir)
        
        # If we got here, we need to download the model
        logger.info(f"Downloading model {model_name} to {model_dir}...")
        
        # Create the directory
        os.makedirs(model_dir, exist_ok=True)
        
        # Download model files using snapshot_download with proper parameters for model files
        logger.info("Starting download - this may take a while for large models...")
        snapshot_download(
            repo_id=model_name,
            local_dir=model_dir,
            local_dir_use_symlinks=False,  # Avoid symlinks for better compatibility
            token=api_token,  # Use the provided API token
            resume_download=True,  # Resume interrupted downloads
            max_workers=4,  # Limit number of parallel downloads to avoid timeout issues
            ignore_patterns=["*.md", "*.h5", "*.ot"]  # Exclude some unwanted file types, but keep model weights
        )
        
        logger.info(f"Model {model_name} downloaded successfully to {model_dir}")
        return model_dir
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {e}")
        raise

def download_embedding_model(force=False, api_token=None):
    """
    Download the embedding model
    
    Args:
        force: If True, redownload even if model exists
        api_token: Hugging Face API token for accessing gated models
        
    Returns:
        Path to the local embedding model directory
    """
    model_name = DEFAULT_EMBEDDING_MODEL
    logger.info(f"Downloading embedding model {model_name}...")
    return download_model(model_name, force, api_token)

def verify_model(model_dir):
    """
    Verify that the model directory contains all required files
    
    Args:
        model_dir: Path to the model directory
        
    Returns:
        True if model appears complete, False otherwise
    """
    required_files = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_dir, file)):
            missing_files.append(file)
    
    # Special case for embedding models which might not have tokenizer files
    if "e5-large-v2" in model_dir:
        # For E5 models, we mainly need the model files
        special_files = [f for f in os.listdir(model_dir) if f.endswith('.bin') or f.endswith('.safetensors')]
        if special_files and os.path.exists(os.path.join(model_dir, "config.json")):
            logger.info(f"E5 embedding model verification passed with {len(special_files)} model files")
            return True
    
    if missing_files:
        logger.warning(f"Model appears incomplete. Missing files: {', '.join(missing_files)}")
        return False
    
    # Check for model files with various extensions
    model_files = []
    for ext in [".bin", ".safetensors", ".pt", ".gguf"]:
        model_files.extend([f for f in os.listdir(model_dir) if f.endswith(ext)])
    
    if not model_files:
        logger.warning("No model weight files found. Looking for specific model architecture files...")
        
        # Check for model-specific file patterns
        if any(f.startswith("pytorch_model") for f in os.listdir(model_dir)):
            logger.info("Found pytorch_model files - this appears to be a sharded model")
            return True
            
        if any("model.safetensors" in f for f in os.listdir(model_dir)):
            logger.info("Found model.safetensors files - this appears to be a valid model")
            return True
            
        # For safetensors sharded models, look for numbered safetensors files
        safetensors_files = [f for f in os.listdir(model_dir) if f.endswith('.safetensors')]
        if safetensors_files:
            logger.info(f"Found {len(safetensors_files)} safetensors files - this appears to be a valid model")
            return True
            
        # For mistral models specifically, check for model-specific patterns
        if "mistral" in model_dir.lower():
            if os.path.exists(os.path.join(model_dir, "pytorch_model.bin.index.json")):
                logger.info("Found Mistral model index file")
                return True
            
            # Check if we have any of the expected shard naming patterns
            shard_patterns = ["model-00001-of-", "consolidated.00.safetensors"]
            for pattern in shard_patterns:
                matching_files = [f for f in os.listdir(model_dir) if pattern in f]
                if matching_files:
                    logger.info(f"Found Mistral model shards: {len(matching_files)} files matching {pattern}")
                    return True
        
        # If we get here, we didn't find any expected model weight files
        logger.warning("No model weight files found.")
        logger.warning("This directory may only contain metadata without the actual model weights.")
        return False
    
    logger.info(f"Found {len(model_files)} model weight files: {', '.join(model_files[:3])}{'...' if len(model_files) > 3 else ''}")
    logger.info(f"Model verification passed. Model appears complete.")
    return True

def list_cached_models():
    """List all models currently cached in the models directory"""
    if not os.path.exists(MODELS_DIR):
        logger.info("Models directory does not exist yet.")
        return []
    
    models = []
    for item in os.listdir(MODELS_DIR):
        item_path = os.path.join(MODELS_DIR, item)
        if os.path.isdir(item_path):
            # Check if it's a valid model directory
            if os.path.exists(os.path.join(item_path, "config.json")):
                # Convert directory name back to model name format
                model_name = item.replace('_', '/')
                models.append((model_name, item_path))
    
    return models

def main():
    """Main function to download models"""
    parser = argparse.ArgumentParser(description="Download and cache models for the RAG generator")
    parser.add_argument("--model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3",
                        help="Name of the model to download")
    parser.add_argument("--embedding", action="store_true",
                        help=f"Download the embedding model ({DEFAULT_EMBEDDING_MODEL})")
    parser.add_argument("--force", action="store_true",
                        help="Force redownload even if model exists")
    parser.add_argument("--list", action="store_true",
                        help="List all cached models")
    parser.add_argument("--token", type=str, 
                        help="Hugging Face API token for accessing gated models")
    parser.add_argument("--clean", action="store_true",
                        help="Clean incomplete model downloads before downloading")
    
    args = parser.parse_args()
    
    # Set up the models directory
    setup_model_directory()
    
    if args.list:
        models = list_cached_models()
        if models:
            logger.info("Cached models:")
            for model_name, model_path in models:
                logger.info(f"  - {model_name} ({model_path})")
                # Check if model is valid
                if verify_model(model_path):
                    print(f"  ✓ {model_name} - VALID")
                else:
                    print(f"  ✗ {model_name} - INCOMPLETE")
        else:
            logger.info("No models are currently cached.")
        return
    
    # Get API token from argument or environment
    api_token = args.token
    if not api_token:
        # Try to get from environment variable
        api_token = os.environ.get("HF_TOKEN")
        if api_token:
            logger.info("Using Hugging Face API token from environment variable")
    
    # If still no token and it's a gated repo, prompt user
    if not api_token and ("mistralai" in args.model or "meta-llama" in args.model):
        print("\nThis appears to be a gated model that requires authentication.")
        print("Please provide your Hugging Face API token.")
        api_token = input("Enter HF API token (or press Enter to try without token): ").strip()
    
    try:
        if args.embedding:
            # If embedding option is specified, download the embedding model
            model_dir = download_embedding_model(args.force, api_token)
            if verify_model(model_dir):
                logger.info(f"Embedding model {DEFAULT_EMBEDDING_MODEL} is ready to use at: {model_dir}")
                print(f"\nSUCCESS: Embedding model {DEFAULT_EMBEDDING_MODEL} is ready to use.")
                print(f"Model directory: {model_dir}")
                print("\nYou can now use this embedding model in your RAG generator by specifying this path.")
            else:
                logger.error(f"Embedding model verification failed for {DEFAULT_EMBEDDING_MODEL}")
                print(f"\nWARNING: Embedding model verification failed for {DEFAULT_EMBEDDING_MODEL}.")
                print("The model may be incomplete or corrupted.")
                print("Try running with the --clean and --force options to start fresh:")
                print(f"  python model_downloader.py --embedding --clean --force")
        else:
            # If clean option is specified, remove the model directory if it exists
            if args.clean:
                safe_name = args.model.replace('/', '_')
                model_dir = os.path.join(MODELS_DIR, safe_name)
                if os.path.exists(model_dir):
                    logger.info(f"Cleaning model directory: {model_dir}")
                    try:
                        shutil.rmtree(model_dir)
                        logger.info(f"Successfully removed model directory. Will download fresh copy.")
                    except Exception as e:
                        logger.error(f"Error removing model directory: {e}")
                        print(f"Error cleaning model directory: {e}")
                        return

            # Download the specified model
            model_dir = download_model(args.model, args.force, api_token)
            if verify_model(model_dir):
                logger.info(f"Model {args.model} is ready to use at: {model_dir}")
                print(f"\nSUCCESS: Model {args.model} is ready to use.")
                print(f"Model directory: {model_dir}")
                print("\nYou can now use this model in your RAG generator by specifying this path.")
            else:
                logger.error(f"Model verification failed for {args.model}")
                print(f"\nWARNING: Model verification failed for {args.model}.")
                print("The model may be incomplete or corrupted.")
                print("Try running with the --clean and --force options to start fresh:")
                print(f"  python model_downloader.py --model {args.model} --clean --force --token YOUR_HF_TOKEN")
    except Exception as e:
        logger.error(f"Model download failed: {e}")
        print(f"\nERROR: Model download failed: {e}")
        
        if "401 Client Error" in str(e) and "Access to model" in str(e):
            print("\nAuthentication error: This model requires a valid Hugging Face API token.")
            print("You can run the script again with your token:")
            print(f"  python model_downloader.py --token YOUR_HF_TOKEN")
            print("Or set the HF_TOKEN environment variable before running:")
            print(f"  export HF_TOKEN=YOUR_HF_TOKEN")
            print(f"  python model_downloader.py")

if __name__ == "__main__":
    main()