# -*- coding: utf-8 -*-
"""
Memory manager optimized for RTX 6000 Ada (48GB VRAM).

This module provides comprehensive memory management capabilities specifically
optimized for the NVIDIA RTX 6000 Ada GPU with 48GB VRAM. It handles process
conflicts, automatic memory optimization, continuous monitoring, and provides
context managers for safe memory operations.

Key Features:
    - Real-time GPU memory monitoring and reporting
    - Automatic detection and termination of competing processes
    - Memory threshold management with configurable alerts
    - Context managers for safe memory operations
    - Optimized model loading configurations
    - Background monitoring with automatic conflict resolution
    - CUDA out-of-memory error recovery

Classes:
    RTX6000MemoryManager: Main memory management class for RTX 6000 Ada

Functions:
    check_memory_conflicts: Check and resolve memory conflicts
    kill_competing_processes: Kill competing GPU processes
    managed_memory_context: Get managed memory context manager
    optimize_for_models: Optimize memory for specific model loading
    get_model_loading_config: Get optimized model loading configuration
    start_memory_monitoring: Start background memory monitoring
    log_memory_status: Log detailed memory status

Global Variables:
    rtx6000_memory_manager: Global instance for application-wide use
    logger: Configured logger for memory management output

Hardware Requirements:
    - NVIDIA RTX 6000 Ada (48GB VRAM) - optimal
    - CUDA-compatible GPU with 12GB+ VRAM - minimum
    - nvidia-smi utility for process monitoring
    - CUDA toolkit and drivers

Dependencies:
    - torch: PyTorch for CUDA memory management
    - psutil: System and process utilities
    - subprocess: Process execution and control
    - threading: Background monitoring support

Usage:
    >>> from rtx6000_memory_manager import managed_memory_context
    >>> with managed_memory_context():
    ...     # Safe memory operations here
    ...     model = load_large_model()

    >>> from rtx6000_memory_manager import optimize_for_models
    >>> optimize_for_models(["mistral-7b-instruct", "e5-large-v2"])
"""
import os
import gc
import logging
import time
import subprocess
import threading
from contextlib import contextmanager
from typing import Optional, Any, Dict, List, Union
import psutil

logger = logging.getLogger(__name__)

# Set optimized CUDA environment variables for RTX 6000 Ada
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:2048,expandable_segments:True,garbage_collection_threshold:0.8"


class RTX6000MemoryManager:
    """
    Memory manager optimized for RTX 6000 Ada with 48GB VRAM.
    
    This class provides comprehensive memory management for the RTX 6000 Ada GPU,
    including automatic process conflict resolution, memory threshold monitoring,
    and optimized configurations for large model loading and inference.
    
    Attributes:
        torch: PyTorch module (lazy-loaded)
        total_vram_gb (float): Total VRAM capacity in GB (48GB for RTX 6000 Ada)
        critical_threshold_gb (float): Critical memory threshold (2GB remaining)
        warning_threshold_gb (float): Warning memory threshold (5GB remaining)
        safe_threshold_gb (float): Safe operation threshold (10GB remaining)
        last_cleanup (float): Timestamp of last cleanup operation
        competing_processes (List): List of detected competing processes
        models_loaded (bool): Flag indicating if models are currently loaded
        monitor_interval (int): Background monitoring interval in seconds
        auto_kill_competing (bool): Whether to automatically kill competing processes
        
    Example:
        >>> manager = RTX6000MemoryManager()
        >>> status = manager.get_gpu_memory_status()
        >>> print(f"Free VRAM: {status['free']:.1f}GB")
        
        >>> with manager.managed_memory_context():
        ...     # Load and use models safely
        ...     model = load_large_model()
    """
    
    def __init__(self):
        """
        Initialize the RTX 6000 Ada memory manager.
        
        Sets up memory thresholds, monitoring parameters, and initializes
        the manager for optimal RTX 6000 Ada operation with 48GB VRAM.
        """
        self.torch = None
        self.total_vram_gb = 48
        self.critical_threshold_gb = 2.0   # 2GB remaining is critical
        self.warning_threshold_gb = 5.0    # 5GB remaining is warning
        self.safe_threshold_gb = 10.0      # 10GB remaining is safe
        self.last_cleanup = 0
        self.competing_processes = []
        self.models_loaded = False
        
        # Process monitoring configuration
        self.monitor_interval = 30  # Check every 30 seconds
        self.auto_kill_competing = True  # Automatically kill competing processes
        
    def import_torch(self):
        """
        Safely import PyTorch with lazy loading.
        
        This method imports PyTorch only when needed to avoid import overhead
        and potential conflicts during module initialization.
        
        Returns:
            torch module or None: PyTorch module if available, None if import fails
            
        Example:
            >>> torch = manager.import_torch()
            >>> if torch and torch.cuda.is_available():
            ...     print("CUDA available")
        """
        if self.torch is None:
            try:
                import torch
                self.torch = torch
                logger.info("PyTorch imported successfully")
            except ImportError as e:
                logger.error(f"Failed to import PyTorch: {e}")
        return self.torch
    
    def get_gpu_memory_status(self) -> Dict[str, float]:
        """
        Get current GPU memory status in GB.
        
        This method retrieves detailed GPU memory information including
        total, allocated, reserved, and free memory amounts.
        
        Returns:
            Dict[str, float]: Dictionary containing memory status:
                - total: Total GPU memory in GB
                - allocated: Currently allocated memory in GB
                - reserved: Reserved memory in GB
                - free: Available free memory in GB
                
        Example:
            >>> status = manager.get_gpu_memory_status()
            >>> print(f"Free: {status['free']:.1f}GB of {status['total']:.1f}GB")
        """
        torch = self.import_torch()
        if torch is None or not torch.cuda.is_available():
            return {"total": 0, "allocated": 0, "free": 0, "reserved": 0}
        
        try:
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            reserved = torch.cuda.memory_reserved(0) / (1024**3)
            free = total - allocated
            
            return {
                "total": total,
                "allocated": allocated,
                "reserved": reserved,
                "free": free
            }
        except Exception as e:
            logger.error(f"Error getting GPU memory status: {e}")
            return {"total": 0, "allocated": 0, "free": 0, "reserved": 0}
    
    def get_gpu_processes(self) -> List[Dict[str, Union[int, str, float]]]:
        """
        Get list of processes currently using GPU memory.
        
        This method uses nvidia-smi to query all processes currently using
        GPU memory and returns detailed information about each process.
        
        Returns:
            List[Dict]: List of process dictionaries containing:
                - pid (int): Process ID
                - name (str): Process name
                - memory_mb (int): Memory usage in MB
                - memory_gb (float): Memory usage in GB
                - cmdline (str): Full command line
                
        Example:
            >>> processes = manager.get_gpu_processes()
            >>> for proc in processes:
            ...     print(f"PID {proc['pid']}: {proc['memory_gb']:.1f}GB")
        """
        try:
            result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            processes = []
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.strip().split(', ')
                        if len(parts) >= 3:
                            try:
                                pid = int(parts[0])
                                name = parts[1]
                                memory_mb = int(parts[2])
                                
                                # Get additional process info
                                try:
                                    proc = psutil.Process(pid)
                                    cmdline = ' '.join(proc.cmdline())
                                except:
                                    cmdline = name
                                
                                processes.append({
                                    'pid': pid,
                                    'name': name,
                                    'memory_mb': memory_mb,
                                    'memory_gb': memory_mb / 1024,
                                    'cmdline': cmdline
                                })
                            except ValueError:
                                continue
            
            return processes
        except Exception as e:
            logger.error(f"Error getting GPU processes: {e}")
            return []
    
    def identify_competing_processes(self) -> List[Dict[str, Union[int, str, float]]]:
        """
        Identify processes that are competing for GPU memory.
        
        This method analyzes running GPU processes to identify those that
        are likely competing with the current application for GPU resources,
        typically other ML/AI frameworks or applications.
        
        Returns:
            List[Dict]: List of competing process dictionaries with same
                       structure as get_gpu_processes()
                       
        Process Detection Criteria:
            - Not the current process (different PID)
            - Command line contains ML/AI framework keywords
            - Keywords: python, jupyter, streamlit, torch, tensorflow, etc.
            
        Example:
            >>> competing = manager.identify_competing_processes()
            >>> if competing:
            ...     print(f"Found {len(competing)} competing processes")
        """
        processes = self.get_gpu_processes()
        current_pid = os.getpid()
        
        competing = []
        for proc in processes:
            if proc['pid'] != current_pid:
                # Check if it's likely a competing ML/AI process
                cmdline_lower = proc['cmdline'].lower()
                if any(keyword in cmdline_lower for keyword in [
                    'python', 'jupyter', 'streamlit', 'gradio', 'fastapi', 'torch', 'tensorflow',
                    'transformers', 'huggingface', 'model', 'training', 'inference'
                ]):
                    competing.append(proc)
                    logger.warning(f"Found competing process: PID {proc['pid']}, {proc['name']}, {proc['memory_gb']:.1f}GB")
        
        return competing
    
    def kill_competing_processes(self, min_memory_gb: float = 2.0) -> int:
        """
        Kill competing processes to free GPU memory.
        
        This method identifies and terminates competing processes that are
        using significant GPU memory, helping to free resources for the
        current application.
        
        Args:
            min_memory_gb (float): Minimum memory usage threshold for killing
                                 processes (default: 2.0GB)
        
        Returns:
            int: Number of processes successfully killed
            
        Process Termination:
            - Processes are sorted by memory usage (highest first)
            - Only processes using >= min_memory_gb are killed
            - Uses SIGKILL (signal 9) for immediate termination
            - Includes error handling for permission and process issues
            
        Example:
            >>> killed = manager.kill_competing_processes(min_memory_gb=1.0)
            >>> print(f"Killed {killed} competing processes")
        """
        competing = self.identify_competing_processes()
        killed = 0
        
        # Sort by memory usage (highest first)
        competing.sort(key=lambda x: x['memory_gb'], reverse=True)
        
        for proc in competing:
            if proc['memory_gb'] >= min_memory_gb:  # Only kill processes using significant memory
                try:
                    logger.warning(f"Killing competing process: PID {proc['pid']} ({proc['memory_gb']:.1f}GB)")
                    os.kill(proc['pid'], 9)  # SIGKILL
                    killed += 1
                    time.sleep(1)  # Wait a bit between kills
                except ProcessLookupError:
                    logger.info(f"Process {proc['pid']} already terminated")
                except PermissionError:
                    logger.error(f"Permission denied killing process {proc['pid']}")
                except Exception as e:
                    logger.error(f"Error killing process {proc['pid']}: {e}")
        
        if killed > 0:
            logger.info(f"Killed {killed} competing processes")
            time.sleep(3)  # Wait for memory to be freed
            self.clear_gpu_cache()
        
        return killed
    
    def clear_gpu_cache(self) -> None:
        """
        Clear GPU cache and perform garbage collection.
        
        This method performs comprehensive cleanup of GPU memory caches
        and forces garbage collection to free up available memory.
        
        Operations Performed:
            - Python garbage collection
            - PyTorch CUDA cache clearing
            - CUDA synchronization
            
        Example:
            >>> manager.clear_gpu_cache()
            # GPU cache cleared and memory freed
        """
        gc.collect()
        torch = self.import_torch()
        if torch and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                logger.info("GPU cache cleared")
            except Exception as e:
                logger.error(f"Error clearing GPU cache: {e}")
    
    def check_and_resolve_memory_conflicts(self) -> bool:
        """
        Check for memory conflicts and resolve them automatically.
        
        This method monitors current memory usage against defined thresholds
        and takes appropriate action to resolve memory conflicts, including
        cache clearing and process termination if necessary.
        
        Returns:
            bool: True if conflicts were resolved or no conflicts found,
                  False if conflicts persist
                  
        Memory Thresholds:
            - Critical (<2GB): Aggressive cleanup including process killing
            - Warning (<5GB): Standard cache clearing
            - Safe (>10GB): No action needed
            
        Resolution Strategy:
            1. Check current memory status
            2. If critical: clear cache + kill competing processes
            3. If warning: clear cache only
            4. Re-check memory status after actions
            
        Example:
            >>> resolved = manager.check_and_resolve_memory_conflicts()
            >>> if resolved:
            ...     print("Memory conflicts resolved")
        """
        status = self.get_gpu_memory_status()
        
        if status['free'] < self.critical_threshold_gb:
            logger.error(f"Critical memory situation: {status['free']:.1f}GB free")
            
            # First try standard cleanup
            self.clear_gpu_cache()
            
            # Check again
            status = self.get_gpu_memory_status()
            if status['free'] < self.critical_threshold_gb:
                logger.warning("Standard cleanup insufficient, checking for competing processes")
                
                # Kill competing processes if auto-kill is enabled
                if self.auto_kill_competing:
                    killed = self.kill_competing_processes()
                    if killed > 0:
                        # Check memory again after killing processes
                        time.sleep(2)
                        status = self.get_gpu_memory_status()
                        logger.info(f"Memory after killing processes: {status['free']:.1f}GB free")
                        return True
                else:
                    # Just log competing processes
                    competing = self.identify_competing_processes()
                    if competing:
                        logger.warning("Found competing processes (auto-kill disabled):")
                        for proc in competing:
                            logger.warning(f"  PID {proc['pid']}: {proc['name']} ({proc['memory_gb']:.1f}GB)")
                        return False
        
        elif status['free'] < self.warning_threshold_gb:
            logger.warning(f"Memory warning: {status['free']:.1f}GB free")
            self.clear_gpu_cache()
            return True
        
        return True
    
    def optimize_for_models(self, model_names: List[str]) -> bool:
        """
        Optimize memory allocation for specific models.
        
        This method prepares the GPU memory environment for loading specific
        models by estimating memory requirements and ensuring sufficient
        free memory is available.
        
        Args:
            model_names (List[str]): List of model names/identifiers to optimize for
            
        Returns:
            bool: True if sufficient memory is available, False otherwise
            
        Memory Estimation:
            - 7B models (e.g., Mistral-7B): ~15GB in FP16
            - E5-Large embedding model: ~3GB
            - Reranker models: ~1GB
            - Unknown models: ~5GB default estimate
            - Includes 5GB safety buffer
            
        Example:
            >>> models = ["mistral-7b-instruct", "e5-large-v2", "reranker"]
            >>> if manager.optimize_for_models(models):
            ...     print("Memory optimized for model loading")
        """
        logger.info(f"Optimizing memory for models: {model_names}")
        
        # Calculate estimated memory requirements
        estimated_memory = 0
        for model_name in model_names:
            if "mistral-7b" in model_name.lower() or "7b" in model_name.lower():
                estimated_memory += 15  # ~15GB for 7B model in FP16
            elif "e5-large" in model_name.lower():
                estimated_memory += 3   # ~3GB for E5-large
            elif "reranker" in model_name.lower():
                estimated_memory += 1   # ~1GB for reranker
            else:
                estimated_memory += 5   # Default estimate
        
        logger.info(f"Estimated memory requirement: {estimated_memory}GB")
        
        # Check if we have enough memory
        status = self.get_gpu_memory_status()
        available_memory = status['free']
        
        if available_memory < estimated_memory + 5:  # Need 5GB buffer
            logger.warning(f"Insufficient memory. Available: {available_memory:.1f}GB, Needed: {estimated_memory + 5}GB")
            
            # Try to free memory
            self.check_and_resolve_memory_conflicts()
            
            # Check again
            status = self.get_gpu_memory_status()
            if status['free'] < estimated_memory + 5:
                logger.error("Still insufficient memory after cleanup")
                return False
        
        return True
    
    def monitor_memory_continuously(self) -> None:
        """
        Continuously monitor memory and resolve conflicts in background.
        
        This method runs in a background thread to continuously monitor
        GPU memory usage and automatically resolve conflicts as they arise.
        It runs indefinitely until the application terminates.
        
        Monitoring Behavior:
            - Checks memory every monitor_interval seconds (default: 30s)
            - Automatically resolves conflicts when detected
            - Handles exceptions gracefully with extended retry intervals
            - Runs as daemon thread (terminates with main process)
            
        Example:
            >>> manager.monitor_memory_continuously()
            # Runs continuously in background (typically called by start_monitoring)
        """
        while True:
            try:
                self.check_and_resolve_memory_conflicts()
                time.sleep(self.monitor_interval)
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(60)  # Wait longer on error
    
    def start_monitoring(self) -> None:
        """
        Start background memory monitoring thread.
        
        This method starts a daemon thread that continuously monitors
        GPU memory usage and automatically resolves conflicts. The thread
        runs in the background and doesn't block the main application.
        
        Thread Configuration:
            - Daemon thread (terminates with main process)
            - Calls monitor_memory_continuously() in loop
            - Automatic conflict resolution enabled
            
        Example:
            >>> manager.start_monitoring()
            # Background monitoring started
        """
        monitor_thread = threading.Thread(target=self.monitor_memory_continuously, daemon=True)
        monitor_thread.start()
        logger.info("Started background memory monitoring")
    
    @contextmanager
    def managed_memory_context(self):
        """
        Context manager for operations requiring careful memory management.
        
        This context manager provides safe execution environment for memory-
        intensive operations with automatic cleanup and error recovery.
        
        Yields:
            None: Context for safe memory operations
            
        Context Behavior:
            - Pre-operation: Memory conflict resolution
            - During operation: CUDA OOM error handling
            - Post-operation: Memory status logging
            - Emergency cleanup on CUDA OOM errors
            
        Raises:
            MemoryError: If CUDA out of memory occurs (after cleanup)
            
        Example:
            >>> with manager.managed_memory_context():
            ...     model = load_large_model()  # Safe loading
            ...     result = model.generate(text)  # Safe inference
        """
        # Pre-operation cleanup
        initial_status = self.get_gpu_memory_status()
        logger.info(f"Starting operation with {initial_status['free']:.1f}GB free")
        
        # Resolve any conflicts before starting
        self.check_and_resolve_memory_conflicts()
        
        try:
            yield
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                logger.error(f"CUDA OOM during operation: {e}")
                # Emergency cleanup
                self.clear_gpu_cache()
                self.kill_competing_processes(min_memory_gb=1.0)  # Kill any process using >1GB
                raise MemoryError("CUDA out of memory - cleaned up competing processes") from e
            else:
                raise
        finally:
            # Post-operation cleanup
            final_status = self.get_gpu_memory_status()
            logger.info(f"Operation completed with {final_status['free']:.1f}GB free")
    
    def get_model_loading_config(self) -> Dict[str, Any]:
        """
        Get optimized model loading configuration for RTX 6000 Ada.
        
        This method returns model loading parameters optimized for the current
        GPU memory situation and RTX 6000 Ada capabilities.
        
        Returns:
            Dict[str, Any]: Model loading configuration dictionary containing:
                - device_map: GPU device mapping
                - torch_dtype: Data type for model weights
                - low_cpu_mem_usage: CPU memory optimization
                - trust_remote_code: Security setting
                - Additional memory and attention optimizations
                
        Configuration Tiers (based on available memory):
            - >30GB: Full FP16 with Flash Attention 2
            - >20GB: Standard FP16
            - >10GB: FP16 with memory limits
            - <10GB: 8-bit quantization with tight limits
            
        Example:
            >>> config = manager.get_model_loading_config()
            >>> model = AutoModel.from_pretrained(model_name, **config)
        """
        status = self.get_gpu_memory_status()
        
        config = {
            "device_map": {"": 0},  # Use GPU 0
            "torch_dtype": "float16",  # Use FP16 for efficiency
            "low_cpu_mem_usage": True,
            "trust_remote_code": False,
        }
        
        # Adjust based on available memory
        if status['free'] > 30:  # More than 30GB free
            config.update({
                "torch_dtype": "float16",
                "attn_implementation": "flash_attention_2",  # Use flash attention if available
            })
        elif status['free'] > 20:  # More than 20GB free
            config.update({
                "torch_dtype": "float16",
            })
        elif status['free'] > 10:  # More than 10GB free
            config.update({
                "torch_dtype": "float16",
                "max_memory": {0: f"{int(status['free'] - 5)}GB"},  # Reserve 5GB
            })
        else:  # Less than 10GB free - use quantization
            config.update({
                "load_in_8bit": True,
                "max_memory": {0: f"{int(status['free'] - 2)}GB"},  # Reserve 2GB
            })
        
        return config
    
    def log_detailed_status(self) -> None:
        """
        Log detailed memory and process status information.
        
        This method provides comprehensive logging of current GPU memory
        status, process information, and system state for debugging and
        monitoring purposes.
        
        Information Logged:
            - Total, allocated, free, and reserved VRAM amounts
            - Memory usage percentages
            - List of all GPU processes with memory usage
            - Formatted output for easy reading
            
        Example:
            >>> manager.log_detailed_status()
            # Logs detailed memory and process information
        """
        status = self.get_gpu_memory_status()
        processes = self.get_gpu_processes()
        
        logger.info("=== RTX 6000 Ada Memory Status ===")
        logger.info(f"Total VRAM: {status['total']:.1f}GB")
        logger.info(f"Allocated: {status['allocated']:.1f}GB ({status['allocated']/status['total']*100:.1f}%)")
        logger.info(f"Free: {status['free']:.1f}GB ({status['free']/status['total']*100:.1f}%)")
        logger.info(f"Reserved: {status['reserved']:.1f}GB")
        
        if processes:
            logger.info("GPU Processes:")
            for proc in processes:
                logger.info(f"  PID {proc['pid']}: {proc['name']} - {proc['memory_gb']:.1f}GB")
        
        logger.info("=" * 35)


# Create global instance for application-wide use
rtx6000_memory_manager = RTX6000MemoryManager()


# Convenience functions for easy access to manager functionality
def check_memory_conflicts() -> bool:
    """
    Check and resolve memory conflicts using the global manager.
    
    Returns:
        bool: True if conflicts resolved or none found, False otherwise
        
    Example:
        >>> if check_memory_conflicts():
        ...     print("Memory conflicts resolved")
    """
    return rtx6000_memory_manager.check_and_resolve_memory_conflicts()


def kill_competing_processes(min_memory_gb: float = 2.0) -> int:
    """
    Kill competing processes using the global manager.
    
    Args:
        min_memory_gb (float): Minimum memory usage threshold for killing processes
        
    Returns:
        int: Number of processes successfully killed
        
    Example:
        >>> killed = kill_competing_processes(min_memory_gb=1.0)
        >>> print(f"Killed {killed} processes")
    """
    return rtx6000_memory_manager.kill_competing_processes(min_memory_gb)


def managed_memory_context():
    """
    Get managed memory context manager from global manager.
    
    Returns:
        contextmanager: Context manager for safe memory operations
        
    Example:
        >>> with managed_memory_context():
        ...     # Safe memory operations
        ...     model = load_model()
    """
    return rtx6000_memory_manager.managed_memory_context()


def optimize_for_models(model_names: List[str]) -> bool:
    """
    Optimize memory for specific models using the global manager.
    
    Args:
        model_names (List[str]): List of model names to optimize for
        
    Returns:
        bool: True if sufficient memory available, False otherwise
        
    Example:
        >>> if optimize_for_models(["mistral-7b", "e5-large"]):
        ...     print("Memory optimized for models")
    """
    return rtx6000_memory_manager.optimize_for_models(model_names)


def get_model_loading_config() -> Dict[str, Any]:
    """
    Get model loading configuration from global manager.
    
    Returns:
        Dict[str, Any]: Optimized model loading configuration
        
    Example:
        >>> config = get_model_loading_config()
        >>> model = AutoModel.from_pretrained(model_name, **config)
    """
    return rtx6000_memory_manager.get_model_loading_config()


def start_memory_monitoring() -> None:
    """
    Start background memory monitoring using the global manager.
    
    Example:
        >>> start_memory_monitoring()
        # Background monitoring started
    """
    rtx6000_memory_manager.start_monitoring()


def log_memory_status() -> None:
    """
    Log detailed memory status using the global manager.
    
    Example:
        >>> log_memory_status()
        # Detailed memory status logged
    """
    rtx6000_memory_manager.log_detailed_status()


# Auto-start monitoring when module is imported
start_memory_monitoring()