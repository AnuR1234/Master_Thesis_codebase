"""
Optimized memory management module for single RTX 4090 GPU system
"""
import gc
import logging
import signal
import atexit
import threading
import time
import os
import sys
import weakref
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Delay torch import to avoid issues
_torch_imported = False
_torch = None
_psutil_imported = False
_psutil = None
def check_cuda_config():
    """Check CUDA configuration and log values"""
    # Check if PYTORCH_CUDA_ALLOC_CONF is set
    conf_value = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "not set")
    logger.info(f"PYTORCH_CUDA_ALLOC_CONF: {conf_value}")
    
    # Check other relevant CUDA configs
    try:
        torch = import_torch()  # Use the import_torch function instead of direct import
        if torch and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"GPU {i}: {props.name}, SM: {props.major}.{props.minor}, "
                           f"Memory: {props.total_memory/1024**3:.1f} GB")
    except Exception as e:
        logger.error(f"Error checking CUDA config: {e}")
def import_torch():
    """Import torch only when needed to avoid early import issues"""
    global _torch, _torch_imported
    if not _torch_imported:
        try:
            import torch as _torch_module
            _torch = _torch_module
            _torch_imported = True
            logger.info("PyTorch imported successfully")
        except ImportError as e:
            logger.error(f"Error importing PyTorch: {e}")
    return _torch

def import_psutil():
    """Import psutil only when needed"""
    global _psutil, _psutil_imported
    if not _psutil_imported:
        try:
            import psutil as _psutil_module
            _psutil = _psutil_module
            _psutil_imported = True
            logger.info("psutil imported successfully")
        except ImportError as e:
            logger.error(f"Error importing psutil: {e}")
    return _psutil

class OptimizedMemoryManager:
    """Optimized memory manager for single RTX 4090 GPU"""
    
    def __init__(self):
        """Initialize the optimized memory manager"""
        self.resources_to_clean = []
        self.gpu_count = 0
        self.main_thread_id = threading.current_thread().ident
        self.is_shutting_down = False
        
        # Set memory thresholds for single GPU - INCREASED to reduce excessive cleanups
        self.max_memory_percentage = 90.0  # Increased from 85% to 90%
        self.mid_memory_percentage = 80.0  # Increased from 70% to 80%
        self.low_memory_percentage = 60.0  # Increased from 50% to 60%
        
        # Memory fragmentation threshold
        self.fragmentation_threshold = 30.0  # Percentage threshold for fragmentation
        
        # Tracking timestamps for optimized cleanup scheduling
        self.last_cleanup_time = time.time()
        self.last_scheduled_cleanup = time.time()
        self.last_emergency_cleanup = 0  # Initialize to 0 to allow immediate emergency cleanup if needed
        
        # Keep track of whether streamlit is rerunning
        self.streamlit_session_id = None
        
        # Register cleanup handlers
        atexit.register(self.cleanup_all)
        self._setup_signal_handlers()
        
        # Check for GPUs
        torch = import_torch()
        if torch and torch.cuda.is_available():
            self.gpu_count = torch.cuda.device_count()
            logger.info(f"Optimized Memory Manager initialized with {self.gpu_count} GPUs")
            self._log_gpu_status("Initial GPU status")
        else:
            logger.info("Optimized Memory Manager initialized (no GPUs available)")
    
    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown"""
        try:
            # Only register in main thread to avoid warnings
            if threading.current_thread() is threading.main_thread():
                signal.signal(signal.SIGTERM, self._signal_handler)
                signal.signal(signal.SIGINT, self._signal_handler)
                if hasattr(signal, 'SIGQUIT'):
                    signal.signal(signal.SIGQUIT, self._signal_handler)
                logger.info("Signal handlers registered successfully")
            else:
                logger.info("Skipping signal handler registration in non-main thread")
        except Exception as e:
            logger.warning(f"Could not register signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle signals by cleaning up and then exiting"""
        logger.info(f"Received signal {signum}, cleaning up resources")
        self.cleanup_all()
        sys.exit(0)
    
    def _log_gpu_status(self, message="GPU Status"):
        """Log the current status of all GPUs"""
        torch = import_torch()
        if not torch or not torch.cuda.is_available():
            logger.info("No GPUs available")
            return
            
        logger.info(f"===== {message} =====")
        for i in range(self.gpu_count):
            try:
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                total = torch.cuda.get_device_properties(i).total_memory
                
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"  Allocated: {allocated/1024**3:.2f} GB ({allocated/total*100:.2f}%)")
                logger.info(f"  Reserved: {reserved/1024**3:.2f} GB ({reserved/total*100:.2f}%)")
                logger.info(f"  Total: {total/1024**3:.2f} GB")
                
                # Add more detailed memory tracking for single GPU
                if reserved > 0:
                    fragmentation = (reserved - allocated) / reserved * 100
                    logger.info(f"  Memory fragmentation: {fragmentation:.2f}%")
                    if fragmentation > self.fragmentation_threshold:
                        logger.warning(f"High memory fragmentation detected: {fragmentation:.2f}%")
            except Exception as e:
                logger.error(f"Error logging status for GPU {i}: {e}")
                
        logger.info("=" * 40)
    
    def register_resource(self, resource, cleanup_func=None):
        """
        Register a resource for cleanup
        
        Args:
            resource: The resource to clean up (model, etc.)
            cleanup_func: Optional custom cleanup function
        """
        if resource is None:
            return None
            
        # Use weakref to avoid creating circular references
        resource_ref = weakref.ref(resource)
        self.resources_to_clean.append((resource_ref, cleanup_func))
        resource_name = type(resource).__name__
        logger.debug(f"Registered resource {resource_name} for cleanup")
        
        return resource
    
    def clear_gpu_memory(self):
        """Clear CUDA GPU memory"""
        # Run garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        torch = import_torch()
        if torch and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("GPU memory cleared successfully")
                
                # Log memory status after cleanup
                self._log_gpu_status("GPU Status After Cleanup")
            except Exception as e:
                logger.error(f"Error clearing GPU memory: {e}")
    
    def _unload_tensors_from_gpu(self):
        """Unload all tensors from GPU to CPU - more aggressive cleanup"""
        torch = import_torch()
        if not torch or not torch.cuda.is_available():
            return

        try:
            # Find all tensors and move them to CPU
            count = 0
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) and obj.is_cuda:
                        obj.data = obj.data.cpu()
                        count += 1
                except:
                    pass
            logger.info(f"Moved {count} tensors from GPU to CPU")
        except Exception as e:
            logger.error(f"Error unloading tensors: {e}")
    
    def perform_selective_cleanup(self):
        """Perform selective cleanup that preserves critical models for class name queries"""
        try:
            logger.info("Starting selective cleanup to preserve critical models")
            
            # Only unload generator model, keep retriever and reranker
            from generator import RAGGenerator
            for obj in gc.get_objects():
                if isinstance(obj, RAGGenerator) and hasattr(obj, '_unload_model'):
                    obj._unload_model()
                    logger.info("Selectively unloaded generator model")
            
            # Run garbage collection
            gc.collect()
            
            # Clear CUDA cache
            torch = import_torch()
            if torch and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Selective cleanup completed")
        except Exception as e:
            logger.error(f"Error in selective cleanup: {e}")
    
    def emergency_cleanup(self):
        """
        Perform aggressive memory cleanup for emergency situations
        Optimized for recovering from memory fragmentation and OOM situations
        """
        # Rate limit emergency cleanups (not more often than once per 30 seconds)
        current_time = time.time()
        if current_time - self.last_emergency_cleanup < 30:
            logger.info("Skipping emergency cleanup - last one was less than 30 seconds ago")
            return

        logger.info("Starting emergency memory cleanup procedure")
        self.last_emergency_cleanup = current_time
        
        # Log initial memory state
        self._log_gpu_status("GPU Status BEFORE Emergency Cleanup")
        
        # STEP 1: Identify any large model instances in memory
        try:
            # Model classes that typically consume large amounts of memory
            model_classes = [
                "AutoModelForCausalLM", "LlamaForCausalLM", "MistralForCausalLM", 
                "SentenceTransformer", "CrossEncoder", "Pipeline", "Model"
            ]
            
            # Track found models for reporting
            found_models = []
            
            # Scan for model instances
            for obj in gc.get_objects():
                try:
                    # Check if this is a large PyTorch module
                    if hasattr(obj, '__class__') and 'torch.nn.modules' in str(obj.__class__.__module__):
                        class_name = obj.__class__.__name__
                        if any(model_name in class_name for model_name in model_classes):
                            # Try to move to CPU if on GPU
                            if hasattr(obj, 'parameters'):
                                try:
                                    is_cuda = next(iter(obj.parameters())).is_cuda
                                    if is_cuda:
                                        logger.info(f"Moving {class_name} to CPU")
                                        obj.to('cpu')
                                        found_models.append(class_name)
                                except (StopIteration, RuntimeError):
                                    # Handle case where model has no parameters or other errors
                                    pass
                except Exception as obj_error:
                    # Safely continue if an object can't be inspected
                    continue
                    
            if found_models:
                logger.info(f"Found and moved to CPU: {', '.join(found_models)}")
        except Exception as scan_error:
            logger.error(f"Error scanning for model instances: {scan_error}")

        # STEP 2: Force deallocate any tensors still on GPU
        torch = import_torch()
        if torch and torch.cuda.is_available():
            try:
                # Log memory status
                before_allocated = torch.cuda.memory_allocated(0)
                before_reserved = torch.cuda.memory_reserved(0)
                
                # Find and move all tensors from GPU to CPU
                count = 0
                for obj in gc.get_objects():
                    try:
                        if torch.is_tensor(obj) and obj.is_cuda:
                            obj.data = obj.data.cpu()
                            count += 1
                            
                            # Periodically run GC to prevent memory spikes during cleanup
                            if count % 1000 == 0:
                                gc.collect()
                    except:
                        pass
                        
                logger.info(f"Moved {count} CUDA tensors to CPU")
                
                # Clear CUDA cache after tensor movement
                torch.cuda.empty_cache()
                
                # Log memory change
                after_allocated = torch.cuda.memory_allocated(0)
                after_reserved = torch.cuda.memory_reserved(0)
                
                freed_allocated = (before_allocated - after_allocated) / (1024**3)
                freed_reserved = (before_reserved - after_reserved) / (1024**3)
                
                logger.info(f"Freed {freed_allocated:.2f}GB allocated and {freed_reserved:.2f}GB reserved memory")
                
            except Exception as tensor_error:
                logger.error(f"Error moving tensors from GPU: {tensor_error}")
        
        # STEP 3: Aggressive garbage collection with multiple passes
        try:
            logger.info("Running multi-pass garbage collection")
            
            # First pass - collect obvious garbage
            gc.collect()
            
            # Enable debugging for garbage collection
            gc.set_debug(gc.DEBUG_LEAK)
            
            # Second pass - collect with full debug info
            collected = gc.collect()
            logger.info(f"Collected {collected} objects in second GC pass")
            
            # Disable debug mode
            gc.set_debug(0)
            
            # Third pass - final sweep
            gc.collect()
            
        except Exception as gc_error:
            logger.error(f"Error during garbage collection: {gc_error}")
        
        # STEP 4: Thorough CUDA memory cleanup if available
        if torch and torch.cuda.is_available():
            try:
                logger.info("Performing thorough CUDA memory cleanup")
                
                # Empty cache multiple times
                for i in range(3):
                    torch.cuda.empty_cache()
                    
                # Advanced memory management if available
                try:
                    # Reset peak memory stats
                    torch.cuda.reset_peak_memory_stats()
                    
                    # Reset accumulated memory stats if available
                    if hasattr(torch.cuda, 'reset_accumulated_memory_stats'):
                        torch.cuda.reset_accumulated_memory_stats()
                    
                    # Attempt to trigger non-public memory compaction APIs
                    # (This is implementation-specific and may change between PyTorch versions)
                    if hasattr(torch.cuda, '_memory_guards') and callable(getattr(torch.cuda, '_memory_guards', None)):
                        # Force memory compaction with guards
                        with torch.cuda._memory_guards():
                            pass
                    
                    # Last resort: try to dump memory snapshot to identify leaks if debug tools available
                    if hasattr(torch.cuda, 'memory') and hasattr(torch.cuda.memory, '_dump_snapshot'):
                        try:
                            torch.cuda.memory._dump_snapshot()
                        except:
                            pass
                            
                except Exception as advanced_error:
                    logger.debug(f"Error in advanced memory cleanup: {advanced_error}")
                
                # Force synchronize CUDA device
                torch.cuda.synchronize()
                
                # Final empty cache after sync
                torch.cuda.empty_cache()
                
            except Exception as cuda_error:
                logger.error(f"Error in CUDA memory cleanup: {cuda_error}")
        
        # STEP 5: Check for memory fragmentation
        if torch and torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated(0)
                reserved = torch.cuda.memory_reserved(0)
                total = torch.cuda.get_device_properties(0).total_memory
                
                fragmentation = 0
                if reserved > 0:
                    fragmentation = (reserved - allocated) / reserved * 100
                    
                logger.info(f"Memory fragmentation after cleanup: {fragmentation:.2f}%")
                
                # If fragmentation remains high, attempt more aggressive measures
                if fragmentation > 30:
                    logger.warning(f"Memory still fragmented ({fragmentation:.2f}%) after cleanup")
                    
                    # Attempt to use CUDA caching allocator to reduce fragmentation
                    try:
                        # Create and immediately delete a few large tensors to consolidate memory
                        max_tensor_size = max(int((reserved - allocated) * 0.8), 1024*1024*10)  # at least 10MB
                        logger.info(f"Creating temporary tensor of size {max_tensor_size/1024/1024:.2f}MB to consolidate memory")
                        
                        # Create tensor (device takes a device instance or index)
                        temp_tensor = torch.empty(max_tensor_size, device=0)
                        # Immediately delete and clear cache
                        del temp_tensor
                        torch.cuda.empty_cache()
                        
                    except Exception as tensor_error:
                        logger.error(f"Error creating temporary tensor: {tensor_error}")
            except Exception as frag_error:
                logger.error(f"Error checking memory fragmentation: {frag_error}")
        
        # STEP 6: Explicitly unload any models we may have registered
        self._unload_models()
        
        # STEP 7: Run through all registered resources
        cleaned_resources = 0
        for resource_ref, cleanup_func in self.resources_to_clean:
            try:
                # Get the actual resource from the weakref
                resource = resource_ref()
                
                # Skip if the reference has been garbage collected
                if resource is None:
                    continue
                
                # Try to clean up the resource
                if cleanup_func and callable(cleanup_func):
                    cleanup_func(resource)
                    cleaned_resources += 1
                elif hasattr(resource, '_unload_model') and callable(getattr(resource, '_unload_model')):
                    resource._unload_model()
                    cleaned_resources += 1
                elif hasattr(resource, 'to') and callable(getattr(resource, 'to')):
                    try:
                        resource.to('cpu')
                        cleaned_resources += 1
                    except Exception as e:
                        logger.debug(f"Error moving resource to CPU: {e}")
            except Exception as e:
                logger.debug(f"Error cleaning up resource: {e}")
        
        logger.info(f"Cleaned up {cleaned_resources} registered resources")
        
        # STEP 8: Final garbage collection and memory check
        gc.collect()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        # Log final memory state
        self._log_gpu_status("GPU Status AFTER Emergency Cleanup")
        
        # Get system info
        try:
            psutil = import_psutil()
            if psutil:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                logger.info(f"System memory usage after cleanup: {memory_info.rss/1024/1024:.2f} MB")
        except Exception as system_error:
            logger.debug(f"Error getting system memory info: {system_error}")
        
        logger.info("Emergency memory cleanup completed")
    
    def _unload_models(self):
        """Unload any known model types used in the application"""
        try:
            # Find all modules that might have loaded models
            modules_to_check = ['embedding', 'generator', 'retriever', 'pipeline']
            for module_name in modules_to_check:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    
                    # Look for instances of classes that might have _unload_model methods
                    for obj_name in dir(module):
                        try:
                            obj = getattr(module, obj_name)
                            
                            # Check if this is an instance with an _unload_model method
                            if hasattr(obj, '_unload_model') and callable(getattr(obj, '_unload_model')):
                                try:
                                    # Call the instance method properly
                                    obj._unload_model()
                                    logger.info(f"Unloaded model from {module_name}.{obj_name}")
                                except Exception as e:
                                    logger.error(f"Error unloading model instance {module_name}.{obj_name}: {e}")
                                    
                            # Also look for class definitions with _unload_model
                            elif isinstance(obj, type) and hasattr(obj, '_unload_model'):
                                # This is a class definition, not an instance
                                logger.debug(f"Found class with _unload_model: {module_name}.{obj_name}")
                                
                                # Look for instances of this class in the module
                                for attr_name in dir(module):
                                    try:
                                        attr = getattr(module, attr_name)
                                        if isinstance(attr, obj) and hasattr(attr, '_unload_model'):
                                            # Call the unload method on the instance
                                            attr._unload_model()
                                            logger.info(f"Unloaded model from instance {module_name}.{attr_name}")
                                    except Exception as inner_e:
                                        logger.debug(f"Error checking instance {attr_name}: {inner_e}")
                        except Exception as obj_error:
                            logger.debug(f"Error accessing object {obj_name}: {obj_error}")
            
            # Also check for global instances that might be directly in the module
            for module_name in modules_to_check:
                if module_name in sys.modules:
                    module = sys.modules[module_name]
                    
                    # Check for specific global instance names
                    for instance_name in ['generator', 'retriever', 'pipeline', 'embedder']:
                        if hasattr(module, instance_name):
                            instance = getattr(module, instance_name)
                            if hasattr(instance, '_unload_model') and callable(getattr(instance, '_unload_model')):
                                try:
                                    instance._unload_model()
                                    logger.info(f"Unloaded model from global instance {module_name}.{instance_name}")
                                except Exception as e:
                                    logger.error(f"Error unloading model from global instance {module_name}.{instance_name}: {e}")
            
            logger.info("Attempted to unload all models")
        except Exception as e:
            logger.error(f"Error in _unload_models: {e}")
    
    def cleanup_all(self):
        """Clean up all registered resources and free memory"""
        # Set shutdown flag
        self.is_shutting_down = True
        
        # Log memory before cleanup
        logger.info("Starting memory cleanup")
        self._log_gpu_status("GPU Status Before Cleanup")
        
        # Clean specific resources
        resources_to_remove = []
        
        for resource_ref, cleanup_func in self.resources_to_clean:
            try:
                # Get the actual resource from the weakref
                resource = resource_ref()
                
                # Skip if the reference has been garbage collected
                if resource is None:
                    resources_to_remove.append((resource_ref, cleanup_func))
                    continue
                
                # Try to clean up the resource
                if cleanup_func and callable(cleanup_func):
                    cleanup_func(resource)
                elif hasattr(resource, '_unload_model') and callable(getattr(resource, '_unload_model')):
                    # If it has an unload method, call it
                    resource._unload_model()
                    logger.info(f"Called _unload_model on {type(resource).__name__}")
                elif hasattr(resource, 'to') and callable(getattr(resource, 'to')):
                    # If it's a PyTorch model, move to CPU
                    try:
                        resource.to('cpu')
                        logger.info(f"Moved model to CPU")
                    except Exception as e:
                        logger.error(f"Error moving model to CPU: {e}")
                
                resources_to_remove.append((resource_ref, cleanup_func))
            except Exception as e:
                logger.error(f"Error cleaning up resource: {e}")
        
        # Remove cleaned resources from the list
        for resource_item in resources_to_remove:
            if resource_item in self.resources_to_clean:
                self.resources_to_clean.remove(resource_item)
        
        # Run garbage collection multiple times
        for _ in range(2):
            gc.collect()
        
        # Clear CUDA cache
        torch = import_torch()
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Log memory after cleanup
        self._log_gpu_status("GPU Status After Cleanup")
        
        # Print system memory info
        psutil = import_psutil()
        if psutil:
            try:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                logger.info(f"System memory usage after cleanup: {memory_info.rss/1024/1024:.2f} MB")
            except Exception as e:
                logger.error(f"Error getting system memory info: {e}")
        
        logger.info("All memory cleanup completed")
    
    def check_memory_pressure(self) -> bool:
        """
        Check if memory pressure is high and cleanup should run
        More aggressive thresholds for single GPU
        
        Returns:
            True if cleanup should be run, False otherwise
        """
        try:
            torch = import_torch()
            if torch and torch.cuda.is_available():
                for i in range(self.gpu_count):
                    allocated = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    total = torch.cuda.get_device_properties(i).total_memory
                    percentage = (allocated / total) * 100
                    
                    # Check for memory fragmentation
                    fragmentation = 0
                    if reserved > 0:
                        fragmentation = (reserved - allocated) / reserved * 100
                    
                    # Use a tiered approach for single GPU
                    current_time = time.time()
                    
                    # Critical memory pressure - emergency cleanup
                    if percentage > self.max_memory_percentage:
                        # Rate limit emergency cleanups
                        if current_time - self.last_emergency_cleanup >= 120:  # Not more than once per 2 minutes
                            logger.warning(f"Critical memory usage on GPU {i}: {percentage:.2f}% > {self.max_memory_percentage}%")
                            self.last_emergency_cleanup = current_time
                            self.emergency_cleanup()
                            return True
                    
                    # High memory pressure - standard cleanup
                    elif percentage > self.mid_memory_percentage:
                        # Rate limit standard cleanups
                        if current_time - self.last_cleanup_time >= 30:  # Not more than once per 30 seconds
                            logger.info(f"High memory usage on GPU {i}: {percentage:.2f}% > {self.mid_memory_percentage}%")
                            self.last_cleanup_time = current_time
                            return True
                    
                    # Medium memory pressure - scheduled cleanup
                    elif percentage > self.low_memory_percentage:
                        # Rate limit scheduled cleanups
                        if current_time - self.last_scheduled_cleanup >= 300:  # Not more than once per 5 minutes
                            logger.info(f"Medium memory usage on GPU {i}: {percentage:.2f}% > {self.low_memory_percentage}%")
                            self.last_scheduled_cleanup = current_time
                            return True
                    
                    # Check for high fragmentation regardless of memory usage
                    if fragmentation > self.fragmentation_threshold:
                        # Rate limit fragmentation cleanups
                        if current_time - self.last_cleanup_time >= 60:  # Not more than once per minute
                            logger.warning(f"High memory fragmentation: {fragmentation:.2f}% > {self.fragmentation_threshold}%")
                            self.last_cleanup_time = current_time
                            return True
            
            # Also check system memory
            psutil = import_psutil()
            if psutil:
                mem = psutil.virtual_memory()
                percentage = mem.percent
                
                if percentage > self.max_memory_percentage:
                    logger.warning(f"High system memory usage: {percentage:.2f}% > {self.max_memory_percentage}%")
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking memory pressure: {e}")
            return False
    
    def get_memory_stats(self):
        """
        Get current memory statistics
        
        Returns:
            Dictionary with GPU and system memory statistics
        """
        stats = {
            "system": {},
            "gpus": []
        }
        
        # System memory
        psutil = import_psutil()
        if psutil:
            try:
                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()
                stats["system"]["total"] = psutil.virtual_memory().total
                stats["system"]["available"] = psutil.virtual_memory().available
                stats["system"]["process"] = memory_info.rss
                stats["system"]["percent"] = psutil.virtual_memory().percent
            except Exception as e:
                logger.error(f"Error getting system memory stats: {e}")
        
        # GPU memory
        torch = import_torch()
        if torch and torch.cuda.is_available():
            for i in range(self.gpu_count):
                try:
                    allocated = torch.cuda.memory_allocated(i)
                    reserved = torch.cuda.memory_reserved(i)
                    total = torch.cuda.get_device_properties(i).total_memory
                    
                    fragmentation = 0
                    if reserved > 0:
                        fragmentation = (reserved - allocated) / reserved * 100
                    
                    gpu_stats = {
                        "id": i,
                        "name": torch.cuda.get_device_name(i),
                        "total": total,
                        "allocated": allocated,
                        "reserved": reserved,
                        "percent": (allocated / total) * 100,
                        "fragmentation": fragmentation
                    }
                    stats["gpus"].append(gpu_stats)
                except Exception as e:
                    logger.error(f"Error getting GPU {i} stats: {e}")
        
        return stats

    def perform_scheduled_cleanup(self):
        """
        Perform a scheduled cleanup operation to prevent memory fragmentation
        Less aggressive than emergency cleanup but more thorough than standard
        """
        try:
            # Rate limit scheduled cleanups
            current_time = time.time()
            if current_time - self.last_scheduled_cleanup < 300:  # Not more than once per 5 minutes
                logger.info("Skipping scheduled cleanup - last one was less than 5 minutes ago")
                return
            
            logger.info("Starting scheduled cleanup to prevent memory fragmentation")
            self.last_scheduled_cleanup = current_time
            
            # Log memory status before cleanup
            self._log_gpu_status("GPU Status Before Scheduled Cleanup")
            
            # Run garbage collection
            gc.collect()
            
            # Clear CUDA cache
            torch = import_torch()
            if torch and torch.cuda.is_available():
                allocated_before = torch.cuda.memory_allocated(0)
                reserved_before = torch.cuda.memory_reserved(0)
                
                # Empty cache
                torch.cuda.empty_cache()
                
                # Check memory after cleanup
                allocated_after = torch.cuda.memory_allocated(0)
                reserved_after = torch.cuda.memory_reserved(0)
                
                # Calculate changes
                allocated_diff = (allocated_before - allocated_after) / (1024**3)
                reserved_diff = (reserved_before - reserved_after) / (1024**3)
                
                logger.info(f"Scheduled cleanup released: allocated={allocated_diff:.2f}GB, reserved={reserved_diff:.2f}GB")
                
                # Check if memory is still fragmented
                if reserved_after > 0:
                    fragmentation = (reserved_after - allocated_after) / reserved_after * 100
                    if fragmentation > self.fragmentation_threshold:
                        logger.warning(f"Memory still fragmented ({fragmentation:.2f}%) after scheduled cleanup.")
                        
                        # Check if we should do a more aggressive cleanup
                        if fragmentation > 50 and current_time - self.last_emergency_cleanup >= 300:
                            logger.warning("Severe fragmentation detected. Performing emergency cleanup.")
                            self.emergency_cleanup()
            
            # Log memory status after cleanup
            self._log_gpu_status("GPU Status After Scheduled Cleanup")
        except Exception as e:
            logger.error(f"Error in scheduled cleanup: {e}")

    @contextmanager
    def resource_context(self, resource, cleanup_func=None):
        """
        Context manager for automatic resource cleanup
        
        Args:
            resource: Resource to register for cleanup
            cleanup_func: Optional custom cleanup function
            
        Yields:
            The resource for use in the context
        """
        if resource is None:
            yield None
            return
            
        try:
            self.register_resource(resource, cleanup_func)
            yield resource
        finally:
            # Clean up if not already shutting down
            if not self.is_shutting_down:
                try:
                    if cleanup_func and callable(cleanup_func):
                        cleanup_func(resource)
                    elif hasattr(resource, '_unload_model') and callable(getattr(resource, '_unload_model')):
                        resource._unload_model()
                    elif hasattr(resource, 'to') and callable(getattr(resource, 'to')):
                        # If it's a PyTorch model, move to CPU
                        try:
                            resource.to('cpu')
                        except:
                            pass
                except Exception as e:
                    logger.error(f"Error in resource context cleanup: {e}")
                
                # Remove from resources list
                self.resources_to_clean = [
                    (r, f) for r, f in self.resources_to_clean 
                    if r is not weakref.ref(resource)
                ]

# Create a global memory manager instance
memory_manager = OptimizedMemoryManager()

def cleanup_on_conversation_end():
    """
    Function to call when a conversation ends
    More thorough cleanup for single GPU setup
    """
    # Standard garbage collection
    gc.collect()
    
    # Clear GPU memory
    memory_manager.clear_gpu_memory()
    
    # Check if we need more aggressive cleanup
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(0)
            total = torch.cuda.get_device_properties(0).total_memory
            percentage = (allocated / total) * 100
            
            # Check for fragmentation
            reserved = torch.cuda.memory_reserved(0)
            fragmentation = 0
            if reserved > 0:
                fragmentation = (reserved - allocated) / reserved * 100
            
            if percentage > 60 or fragmentation > 30:  # If still using over 60% or high fragmentation
                logger.warning(f"High memory usage or fragmentation after standard cleanup: usage={percentage:.2f}%, fragmentation={fragmentation:.2f}%. Performing more aggressive cleanup.")
                # More aggressive cleanup
                memory_manager.perform_scheduled_cleanup()
            
            logger.info(f"Memory cleaned up after conversation end. Current usage: {percentage:.2f}%, fragmentation: {fragmentation:.2f}%")
    except Exception as e:
        logger.error(f"Error during post-conversation cleanup: {e}")
    
    logger.info("Memory cleaned up after conversation end")

def emergency_cleanup():
    """Perform emergency cleanup when memory usage is critical"""
    memory_manager.emergency_cleanup()

def get_memory_stats():
    """Get current memory statistics"""
    return memory_manager.get_memory_stats()

def check_memory_pressure():
    """Check if memory pressure is high and cleanup should be performed"""
    return memory_manager.check_memory_pressure()

def perform_scheduled_cleanup():
    """Perform scheduled cleanup to prevent memory fragmentation in single GPU setup"""
    memory_manager.perform_scheduled_cleanup()

def perform_selective_cleanup():
    """Perform selective cleanup that preserves retriever models for class queries"""
    memory_manager.perform_selective_cleanup()

def register_model_for_cleanup(model, device_id=None):
    """Register a model for cleanup when app exits"""
    return memory_manager.register_resource(model)

def cleanup_all_resources():
    """Clean up all registered resources and memory"""
    memory_manager.cleanup_all()