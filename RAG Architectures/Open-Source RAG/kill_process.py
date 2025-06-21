#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced GPU process cleanup script for RTX 6000 Ada with intelligent process management.

This script provides automated identification and cleanup of competing Python processes
that consume GPU memory, specifically optimized for RTX 6000 Ada (48GB VRAM) systems
running multiple RAG/ML workloads. It intelligently distinguishes between system
processes, current processes, and competing instances to safely free GPU resources.

Key Features:
    - Intelligent process classification and filtering
    - Safe process termination with graceful fallback
    - Real-time GPU memory monitoring and reporting
    - Interactive confirmation for safety
    - Comprehensive logging and status reporting
    - RTX 6000 Ada optimization with 48GB VRAM awareness

Process Classification:
    - Current Process: The running script instance (preserved)
    - System Processes: OS and desktop environment processes (preserved)
    - Competing Processes: Other Python ML/RAG instances (targeted for cleanup)
    - Unknown Processes: Analyzed but handled conservatively

Safety Features:
    - Interactive confirmation before process termination
    - Graceful SIGTERM before forceful SIGKILL
    - Process existence verification before and after termination
    - Comprehensive error handling and permission management
    - Memory usage thresholds for selective targeting

Memory Management:
    - Real-time VRAM usage monitoring
    - Before/after memory comparison
    - Estimated vs actual memory recovery reporting
    - Optimization recommendations based on available memory

Usage:
    python kill_process.py

    The script will:
    1. Scan for GPU processes using nvidia-smi
    2. Classify and analyze each process
    3. Identify competing Python instances
    4. Request user confirmation
    5. Safely terminate competing processes
    6. Report memory recovery results

Dependencies:
    - nvidia-smi: NVIDIA GPU monitoring utility
    - psutil: System and process utilities
    - subprocess: Process execution and control

Example Output:
    ?? RTX 6000 Ada GPU Process Cleanup
    ?? Initial GPU Memory: 12.5GB free / 48.0GB total (26.0% free)
    ?? Analyzing GPU processes...
    ?? Found 2 competing Python processes using 8.2GB
    ? After cleanup: 41.3GB free / 48.0GB total (86.1% free)
"""
import subprocess
import os
import time
import signal
import psutil
import sys
from typing import List, Dict, Any, Optional


def get_gpu_processes() -> List[Dict[str, Any]]:
    """
    Get all processes currently using GPU memory via nvidia-smi.
    
    This function queries nvidia-smi to retrieve information about all processes
    currently utilizing GPU compute resources. It parses the CSV output to
    extract process IDs, names, and memory usage.
    
    Returns:
        List[Dict[str, Any]]: List of process dictionaries containing:
            - pid (int): Process ID
            - name (str): Process name/executable
            - memory_mb (int): Memory usage in megabytes
            - memory_gb (float): Memory usage in gigabytes
            
    Example:
        >>> processes = get_gpu_processes()
        >>> for proc in processes:
        ...     print(f"PID {proc['pid']}: {proc['name']} using {proc['memory_gb']:.1f}GB")
    """
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-compute-apps=pid,process_name,used_memory', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = line.strip().split(', ')
                if len(parts) >= 3:
                    try:
                        pid = int(parts[0])
                        name = parts[1]
                        memory_mb = int(parts[2])
                        processes.append({
                            'pid': pid,
                            'name': name,
                            'memory_mb': memory_mb,
                            'memory_gb': memory_mb / 1024
                        })
                    except ValueError:
                        continue
        return processes
    except Exception as e:
        print(f"? Error getting GPU processes: {e}")
        return []


def get_process_details(pid: int) -> Optional[Dict[str, Any]]:
    """
    Get detailed information about a specific process using psutil.
    
    This function retrieves comprehensive process information including
    command line arguments, working directory, creation time, and status
    for process classification and analysis.
    
    Args:
        pid (int): Process ID to analyze
        
    Returns:
        Optional[Dict[str, Any]]: Process details dictionary containing:
            - pid (int): Process ID
            - name (str): Process name
            - cmdline (str): Full command line with arguments
            - cwd (str): Current working directory
            - create_time (float): Process creation timestamp
            - status (str): Current process status
            Returns None if process cannot be accessed or doesn't exist
            
    Example:
        >>> details = get_process_details(1234)
        >>> if details:
        ...     print(f"Process: {details['name']}")
        ...     print(f"Command: {details['cmdline']}")
    """
    try:
        proc = psutil.Process(pid)
        return {
            'pid': pid,
            'name': proc.name(),
            'cmdline': ' '.join(proc.cmdline()),
            'cwd': proc.cwd() if hasattr(proc, 'cwd') else 'N/A',
            'create_time': proc.create_time(),
            'status': proc.status()
        }
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


def identify_competing_processes() -> List[Dict[str, Any]]:
    """
    Identify Python processes that are likely competing for GPU memory.
    
    This function analyzes all GPU processes to identify competing Python
    instances while preserving system processes and the current script.
    It uses intelligent classification based on process names, command lines,
    and memory usage patterns.
    
    Returns:
        List[Dict[str, Any]]: List of competing process dictionaries with
                             combined GPU usage and detailed process information
                             
    Classification Logic:
        - Current Process: Skip (identified by PID)
        - System Processes: Skip if low memory usage (<1GB)
        - Python Processes: Target if in test_env and using >500MB
        - Unknown Processes: Log but don't target
        
    Output Categories:
        ? Current Process: The running script
        ??? System Process: OS/desktop processes (preserved)
        ?? Competing Python Process: Target for cleanup
        ?? Other Process: Non-competing GPU user
        ? Unknown Process: Inaccessible process
        
    Example:
        >>> competing = identify_competing_processes()
        >>> total_memory = sum(p['memory_gb'] for p in competing)
        >>> print(f"Found {len(competing)} processes using {total_memory:.1f}GB")
    """
    gpu_processes = get_gpu_processes()
    current_pid = os.getpid()
    
    competing = []
    system_processes = {'Xorg', 'kwin_x11', 'plasmashell', 'chrome', 'firefox'}
    
    print("?? Analyzing GPU processes...")
    print(f"?? Found {len(gpu_processes)} processes using GPU:")
    
    for proc in gpu_processes:
        pid = proc['pid']
        name = proc['name']
        memory_gb = proc['memory_gb']
        
        # Skip current process
        if pid == current_pid:
            print(f"   ? PID {pid}: {name} ({memory_gb:.1f}GB) - CURRENT PROCESS")
            continue
            
        # Skip system processes with low memory usage
        if any(sys_proc in name for sys_proc in system_processes) and memory_gb < 1.0:
            print(f"   ???  PID {pid}: {name} ({memory_gb:.1f}GB) - SYSTEM PROCESS")
            continue
        
        # Get detailed process info
        details = get_process_details(pid)
        if details:
            # Check if it's a Python process in test_env (your competing instances)
            if ('python' in name.lower() and 
                'test_env' in details['cmdline'] and 
                memory_gb > 0.5):  # Using more than 500MB
                
                competing.append({
                    **proc,
                    **details
                })
                print(f"   ?? PID {pid}: {name} ({memory_gb:.1f}GB) - COMPETING PYTHON PROCESS")
            else:
                print(f"   ??  PID {pid}: {name} ({memory_gb:.1f}GB) - OTHER PROCESS")
        else:
            print(f"   ? PID {pid}: {name} ({memory_gb:.1f}GB) - UNKNOWN PROCESS")
    
    return competing


def kill_process_safely(pid: int, name: str, memory_gb: float) -> bool:
    """
    Safely terminate a process with graceful fallback to forceful termination.
    
    This function implements a two-stage process termination strategy:
    1. Graceful termination with SIGTERM (allows cleanup)
    2. Forceful termination with SIGKILL (if graceful fails)
    
    Args:
        pid (int): Process ID to terminate
        name (str): Process name for logging
        memory_gb (float): Memory usage for logging
        
    Returns:
        bool: True if process was successfully terminated, False otherwise
        
    Termination Process:
        1. Send SIGTERM signal for graceful shutdown
        2. Wait up to 3 seconds for process to exit
        3. If still running, send SIGKILL for immediate termination
        4. Verify process termination
        5. Handle permission errors and other exceptions
        
    Error Handling:
        - ProcessLookupError: Process already terminated
        - PermissionError: Insufficient privileges (suggest sudo)
        - Other exceptions: General error logging
        
    Example:
        >>> success = kill_process_safely(1234, "python", 2.5)
        >>> if success:
        ...     print("Process terminated successfully")
    """
    try:
        # First try SIGTERM (graceful)
        print(f"   ?? Sending SIGTERM to PID {pid} ({name}, {memory_gb:.1f}GB)...")
        os.kill(pid, signal.SIGTERM)
        
        # Wait up to 3 seconds for graceful shutdown
        for i in range(3):
            try:
                os.kill(pid, 0)  # Check if process still exists
                time.sleep(1)
            except ProcessLookupError:
                print(f"   ? PID {pid} terminated gracefully")
                return True
        
        # If still running, use SIGKILL
        print(f"   ?? Force killing PID {pid} with SIGKILL...")
        os.kill(pid, signal.SIGKILL)
        time.sleep(0.5)
        
        # Verify it's gone
        try:
            os.kill(pid, 0)
            print(f"   ? Failed to kill PID {pid}")
            return False
        except ProcessLookupError:
            print(f"   ? PID {pid} force killed successfully")
            return True
            
    except ProcessLookupError:
        print(f"   ??  PID {pid} already terminated")
        return True
    except PermissionError:
        print(f"   ?? Permission denied for PID {pid} - try running with sudo")
        return False
    except Exception as e:
        print(f"   ? Error killing PID {pid}: {e}")
        return False


def get_gpu_memory_status() -> Optional[Dict[str, float]]:
    """
    Get current GPU memory status including total, used, and free memory.
    
    This function queries nvidia-smi for detailed GPU memory information
    and calculates usage statistics for monitoring and reporting.
    
    Returns:
        Optional[Dict[str, float]]: Memory status dictionary containing:
            - total_mb (int): Total memory in megabytes
            - used_mb (int): Used memory in megabytes  
            - free_mb (int): Free memory in megabytes
            - total_gb (float): Total memory in gigabytes
            - used_gb (float): Used memory in gigabytes
            - free_gb (float): Free memory in gigabytes
            - usage_percent (float): Memory usage percentage
            - free_percent (float): Free memory percentage
            Returns None if query fails
            
    Example:
        >>> status = get_gpu_memory_status()
        >>> if status:
        ...     print(f"GPU Memory: {status['free_gb']:.1f}GB free")
        ...     print(f"Usage: {status['usage_percent']:.1f}%")
    """
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=memory.total,memory.used,memory.free', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        line = result.stdout.strip()
        if line:
            parts = line.split(', ')
            if len(parts) >= 3:
                total = int(parts[0])
                used = int(parts[1])
                free = int(parts[2])
                return {
                    'total_mb': total,
                    'used_mb': used,
                    'free_mb': free,
                    'total_gb': total / 1024,
                    'used_gb': used / 1024,
                    'free_gb': free / 1024,
                    'usage_percent': (used / total) * 100,
                    'free_percent': (free / total) * 100
                }
    except Exception as e:
        print(f"? Error getting GPU memory status: {e}")
    return None


def main() -> None:
    """
    Main cleanup function with comprehensive process management and reporting.
    
    This function orchestrates the complete GPU process cleanup workflow:
    1. Initial memory status assessment
    2. Competing process identification and analysis
    3. Interactive user confirmation for safety
    4. Safe process termination with detailed logging
    5. Memory recovery verification and reporting
    6. Final system status and recommendations
    
    Interactive Flow:
        - Display initial GPU memory status
        - Analyze and classify all GPU processes
        - Show competing processes and memory usage
        - Request user confirmation before cleanup
        - Execute safe process termination
        - Report cleanup results and memory recovery
        - Provide optimization recommendations
        
    Safety Features:
        - User confirmation required before any termination
        - Graceful process shutdown with fallback
        - Comprehensive error handling
        - Process classification to avoid system damage
        
    Memory Optimization:
        - RTX 6000 Ada specific recommendations (48GB VRAM)
        - Memory usage thresholds for different scenarios
        - Before/after memory comparison
        - Actual vs estimated recovery reporting
        
    Example Usage:
        $ python kill_process.py
        ?? RTX 6000 Ada GPU Process Cleanup
        ?? Initial GPU Memory: 12.5GB free / 48.0GB total
        ?? Found 2 competing processes using 8.2GB
        ??  About to kill 2 processes. Continue? (y/N): y
        ? Cleanup complete: 41.3GB free (86.1%)
    """
    print("?? RTX 6000 Ada GPU Process Cleanup")
    print("=" * 50)
    
    # Show initial status
    initial_status = get_gpu_memory_status()
    if initial_status:
        print(f"?? Initial GPU Memory: {initial_status['free_gb']:.1f}GB free / {initial_status['total_gb']:.1f}GB total ({initial_status['free_percent']:.1f}% free)")
    
    print()
    
    # Identify competing processes
    competing = identify_competing_processes()
    
    if not competing:
        print("\n? No competing Python processes found!")
        return
    
    print(f"\n?? Found {len(competing)} competing Python processes:")
    total_memory = sum(proc['memory_gb'] for proc in competing)
    print(f"?? Total memory used by competing processes: {total_memory:.1f}GB")
    
    # Ask for confirmation
    print(f"\n??  About to kill {len(competing)} processes. Continue? (y/N): ", end='')
    try:
        response = input().lower().strip()
        if response not in ['y', 'yes']:
            print("? Aborted by user")
            return
    except KeyboardInterrupt:
        print("\n? Aborted by user")
        return
    
    print(f"\n?? Killing {len(competing)} competing processes...")
    
    # Kill processes
    killed_count = 0
    freed_memory = 0
    
    for proc in competing:
        pid = proc['pid']
        name = proc['name']
        memory_gb = proc['memory_gb']
        
        if kill_process_safely(pid, name, memory_gb):
            killed_count += 1
            freed_memory += memory_gb
    
    print(f"\n?? Cleanup Summary:")
    print(f"   ? Killed: {killed_count}/{len(competing)} processes")
    print(f"   ?? Estimated freed memory: {freed_memory:.1f}GB")
    
    # Wait for GPU memory to be released
    print(f"\n? Waiting 5 seconds for GPU memory to be released...")
    time.sleep(5)
    
    # Show final status
    final_status = get_gpu_memory_status()
    if final_status and initial_status:
        freed_actual = final_status['free_gb'] - initial_status['free_gb']
        print(f"\n?? Final GPU Memory:")
        print(f"   ?? Before: {initial_status['free_gb']:.1f}GB free ({initial_status['free_percent']:.1f}%)")
        print(f"   ? After:  {final_status['free_gb']:.1f}GB free ({final_status['free_percent']:.1f}%)")
        print(f"   ?? Actually freed: {freed_actual:.1f}GB")
        
        # RTX 6000 Ada specific recommendations
        if final_status['free_gb'] > 30:
            print(f"\n?? Excellent! You now have {final_status['free_gb']:.1f}GB free for your RAG system!")
        elif final_status['free_gb'] > 20:
            print(f"\n? Good! You have {final_status['free_gb']:.1f}GB free - should be enough for your RAG system.")
        else:
            print(f"\n??  Only {final_status['free_gb']:.1f}GB free - you may need to kill more processes.")
    
    print(f"\n?? Final GPU process list:")
    final_processes = get_gpu_processes()
    for proc in final_processes:
        print(f"   PID {proc['pid']}: {proc['name']} ({proc['memory_gb']:.1f}GB)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n? Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n?? Unexpected error: {e}")
        sys.exit(1)