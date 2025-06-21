#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete run script for the Enhanced Anti-Hallucination RAG System.

This module provides a comprehensive management interface for the Enhanced 
Anti-Hallucination RAG (Retrieval-Augmented Generation) System. It handles
system initialization, dependency checking, testing, and application launching
with specific optimizations for RTX 6000 Ada GPU hardware.

Key Features:
    - Automated dependency checking and installation
    - RTX 6000 Ada GPU detection and optimization
    - Comprehensive system health checks
    - Anti-hallucination testing suite
    - Streamlit application launcher
    - Real-time monitoring dashboard
    - Memory management and CUDA configuration

Classes:
    None (module uses functional programming approach)

Functions:
    setup_environment: Setup environment variables for RTX 6000 Ada
    check_dependencies: Check if required dependencies are installed
    check_cuda: Check CUDA availability and RTX 6000 Ada detection
    check_files: Check if required files exist
    check_qdrant_connection: Check Qdrant vector database connection
    run_tests: Run comprehensive anti-hallucination tests
    run_quick_test: Run a quick test to verify system functionality
    run_streamlit: Run the Streamlit web application
    show_system_info: Show detailed system information
    main: Main function to run the enhanced system

Dependencies:
    - streamlit: Web interface framework
    - torch: PyTorch deep learning framework
    - transformers: Hugging Face transformers library
    - sentence-transformers: Sentence embedding models
    - qdrant-client: Vector database client
    - numpy: Numerical computing library
    - nest-asyncio: Nested event loop support
    - psutil: System and process utilities
    - fastembed: Fast embedding generation

Hardware Requirements:
    - NVIDIA RTX 6000 Ada (48GB VRAM) - recommended
    - CUDA-compatible GPU with 12GB+ VRAM - minimum
    - 32GB+ system RAM - recommended

Usage:
    python run_system.py

    This will launch an interactive menu with options for:
    1. Running comprehensive tests
    2. Starting the Streamlit application
    3. Quick testing and launching
    4. System information display
    5. Functionality testing
    6. Monitoring dashboard
    7. Exit
"""
import os
import sys
import subprocess
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Set up logging with detailed formatting
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_environment() -> None:
    """
    Setup environment variables for RTX 6000 Ada GPU optimization.
    
    This function configures PyTorch CUDA memory allocation, disables
    unnecessary warnings, and sets up optimal environment variables
    for the RTX 6000 Ada GPU with 48GB VRAM.
    
    Environment Variables Set:
        - PYTORCH_CUDA_ALLOC_CONF: CUDA memory allocation strategy
        - CUDA_LAUNCH_BLOCKING: Synchronous CUDA kernel execution
        - TRANSFORMERS_NO_ADVISORY_WARNINGS: Disable transformer warnings
        - DISABLE_FLASH_ATTENTION: Disable flash attention for stability
        - TOKENIZERS_PARALLELISM: Disable tokenizer parallelism
        
    Example:
        >>> setup_environment()
        # Environment configured for RTX 6000 Ada optimization
    """
    env_vars = {
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:4096,expandable_segments:True,garbage_collection_threshold:0.8",
        "CUDA_LAUNCH_BLOCKING": "0",
        "TRANSFORMERS_NO_ADVISORY_WARNINGS": "1",
        "DISABLE_FLASH_ATTENTION": "1",
        "TOKENIZERS_PARALLELISM": "false"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        logger.info(f"Set {key}={value}")


def check_dependencies() -> bool:
    """
    Check if required dependencies are installed and offer installation.
    
    This function verifies the presence of all required Python packages
    needed for the Enhanced Anti-Hallucination RAG System. If packages
    are missing, it offers to install them automatically.
    
    Returns:
        bool: True if all dependencies are available, False otherwise
        
    Required Packages:
        - streamlit: Web interface framework
        - torch: PyTorch deep learning framework
        - transformers: Hugging Face transformers
        - sentence-transformers: Sentence embedding models
        - qdrant-client: Vector database client
        - numpy: Numerical computing
        - nest-asyncio: Async support for Streamlit
        - psutil: System utilities
        - fastembed: Fast embedding generation
        
    Example:
        >>> if check_dependencies():
        ...     print("All dependencies available")
        ... else:
        ...     print("Missing dependencies found")
    """
    required_packages = [
        'streamlit', 'torch', 'transformers', 'sentence-transformers', 
        'qdrant-client', 'numpy', 'nest-asyncio', 'psutil', 'fastembed'
    ]
    
    logger.info("?? Checking dependencies...")
    missing_packages = []
    
    for package in required_packages:
        try:
            # Handle special cases for package imports
            if package == 'sentence-transformers':
                import sentence_transformers
            elif package == 'qdrant-client':
                import qdrant_client
            elif package == 'nest-asyncio':
                import nest_asyncio
            else:
                __import__(package.replace('-', '_'))
            logger.info(f"? {package} is installed")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"? {package} is missing")
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        install = input("Do you want to install missing packages? (y/n): ")
        if install.lower() == 'y':
            logger.info("Installing missing packages...")
            for package in missing_packages:
                logger.info(f"Installing {package}...")
                try:
                    subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
                    logger.info(f"? Successfully installed {package}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"? Failed to install {package}: {e}")
        else:
            logger.error("Cannot proceed without required packages")
            return False
    
    return True


def check_cuda() -> bool:
    """
    Check CUDA availability and RTX 6000 Ada GPU detection.
    
    This function detects available CUDA devices, identifies RTX 6000 Ada
    GPUs, and reports memory status. It provides specific optimizations
    recommendations based on detected hardware.
    
    Returns:
        bool: True if CUDA is available, False if only CPU is available
        
    GPU Detection:
        - RTX 6000 Ada: Optimal configuration detected
        - High-memory GPU (>40GB): RTX 6000 Ada optimizations applied
        - Standard GPU (<40GB): Performance warnings issued
        - No CUDA: CPU fallback with performance limitations
        
    Example:
        >>> if check_cuda():
        ...     print("CUDA GPU detected and configured")
        ... else:
        ...     print("CPU-only mode - limited performance")
    """
    logger.info("?? Checking CUDA and GPU status...")
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                free_memory = total_memory - allocated
                
                logger.info(f"? GPU {i}: {device_name}")
                logger.info(f"   Total VRAM: {total_memory:.1f}GB")
                logger.info(f"   Free VRAM: {free_memory:.1f}GB")
                
                # Check if it's RTX 6000 Ada
                if "RTX 6000" in device_name or "A6000" in device_name:
                    logger.info(f"?? RTX 6000 Ada detected! System optimized for {total_memory:.0f}GB VRAM")
                elif total_memory > 40:
                    logger.info(f"?? High-memory GPU detected ({total_memory:.0f}GB) - using RTX 6000 Ada optimizations")
                else:
                    logger.warning(f"?? GPU has {total_memory:.0f}GB VRAM - consider adjusting batch sizes")
            
            return True
        else:
            logger.warning("? CUDA not available - will use CPU (performance will be limited)")
            return False
    except ImportError:
        logger.error("? PyTorch not installed")
        return False


def check_files() -> bool:
    """
    Check if required and optional system files exist.
    
    This function verifies the presence of all critical system files
    needed for the Enhanced Anti-Hallucination RAG System operation.
    It distinguishes between required files (critical for operation)
    and optional files (enhance functionality but not essential).
    
    Returns:
        bool: True if all required files exist, False otherwise
        
    Required Files:
        - config.py: System configuration
        - pipeline.py: Main RAG pipeline
        - generator.py: Text generation component
        - retriever.py: Document retrieval component
        - embedding.py: Embedding model interface
        - streamlit_app.py: Web interface
        
    Optional Files:
        - test_scenarios.py: Test suite
        - hallucination_monitor.py: Monitoring system
        - memory_cleanup.py: Memory management
        - rtx6000_memory_manager.py: GPU memory manager
        - timing_utils.py: Performance monitoring
        - torch_fix.py: PyTorch compatibility fixes
        
    Example:
        >>> if check_files():
        ...     print("All required files present")
        ... else:
        ...     print("Missing critical files - system cannot start")
    """
    required_files = [
        'config.py', 'pipeline.py', 'generator.py', 'retriever.py', 
        'embedding.py', 'streamlit_app.py'
    ]
    
    optional_files = [
        'test_scenarios.py', 'hallucination_monitor.py', 'memory_cleanup.py',
        'rtx6000_memory_manager.py', 'timing_utils.py', 'torch_fix.py'
    ]
    
    logger.info("?? Checking required files...")
    missing_files = []
    found_files = []
    
    for file in required_files:
        if os.path.exists(file):
            logger.info(f"? {file} found")
            found_files.append(file)
        else:
            missing_files.append(file)
            logger.error(f"? {file} missing")
    
    # Check optional files
    logger.info("?? Checking optional files...")
    for file in optional_files:
        if os.path.exists(file):
            logger.info(f"? {file} found")
        else:
            logger.info(f"?? {file} not found (optional)")
    
    if missing_files:
        logger.error(f"Missing required files: {missing_files}")
        logger.error("Please ensure all enhanced files are in the current directory")
        
        # Suggest file creation
        print("\n?? Missing files need to be created:")
        for file in missing_files:
            print(f"   - {file}")
        
        return False
    
    return True


def check_qdrant_connection() -> bool:
    """
    Check Qdrant vector database connection and collection status.
    
    This function verifies connectivity to the Qdrant vector database,
    lists available collections, and ensures the system can access
    the document embeddings needed for retrieval operations.
    
    Returns:
        bool: True if Qdrant is accessible, False otherwise
        
    Connection Details:
        - Reads host and port from config.py
        - Lists all available collections
        - Reports collection count and names
        - Handles connection failures gracefully
        
    Example:
        >>> if check_qdrant_connection():
        ...     print("Qdrant database accessible")
        ... else:
        ...     print("Qdrant connection failed - check server status")
    """
    logger.info("?? Checking Qdrant connection...")
    try:
        from config import QDRANT_HOST, QDRANT_PORT
        from qdrant_client import QdrantClient
        
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections()
        
        logger.info(f"? Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        logger.info(f"?? Found {len(collections.collections)} collections")
        
        for collection in collections.collections:
            logger.info(f"   - {collection.name}")
        
        return True
        
    except Exception as e:
        logger.error(f"? Qdrant connection failed: {e}")
        logger.error("Please ensure Qdrant is running and accessible")
        return False


def run_tests() -> bool:
    """
    Run comprehensive anti-hallucination test suite.
    
    This function executes the complete test suite to validate the
    anti-hallucination capabilities of the Enhanced RAG System.
    It runs various test scenarios, evaluates performance, and
    generates detailed reports.
    
    Returns:
        bool: True if tests pass with acceptable score, False otherwise
        
    Test Categories:
        - Factual accuracy tests
        - Hallucination detection tests
        - Out-of-scope query handling
        - Context relevance validation
        - Response quality assessment
        
    Scoring:
        - >80%: Excellent performance
        - 60-80%: Acceptable performance
        - <60%: Requires improvement
        
    Output:
        - Detailed test results to console
        - JSON results file with timestamp
        - Performance metrics and recommendations
        
    Example:
        >>> if run_tests():
        ...     print("All tests passed - system ready for production")
        ... else:
        ...     print("Tests failed - review results for improvements")
    """
    logger.info("?? Running anti-hallucination tests...")
    try:
        # Check if test file exists
        if not os.path.exists('test_scenarios.py'):
            logger.error("? test_scenarios.py not found")
            return False
        
        import asyncio
        from test_scenarios import run_comprehensive_tests, print_test_summary
        from pipeline import EnhancedRAGPipeline
        
        logger.info("Initializing Enhanced RAG Pipeline...")
        pipeline = EnhancedRAGPipeline()
        
        logger.info("Running comprehensive test suite...")
        results = asyncio.run(run_comprehensive_tests(pipeline))
        
        # Print detailed summary
        print("\n" + "=" * 60)
        print_test_summary(results)
        print("=" * 60)
        
        # Save results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"test_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"?? Test results saved to {results_file}")
        
        pass_rate = results['summary']['overall_score']
        if pass_rate > 0.8:
            logger.info(f"? Excellent! Tests passed with {pass_rate:.1%} score")
            return True
        elif pass_rate > 0.6:
            logger.warning(f"?? Tests passed with {pass_rate:.1%} score (acceptable)")
            return True
        else:
            logger.error(f"? Tests failed with {pass_rate:.1%} score")
            return False
        
    except Exception as e:
        logger.error(f"? Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_quick_test() -> bool:
    """
    Run a quick test to verify basic system functionality.
    
    This function performs a rapid system validation by executing
    a single test query through the Enhanced RAG Pipeline. It
    checks basic functionality, response generation, and
    hallucination detection without running the full test suite.
    
    Returns:
        bool: True if quick test passes, False otherwise
        
    Test Process:
        1. Initialize Enhanced RAG Pipeline
        2. Execute sample query with standard parameters
        3. Validate response generation
        4. Check confidence scoring
        5. Verify hallucination risk assessment
        
    Metrics Reported:
        - Confidence level (HIGH/MEDIUM/LOW)
        - Number of retrieved contexts
        - Response length in characters
        - Hallucination risk level
        
    Example:
        >>> if run_quick_test():
        ...     print("Quick test passed - basic functionality working")
        ... else:
        ...     print("Quick test failed - check system configuration")
    """
    logger.info("? Running quick system test...")
    try:
        from pipeline import EnhancedRAGPipeline
        import asyncio
        
        pipeline = EnhancedRAGPipeline()
        
        # Test query
        test_query = "What parameters does the submit method accept?"
        
        logger.info(f"Testing with query: '{test_query}'")
        result = asyncio.run(pipeline.process_query(
            query=test_query,
            use_hybrid=True,
            use_reranker=True,
            top_k=3
        ))
        
        # Check results
        confidence = result.get('confidence_level', 'UNKNOWN')
        context_count = result.get('context_count', 0)
        response_length = len(result.get('response', ''))
        
        logger.info(f"? Quick test completed:")
        logger.info(f"   Confidence: {confidence}")
        logger.info(f"   Contexts: {context_count}")
        logger.info(f"   Response length: {response_length} chars")
        
        # Check for hallucination warnings
        hallucination_check = result.get('hallucination_check', {})
        risk_level = hallucination_check.get('risk_level', 'LOW')
        
        if risk_level == 'LOW':
            logger.info("? No hallucination risk detected")
        else:
            logger.warning(f"?? Hallucination risk: {risk_level}")
        
        return True
        
    except Exception as e:
        logger.error(f"? Quick test failed: {e}")
        return False


def run_streamlit() -> bool:
    """
    Run the Streamlit web application with optimized settings.
    
    This function launches the Streamlit web interface for the
    Enhanced Anti-Hallucination RAG System with configuration
    optimized for performance and user experience.
    
    Returns:
        bool: True if Streamlit starts successfully, False otherwise
        
    Streamlit Configuration:
        - Port: 8501 (default)
        - Address: 0.0.0.0 (accessible from network)
        - Headless: False (browser auto-open enabled)
        - CORS: Disabled for local development
        - Usage stats: Disabled for privacy
        - XSRF protection: Disabled for local use
        
    User Interface Features:
        - Interactive chat interface
        - Real-time confidence scoring
        - Document source display
        - Query enhancement options
        - Response formatting controls
        - Conversation history management
        
    Example:
        >>> if run_streamlit():
        ...     print("Streamlit application started successfully")
        ...     # Browser opens to http://localhost:8501
        ... else:
        ...     print("Failed to start Streamlit application")
    """
    logger.info("?? Starting Enhanced Anti-Hallucination RAG System...")
    
    # Check if streamlit_app.py exists
    if not os.path.exists('streamlit_app.py'):
        logger.error("? streamlit_app.py not found")
        return False
    
    try:
        logger.info("?? Starting Streamlit server...")
        print("\n" + "=" * 60)
        print("??? Enhanced Anti-Hallucination RAG System")
        print("?? RTX 6000 Ada Optimized")
        print("=" * 60)
        print("?? Streamlit will open in your browser shortly...")
        print("?? Access URL: http://localhost:8501")
        print("?? Press Ctrl+C to stop the server")
        print("=" * 60 + "\n")
        
        # Run Streamlit with optimized settings
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ])
        
        return True
        
    except KeyboardInterrupt:
        logger.info("?? Streamlit application stopped by user")
        print("\n? Application stopped successfully")
        return True
    except FileNotFoundError:
        logger.error("? Streamlit not found. Please install: pip install streamlit")
        return False
    except Exception as e:
        logger.error(f"? Error running Streamlit: {e}")
        return False


def show_system_info() -> bool:
    """
    Show detailed system information and component status.
    
    This function gathers and displays comprehensive information
    about the Enhanced Anti-Hallucination RAG System including
    configuration details, component status, GPU memory usage,
    and available collections.
    
    Returns:
        bool: True if system info gathered successfully, False otherwise
        
    Information Displayed:
        - Configuration summary from config.py
        - LLM and embedding model details
        - Available document collections
        - Component import status
        - GPU memory allocation and usage
        - System hardware specifications
        
    Component Status Check:
        - EnhancedRAGPipeline: Main processing pipeline
        - EnhancedRAGGenerator: Text generation component
        - RAGRetriever: Document retrieval system
        - E5EmbeddingModel: Embedding generation
        - HallucinationMonitor: Risk assessment system
        
    Example:
        >>> if show_system_info():
        ...     print("System information displayed successfully")
        ... else:
        ...     print("Failed to gather system information")
    """
    logger.info("?? Gathering system information...")
    
    try:
        from config import print_config_summary, LLM_MODEL, EMBEDDING_MODEL, COLLECTIONS
        
        print("\n" + "=" * 60)
        print("??? SYSTEM INFORMATION")
        print("=" * 60)
        
        # Print config summary
        print_config_summary()
        
        print(f"\n?? Configuration Details:")
        print(f"   LLM Model: {LLM_MODEL}")
        print(f"   Embedding Model: {EMBEDDING_MODEL}")
        print(f"   Available Collections: {list(COLLECTIONS.keys())}")
        
        # Try to import main components
        print(f"\n?? Component Status:")
        components = [
            ('EnhancedRAGPipeline', 'pipeline'),
            ('EnhancedRAGGenerator', 'generator'), 
            ('RAGRetriever', 'retriever'),
            ('E5EmbeddingModel', 'embedding'),
            ('HallucinationMonitor', 'hallucination_monitor')
        ]
        
        for component_name, module_name in components:
            try:
                module = __import__(module_name)
                getattr(module, component_name)
                print(f"   ? {component_name}")
            except (ImportError, AttributeError) as e:
                print(f"   ? {component_name}: {e}")
        
        # Memory info
        try:
            import torch
            if torch.cuda.is_available():
                print(f"\n?? GPU Memory Status:")
                for i in range(torch.cuda.device_count()):
                    allocated = torch.cuda.memory_allocated(i) / (1024**3)
                    reserved = torch.cuda.memory_reserved(i) / (1024**3)
                    total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    print(f"   GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved, {total:.1f}GB total")
        except:
            pass
        
        print("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"? System info check failed: {e}")
        return False


def main() -> bool:
    """
    Main function to run the enhanced system with interactive menu.
    
    This function provides the primary entry point for the Enhanced
    Anti-Hallucination RAG System. It performs system checks, displays
    an interactive menu, and handles user choices for various system
    operations.
    
    Returns:
        bool: True if system ran successfully, False if critical errors occurred
        
    System Checks Performed:
        1. Dependency verification and installation
        2. Required file existence validation
        3. CUDA GPU detection and configuration
        4. Qdrant database connectivity test
        
    Interactive Menu Options:
        1. Run comprehensive anti-hallucination tests
        2. Start Streamlit web application
        3. Quick test then start Streamlit
        4. Show detailed system information
        5. Run quick functionality test
        6. Check monitoring dashboard
        7. Exit application
        
    System Requirements:
        - Minimum 2 out of 4 system checks must pass
        - All required files must be present
        - Python dependencies must be available
        
    Example:
        >>> if main():
        ...     print("System executed successfully")
        ... else:
        ...     print("System encountered critical errors")
    """
    print("\n" + "=" * 70)
    print("???  ENHANCED ANTI-HALLUCINATION RAG SYSTEM")
    print("??  RTX 6000 Ada (48GB VRAM) Optimized")
    print("?  Ultra-Conservative Temperature (0.05)")
    print("??  Multi-Layer Validation & Real-time Monitoring")
    print("=" * 70)
    
    # Setup environment
    setup_environment()
    
    # System checks
    logger.info("?? Running system checks...")
    
    checks_passed = 0
    total_checks = 4
    
    if check_dependencies():
        checks_passed += 1
    
    if check_files():
        checks_passed += 1
    
    if check_cuda():
        checks_passed += 1
    
    if check_qdrant_connection():
        checks_passed += 1
    
    print(f"\n?? System Check Results: {checks_passed}/{total_checks} passed")
    
    if checks_passed < 2:
        logger.error("? Critical system checks failed. Please fix issues before proceeding.")
        return False
    
    # Main menu loop
    while True:
        print("\n" + "=" * 50)
        print("??? WHAT WOULD YOU LIKE TO DO?")
        print("=" * 50)
        print("1. ?? Run comprehensive anti-hallucination tests")
        print("2. ?? Start Streamlit application")
        print("3. ? Run quick test then start Streamlit")
        print("4. ?? Show detailed system information") 
        print("5. ? Run quick functionality test")
        print("6. ?? Check monitoring dashboard")
        print("7. ?? Exit")
        print("=" * 50)
        
        choice = input("\nEnter your choice (1-7): ").strip()
        
        if choice == "1":
            logger.info("?? Running comprehensive tests...")
            if run_tests():
                logger.info("? All tests completed successfully!")
            else:
                logger.error("? Some tests failed. Check the results above.")
        
        elif choice == "2":
            logger.info("?? Starting Streamlit application...")
            run_streamlit()
        
        elif choice == "3":
            logger.info("? Running quick test first...")
            if run_quick_test():
                logger.info("? Quick test passed! Starting Streamlit...")
                run_streamlit()
            else:
                logger.error("? Quick test failed! Fix issues before running Streamlit.")
        
        elif choice == "4":
            show_system_info()
        
        elif choice == "5":
            run_quick_test()
        
        elif choice == "6":
            logger.info("?? Checking monitoring dashboard...")
            try:
                if os.path.exists('hallucination_monitor.py'):
                    from hallucination_monitor import hallucination_monitor
                    summary = hallucination_monitor.get_daily_summary()
                    recent_alerts = hallucination_monitor.get_recent_alerts(limit=5)
                    
                    print(f"\n?? Daily Summary:")
                    print(f"   Total Queries: {summary.get('total_alerts', 0)}")
                    print(f"   High Risk: {summary.get('high_risk_alerts', 0)}")
                    print(f"   Medium Risk: {summary.get('medium_risk_alerts', 0)}")
                    print(f"   Hallucination Rate: {summary.get('hallucination_rate', 0)*100:.1f}%")
                    
                    if recent_alerts:
                        print(f"\n?? Recent Alerts ({len(recent_alerts)}):")
                        for alert in recent_alerts[:3]:
                            print(f"   - {alert['risk_level']}: {alert['query'][:50]}...")
                    else:
                        print("\n? No recent alerts")
                else:
                    logger.warning("Monitoring not available (hallucination_monitor.py not found)")
            except Exception as e:
                logger.error(f"Monitoring check failed: {e}")
        
        elif choice == "7":
            logger.info("?? Goodbye!")
            break
        
        else:
            logger.error("? Invalid choice. Please enter 1-7.")
    
    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n?? Process interrupted by user")
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"? Unexpected error: {e}")
        import traceback
        traceback.print_exc()