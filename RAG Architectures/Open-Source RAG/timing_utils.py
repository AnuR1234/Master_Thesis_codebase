# -*- coding: utf-8 -*-
"""
Utilities for timing and performance tracking in the RAG pipeline.

This module provides comprehensive timing functionality for measuring and analyzing
performance across different components of a RAG (Retrieval-Augmented Generation)
pipeline. It includes both manual timing controls and decorators for automatic
function timing.

Classes:
    TimingStats: Main class for tracking and reporting timing statistics

Functions:
    timed: Decorator for timing synchronous functions
    timed_async: Decorator for timing asynchronous functions

Global Variables:
    timing_stats: Global TimingStats instance for application-wide use
    logger: Configured logger for timing output
"""
import time
import logging
from functools import wraps
from typing import Dict, Any, Callable, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimingStats:
    """
    Class to track and report timing statistics throughout the RAG pipeline.
    
    This class provides functionality to start/stop named timers, collect timing
    data, and generate comprehensive statistics reports. It supports multiple
    concurrent timers and maintains historical data for analysis.
    
    Attributes:
        stats (Dict[str, List[float]]): Historical timing data by timer name
        current_timers (Dict[str, float]): Currently active timers with start times
    
    Example:
        >>> timer = TimingStats()
        >>> timer.start_timer("database_query")
        >>> # ... perform database operation ...
        >>> elapsed = timer.stop_timer("database_query")
        >>> print(f"Query took {elapsed:.2f} seconds")
    """
    
    def __init__(self):
        """
        Initialize timing stats tracker.
        
        Creates empty containers for timing statistics and active timers.
        """
        self.stats = {}
        self.current_timers = {}
    
    def start_timer(self, name: str):
        """
        Start a named timer.
        
        Records the current timestamp for the specified timer name. If a timer
        with the same name is already running, it will be overwritten.
        
        Args:
            name (str): Unique identifier for the timer
            
        Example:
            >>> timer.start_timer("vector_search")
        """
        self.current_timers[name] = time.time()
    
    def stop_timer(self, name: str) -> float:
        """
        Stop a named timer and record the elapsed time.
        
        Calculates the elapsed time since the timer was started, stores it in
        the statistics, and removes the timer from active timers.
        
        Args:
            name (str): Name of the timer to stop
            
        Returns:
            float: Elapsed time in seconds (0.0 if timer was not started)
            
        Example:
            >>> elapsed = timer.stop_timer("vector_search")
            >>> print(f"Vector search completed in {elapsed:.4f}s")
        """
        if name not in self.current_timers:
            logger.warning(f"Timer '{name}' was not started")
            return 0.0
        
        elapsed = time.time() - self.current_timers[name]
        
        # Store the timing
        if name not in self.stats:
            self.stats[name] = []
        
        self.stats[name].append(elapsed)
        
        # Remove from current timers
        del self.current_timers[name]
        
        # Log the timing
        logger.info(f"TIMING: {name} took {elapsed:.4f} seconds")
        
        return elapsed
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get the current timing statistics.
        
        Calculates comprehensive statistics for all recorded timers including
        count, total time, average, minimum, maximum, and most recent timing.
        
        Returns:
            Dict[str, Any]: Dictionary with timing statistics for each timer.
                Each timer entry contains:
                - count (int): Number of recorded timings
                - total (float): Sum of all timings in seconds
                - average (float): Mean timing in seconds
                - min (float): Minimum recorded timing in seconds
                - max (float): Maximum recorded timing in seconds
                - last (float): Most recent timing in seconds
                
        Example:
            >>> stats = timer.get_stats()
            >>> print(f"Average query time: {stats['database_query']['average']:.2f}s")
        """
        result = {}
        
        for name, times in self.stats.items():
            if times:
                result[name] = {
                    "count": len(times),
                    "total": sum(times),
                    "average": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                    "last": times[-1]
                }
        
        return result
    
    def log_stats(self):
        """
        Log the current timing statistics to the configured logger.
        
        Outputs a formatted summary of all timing statistics including count,
        total time, averages, and min/max values for each recorded timer.
        
        Example:
            >>> timer.log_stats()
            # Outputs formatted statistics to logger at INFO level
        """
        stats = self.get_stats()
        
        logger.info("===== TIMING STATISTICS =====")
        
        for name, metrics in stats.items():
            logger.info(f"{name}:")
            logger.info(f"  Count: {metrics['count']}")
            logger.info(f"  Total: {metrics['total']:.4f}s")
            logger.info(f"  Average: {metrics['average']:.4f}s")
            logger.info(f"  Min: {metrics['min']:.4f}s")
            logger.info(f"  Max: {metrics['max']:.4f}s")
            logger.info(f"  Last: {metrics['last']:.4f}s")
        
        logger.info("=============================")

    def reset(self):
        """
        Reset all timing statistics.
        
        Clears all recorded timing data and stops any currently active timers.
        This is useful for starting fresh measurements or clearing data between
        test runs.
        
        Example:
            >>> timer.reset()  # Clear all timing data
        """
        self.stats = {}
        self.current_timers = {}

    def __str__(self) -> str:
        """
        Return string representation of timing stats.
        
        Provides a human-readable summary of all timing statistics suitable
        for printing or logging.
        
        Returns:
            str: Formatted string containing timing statistics
            
        Example:
            >>> print(timer)
            # Outputs formatted timing statistics
        """
        result = "Timing Statistics:\n"
        stats = self.get_stats()
        
        for name, metrics in stats.items():
            result += f"{name}:\n"
            result += f"  Count: {metrics['count']}\n"
            result += f"  Total: {metrics['total']:.4f}s\n"
            result += f"  Average: {metrics['average']:.4f}s\n"
            result += f"  Last: {metrics['last']:.4f}s\n"
        
        return result


# Create a global instance for use throughout the application
timing_stats = TimingStats()


def timed(name: Optional[str] = None):
    """
    Decorator to time a function and log its execution time.
    
    This decorator automatically starts a timer when the function is called
    and stops it when the function returns, recording the elapsed time in
    the global timing_stats instance.
    
    Args:
        name (Optional[str]): Optional name for the timer. If not provided,
            uses the function's __name__ attribute.
    
    Returns:
        Callable: Decorated function that will be automatically timed
        
    Example:
        >>> @timed("database_operation")
        ... def query_database():
        ...     # Database operation here
        ...     pass
        
        >>> @timed()  # Uses function name as timer name
        ... def process_documents():
        ...     # Document processing here
        ...     pass
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            timer_name = name or func.__name__
            timing_stats.start_timer(timer_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                timing_stats.stop_timer(timer_name)
        return wrapper
    return decorator


def timed_async(name: Optional[str] = None):
    """
    Decorator to time an async function and log its execution time.
    
    This decorator automatically starts a timer when the async function is
    called and stops it when the function completes (including awaiting),
    recording the elapsed time in the global timing_stats instance.
    
    Args:
        name (Optional[str]): Optional name for the timer. If not provided,
            uses the function's __name__ attribute.
    
    Returns:
        Callable: Decorated async function that will be automatically timed
        
    Example:
        >>> @timed_async("async_vector_search")
        ... async def search_vectors():
        ...     # Async vector search operation
        ...     pass
        
        >>> @timed_async()  # Uses function name as timer name
        ... async def generate_embeddings():
        ...     # Async embedding generation
        ...     pass
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            timer_name = name or func.__name__
            timing_stats.start_timer(timer_name)
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                timing_stats.stop_timer(timer_name)
        return wrapper
    return decorator