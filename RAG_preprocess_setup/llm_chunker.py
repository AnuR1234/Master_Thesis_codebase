#llm_chunker.py
import re
import os
import logging
import json
import asyncio
import nest_asyncio
import backoff
import time
from datetime import datetime, timedelta
import requests
from tenacity import retry, wait_exponential, stop_after_attempt
import signal
from contextlib import contextmanager
import functools
import concurrent.futures

# These should be imported from the library
from chunking_evaluation import BaseChunker
from chunking_evaluation.utils import openai_token_count
from chunking_evaluation.chunking import RecursiveTokenChunker
from gen_ai_hub.proxy.native.openai import AsyncOpenAI


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager for timeout handling"""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)  # Disable the alarm


class RateLimiter:
    """Basic rate limiter to prevent API overload"""
    def __init__(self, max_requests=20, period=60):
        self.max_requests = max_requests
        self.period = period
        self.counter = 0
        self.lock = None
        import threading
        self.lock = threading.Lock()
        self.last_reset = time.time()

    def wait(self):
        with self.lock:
            now = time.time()
            if now - self.last_reset > self.period:
                self.counter = 0
                self.last_reset = now
            self.counter += 1
            if self.counter > self.max_requests:
                sleep_time = self.period - (now - self.last_reset)
                if sleep_time > 0:
                    time.sleep(sleep_time + 1)  # Add buffer
                self.counter = 0
                self.last_reset = time.time()


class AdaptiveRateLimiter:
    """Rate limiter with adaptive backoff"""
    def __init__(self, initial_max_requests=15, period=60, backoff_factor=1.5, min_rate=5):
        self.max_requests = initial_max_requests
        self.current_max = initial_max_requests
        self.period = period
        self.counter = 0
        self.backoff_factor = backoff_factor
        self.min_rate = min_rate
        self.last_reset = time.time()
        self.consecutive_429 = 0
        self.global_cool_down_until = 0  # Track global cool down period
        import threading
        self.lock = threading.Lock()
        self.last_success_time = time.time()  # Track last successful request

    def wait(self):
        with self.lock:
            now = time.time()
            
            # Check if we're in a global cool down period after 429s
            if now < self.global_cool_down_until:
                sleep_time = self.global_cool_down_until - now
                logging.info(f"In cool down period, waiting {sleep_time:.1f}s before next request")
                time.sleep(sleep_time)
                now = time.time()
            
            # Normal rate limiting logic
            if now - self.last_reset > self.period:
                self.counter = 0
                self.last_reset = now
                
                # Gradually increase rate if no recent failures
                if self.consecutive_429 == 0 and self.current_max < self.max_requests:
                    self.current_max = min(self.current_max + 1, self.max_requests)
                    
            self.counter += 1
            if self.counter > self.current_max:
                sleep_time = self.period - (now - self.last_reset)
                if sleep_time > 0:
                    time.sleep(sleep_time + 1)  # Add buffer
                self.counter = 0
                self.last_reset = time.time()
    
    def report_429(self):
        """Report a 429 error and adjust rate limits accordingly"""
        with self.lock:
            self.consecutive_429 += 1
            
            # Set progressively longer cool down periods based on consecutive failures
            cooldown_time = min(15 * self.consecutive_429, 120)  # Up to 2 minutes
            self.global_cool_down_until = time.time() + cooldown_time
            
            # Reduce rate limit after consecutive 429s
            self.current_max = max(self.min_rate, int(self.current_max / self.backoff_factor))
            logging.error(f"Rate limit exceeded, cooling down for {cooldown_time}s and reducing to {self.current_max} requests per {self.period}s")
                
    def report_success(self):
        """Report a successful call to reset consecutive failures"""
        with self.lock:
            if self.consecutive_429 > 0:
                self.consecutive_429 = 0
                logging.info(f"Rate limit recovery: restored to {self.current_max} requests per {self.period}s")
            self.last_success_time = time.time()


# Simple in-memory cache for LLM responses
class ResponseCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        self.lock = None
        import threading
        self.lock = threading.Lock()
    
    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None
    
    def set(self, key, value):
        with self.lock:
            if len(self.cache) >= self.max_size:
                # Remove a random item
                self.cache.pop(next(iter(self.cache)), None)
            self.cache[key] = value
    
    def get_stats(self):
        with self.lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) * 100 if total > 0 else 0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.1f}%",
                "size": len(self.cache)
            }


class AICoreAuth:
    """Authentication handler for SAP AI Core"""
    def __init__(self):
        self.auth_url = os.getenv('AICORE_AUTH_URL')
        self.client_id = os.getenv('AICORE_CLIENT_ID')
        self.client_secret = os.getenv('AICORE_CLIENT_SECRET')
        self.token = None
        self.token_expiry = None
        import threading
        self.lock = threading.Lock()
    
    def get_token(self):
        """Get a valid authentication token, refreshing if necessary"""
        with self.lock:
            # Check if we have a valid token
            if self.token and self.token_expiry and datetime.now() < self.token_expiry:
                return self.token
            
            # Get a new token
            try:
                logging.info("Requesting new AI Core authentication token...")
                response = requests.post(
                    self.auth_url,
                    data={
                        'grant_type': 'client_credentials',
                        'client_id': self.client_id,
                        'client_secret': self.client_secret
                    },
                    headers={
                        'Content-Type': 'application/x-www-form-urlencoded'
                    }
                )
                
                if response.status_code == 200:
                    token_data = response.json()
                    self.token = token_data.get('access_token')
                    # Set expiry 5 minutes before actual expiry to be safe
                    expires_in = token_data.get('expires_in', 3600)
                    self.token_expiry = datetime.now() + timedelta(seconds=expires_in - 300)
                    logging.info((f"Token obtained, valid until {self.token_expiry}"))
                    return self.token
                else:
                    logging.error(f"Failed to obtain token: {response.status_code} - {response.text}")
                    return None
            except Exception as e:
                logging.error(f"Error obtaining authentication token: {e}")
                return None


class SAPLLMMultiDeploymentChunker(BaseChunker):
    """Extension of the BaseChunker to support multiple deployments"""
    
    def __init__(self, deployment_configs=None, api_version="2024-10-01-preview", timeout=120):
        super().__init__()
        
        try:
            # Create a fresh event loop - IMPORTANT: This needs to happen first
            self.loop = asyncio.new_event_loop()
            # Set this loop as the thread's event loop
            asyncio.set_event_loop(self.loop)
            
            # Apply nest_asyncio to allow nested event loops
            nest_asyncio.apply(self.loop)
            
            # Set default deployment configs if none provided
            if deployment_configs is None:
                default_id = os.getenv('AICORE_DEPLOYMENT_ID')
                if default_id:
                    self.deployment_configs = [{
                        "deployment_id": default_id,
                        "model_name": "gpt-4o"
                    }]
                else:
                    logging.warning("No default deployment ID found in environment variables")
                    self.deployment_configs = []
            else:
                self.deployment_configs = deployment_configs
                
            # Store other parameters
            self.api_version = api_version
            self.timeout = timeout
            
            # Initialize clients dictionary - will create clients on demand
            self.clients = {}
            
            # Initialize the recursive token chunker for initial text splitting
            self.splitter = RecursiveTokenChunker(
                chunk_size=100,  # Increased from 50 to reduce total chunks
                chunk_overlap=0,
                length_function=openai_token_count
            )
            
            # Initialize rate limiter with more aggressive settings
            self.rate_limiter = AdaptiveRateLimiter(
                initial_max_requests=15,  # Increased from 3
                period=60,
                backoff_factor=1.5,
                min_rate=5
            )
            self.rate_limit_failures = 0
            
            # Initialize response cache
            self.response_cache = ResponseCache(max_size=200)
            
            # Create a thread pool for parallel processing
            self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=5)
            
        except Exception as e:
            logging.error(f"Error initializing SAPLLMMultiDeploymentChunker: {e}")
            # Fallback to basic chunker
            self.splitter = RecursiveTokenChunker(
                chunk_size=50,
                chunk_overlap=0,
                length_function=openai_token_count
            )
            self.deployment_configs = None
            self.clients = {}
            self.loop = None
    
    def _get_client(self, deployment_id):
        """Get or create a client for the specified deployment_id"""
        if not deployment_id:
            raise ValueError("deployment_id cannot be None")
            
        if deployment_id in self.clients:
            return self.clients[deployment_id]
            
        # Find the configuration for this deployment
        config = next((cfg for cfg in self.deployment_configs 
                       if cfg["deployment_id"] == deployment_id), None)
        
        if not config:
            raise ValueError(f"No configuration found for deployment_id: {deployment_id}")
        
        # Create an auth instance
        auth = AICoreAuth()
        token = auth.get_token()
        
        if not token:
            raise ValueError("Failed to obtain authentication token")
        
        # Initialize client for this deployment
        client = AsyncOpenAI(
            api_version=self.api_version,
            base_url=os.getenv('AICORE_BASE_URL'),
            api_key="dummy",  # Not actually used with SAP AI Core
            default_headers={
                "Authorization": f"Bearer {token}",
                "AI-Resource-Group": os.getenv('AICORE_RESOURCE_GROUP', 'default'),
                "AI-Deployment-ID": deployment_id
            }
        )
        
        # Store in the clients dictionary
        self.clients[deployment_id] = {
            "client": client,
            "auth": auth,
            "model_name": config["model_name"]
        }
        
        return self.clients[deployment_id]
    
    async def _make_llm_call_async(self, deployment_id, system_prompt, messages):
        """Make an async call to the LLM API with the specified deployment"""
        try:
            # Get or create client for this deployment
            client_info = self._get_client(deployment_id)
            client = client_info["client"]
            model_name = client_info["model_name"]
            auth = client_info["auth"]
            
            # Refresh the token if needed
            token = auth.get_token()
            
            # Update the headers with the fresh token
            client.default_headers.update({
                "Authorization": f"Bearer {token}",
                "AI-Deployment-ID": deployment_id
            })
            
            # Call the API
            response = await client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    *messages
                ],
                max_tokens=800,  # Reduced from 1000
                temperature=0.1  # Lower temperature for more consistent responses
            )
            
            # Extract and return the response text
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error calling LLM API for deployment {deployment_id}: {str(e)}")
            raise
    
    def _make_llm_call(self, deployment_id, system_prompt, messages):
        """Synchronous wrapper around the async LLM API call"""
        if not self.loop.is_running():
            return self.loop.run_until_complete(self._make_llm_call_async(deployment_id, system_prompt, messages))
        else:
            # Create a future to run in the existing loop
            future = asyncio.run_coroutine_threadsafe(
                self._make_llm_call_async(deployment_id, system_prompt, messages), 
                self.loop
            )
            return future.result(timeout=self.timeout)
    
    @retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
    def _call_llm_with_timeout(self, deployment_id, *args, **kwargs):
        """Add retry logic with exponential backoff for a specific deployment"""
        try:
            # Wait for rate limiter
            self.rate_limiter.wait()
            
            # Generate cache key from the arguments (simplified)
            cache_key = str(args) + str(kwargs)
            cached_response = self.response_cache.get(cache_key)
            if cached_response:
                logging.info("Using cached LLM response")
                return cached_response
                
            # Make LLM call
            response = self._make_llm_call(deployment_id, *args, **kwargs)
            
            # Report success to rate limiter
            self.rate_limiter.report_success()
            
            # Cache the response
            self.response_cache.set(cache_key, response)
                
            return response
        except Exception as e:
            # Check for rate limit error and report to limiter
            if "429" in str(e) or "TooManyRequest" in str(e):
                self.rate_limiter.report_429()
                self.rate_limit_failures += 1
                logging.warning("Rate limit hit, backing off...")
                
            raise  # Re-raise all errors for retry handler
    
    async def aclose(self):
        """Properly close async resources"""
        close_tasks = []
        for client_info in self.clients.values():
            if client_info and "client" in client_info:
                close_tasks.append(client_info["client"].close())
        if close_tasks:
            await asyncio.gather(*close_tasks)
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)
    
    def close(self):
        """Close all resources properly"""
        if self.loop and not self.loop.is_closed():
            try:
                # Create and run a task to close all clients
                async def _close():
                    await self.aclose()
                
                # Run the close task in the loop
                self.loop.run_until_complete(_close())
            except Exception as e:
                logging.error(f"Error closing clients: {e}")
            
            # Now close the loop
            try:
                self.loop.close()
            except Exception as e:
                logging.error(f"Error closing loop: {e}")
            
        # Shutdown thread pool if it exists
        if hasattr(self, 'executor') and self.executor:
            try:
                self.executor.shutdown(wait=False)
            except Exception as e:
                logging.error(f"Error shutting down executor: {e}")
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up"""
        try:
            self.close()
        except:
            pass
    
    # Optimize by adding batched processing capability
    async def _batch_process_chunks_async(self, chunks, deployment_id):
        """Process multiple chunks in parallel using asyncio"""
        tasks = []
        for chunk in chunks:
            # Format the chunk for LLM input
            formatted_chunk = f"<|chunk_content|>\n{chunk}\n<|end_chunk_content|>"
            
            semantic_prompt = (
                "You are an assistant specialized in identifying natural break points in text. "
                "Examine the provided content and suggest where to split it into semantically coherent sections. "
                "Focus on topic transitions and respect section headers (lines starting with ##, ###, or ####). "
                "Your task is to create 2-3 meaningful chunks from this content. "
                "Respond with the exact positions where splits should occur, using character positions, "
                "in the format: 'split_at: 500, 1200' (actual numbers will vary based on content)."
            )
            
            message = {"role": "user", "content": formatted_chunk}
            
            # Create async task
            task = self._make_llm_call_async(deployment_id, semantic_prompt, [message])
            tasks.append(task)
        
        # Run all tasks concurrently
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def _batch_process_chunks(self, chunks, deployment_id):
        """Process multiple chunks concurrently"""
        return self.loop.run_until_complete(self._batch_process_chunks_async(chunks, deployment_id))
    
    def split_text(self, text, deployment_id=None):
        """
        Split text into semantic chunks using a specific deployment
        
        Args:
            text: The text to split
            deployment_id: The specific deployment ID to use for processing
        
        Returns:
            List of text chunks
        """
        # If no deployment_id specified, use the first one in configs
        if not deployment_id and self.deployment_configs:
            try:
                deployment_id = self.deployment_configs[0]["deployment_id"]
                logging.info(f"No deployment_id specified, using default: {deployment_id}")
            except (IndexError, KeyError) as e:
                logging.error(f"Failed to get default deployment_id: {e}")
                return self._fallback_chunking(text)
        
        if not deployment_id:
            logging.error("No deployment_id specified and no default found")
            return self._fallback_chunking(text)
        
        # Check if we've had too many rate limit failures recently
        if self.rate_limit_failures >= 5:
            logging.warning("Too many rate limit failures, falling back to rule-based chunking")
            return self._header_based_chunking(text)
        
        try:
            # First quick test to check if LLM is responsive
            cache_key = f"test_ping_{deployment_id}"
            cached_response = self.response_cache.get(cache_key)
            
            if cached_response:
                # Use cached test result
                llm_working = True
            else:
                # Test the LLM with timeout
                system_prompt = "You are a helpful assistant."
                messages = [{"role": "user", "content": "Please respond with the word 'working' and nothing else."}]
                
                test_response = self._call_llm_with_timeout(deployment_id, system_prompt, messages)
                llm_working = "working" in test_response.lower()
                
                # Cache the test result
                self.response_cache.set(cache_key, test_response)
            
            if llm_working:
                self.rate_limit_failures = 0
                
                # Analyze document complexity to determine chunking approach
                is_complex = len(text) > 3000 or text.count('##') > 3 or "Method Dependencies" in text
                
                if not is_complex:
                    # For simpler content, use rule-based chunking
                    return self._header_based_chunking(text)
                
                # Use token-based chunking for initial text splitting
                initial_chunks = self.splitter.split_text(text)
                
                # More aggressive content merging to reduce number of chunks
                max_merged_size = 5000  # Characters
                merged_chunks = []
                current_merged = ""
                
                for chunk in initial_chunks:
                    if len(current_merged) + len(chunk) < max_merged_size:
                        current_merged += chunk + " "
                    else:
                        if current_merged:
                            merged_chunks.append(current_merged.strip())
                        current_merged = chunk + " "
                
                if current_merged:
                    merged_chunks.append(current_merged.strip())
                
                # If we have a small number of merged chunks, process them individually
                if len(merged_chunks) <= 3:
                    final_chunks = []
                    
                    # Use simpler prompt for better performance
                    for chunk in merged_chunks:
                        semantic_prompt = (
                            "You are an assistant specialized in splitting text into meaningful sections. "
                            "The text will be used for document retrieval purposes. "
                            "Identify 2-3 natural break points that create coherent sections. "
                            "Always respect header boundaries (##, ###, ####) and keep them with their content. "
                            "Respond with position numbers where to split, like: 'split_at: 500, 1200'"
                        )
                        
                        # Format the chunk with some context
                        chunk_message = {
                            "role": "user", 
                            "content": f"Split this text at logical points: {chunk[:30]}...\n\n{chunk}"
                        }
                        
                        # Call LLM for semantic chunking
                        split_response = self._call_llm_with_timeout(deployment_id, semantic_prompt, [chunk_message])
                        
                        # Parse split points
                        if 'split_at:' in split_response:
                            split_points = []
                            splits_text = split_response.split('split_at:')[1].strip()
                            # Extract numbers and handle various formatting
                            for part in re.findall(r'\d+', splits_text):
                                try:
                                    point = int(part)
                                    if 0 < point < len(chunk):  # Validate point is in range
                                        split_points.append(point)
                                except ValueError:
                                    continue
                            
                            # Sort and validate split points
                            split_points.sort()
                            
                            # Create chunks based on split points
                            last_pos = 0
                            for pos in split_points:
                                # Find a clean break point near suggested position
                                # Try to find paragraph break
                                para_break = chunk.rfind('\n\n', last_pos, pos)
                                if para_break != -1 and para_break > last_pos + 100:
                                    clean_pos = para_break + 2  # Include the double newline
                                else:
                                    # Try to find sentence break
                                    sentence_break = max(
                                        chunk.rfind('. ', last_pos, pos),
                                        chunk.rfind('.\n', last_pos, pos)
                                    )
                                    if sentence_break != -1 and sentence_break > last_pos + 50:
                                        clean_pos = sentence_break + 2
                                    else:
                                        # Use exact position as fallback
                                        clean_pos = pos
                                
                                final_chunks.append(chunk[last_pos:clean_pos].strip())
                                last_pos = clean_pos
                            
                            # Add the last chunk
                            if last_pos < len(chunk):
                                final_chunks.append(chunk[last_pos:].strip())
                        else:
                            # If splitting failed, add the entire chunk
                            final_chunks.append(chunk)
                    
                    return final_chunks
                else:
                    # For larger documents, batch process chunks in parallel
                    logging.info(f"Batch processing {len(merged_chunks)} chunks in parallel")
                    
                    # Process chunks in batches to avoid overwhelming the LLM service
                    batch_size = 3
                    all_results = []
                    
                    for i in range(0, len(merged_chunks), batch_size):
                        batch = merged_chunks[i:i+batch_size]
                        try:
                            # Rate limit at batch level
                            self.rate_limiter.wait()
                            
                            batch_results = self._batch_process_chunks(batch, deployment_id)
                            all_results.extend(batch_results)
                            
                            self.rate_limiter.report_success()
                        except Exception as e:
                            logging.error(f"Error in batch processing: {e}")
                            # Fall back to processing these chunks individually
                            for chunk in batch:
                                try:
                                    self.rate_limiter.wait()
                                    result = self._call_llm_with_timeout(deployment_id, 
                                                                      "You are a helpful assistant.",
                                                                      [{"role": "user", "content": f"Split this text at logical points: {chunk[:50]}..."}])
                                    all_results.append(result)
                                    self.rate_limiter.report_success()
                                except Exception as inner_e:
                                    logging.error(f"Error processing individual chunk: {inner_e}")
                                    all_results.append(None)
                    
                    # Process the results to create final chunks
                    final_chunks = []
                    
                    for i, (chunk, result) in enumerate(zip(merged_chunks, all_results)):
                        if result is None or isinstance(result, Exception):
                            # If processing failed, add the whole chunk
                            final_chunks.append(chunk)
                            continue
                            
                        # Parse the split points
                        if isinstance(result, str) and 'split_at:' in result:
                            split_points = []
                            splits_text = result.split('split_at:')[1].strip()
                            for part in re.findall(r'\d+', splits_text):
                                try:
                                    point = int(part)
                                    if 0 < point < len(chunk):
                                        split_points.append(point)
                                except ValueError:
                                    continue
                            
                            # Create chunks based on split points
                            last_pos = 0
                            split_points.sort()
                            
                            for pos in split_points:
                                # Find clean break points
                                para_break = chunk.rfind('\n\n', last_pos, pos)
                                if para_break != -1 and para_break > last_pos + 100:
                                    clean_pos = para_break + 2
                                else:
                                    clean_pos = pos
                                    
                                final_chunks.append(chunk[last_pos:clean_pos].strip())
                                last_pos = clean_pos
                            
                            # Add the last chunk
                            if last_pos < len(chunk):
                                final_chunks.append(chunk[last_pos:].strip())
                        else:
                            # Invalid response format, use the whole chunk
                            final_chunks.append(chunk)
                    
                    # Log cache stats periodically
                    if len(final_chunks) % 10 == 0:
                        cache_stats = self.response_cache.get_stats()
                        logging.info(f"Cache stats: {cache_stats}")
                    
                    return final_chunks
            else:
                logging.warning("LLM test response indicates issues, using header-based chunking")
                return self._header_based_chunking(text)
        except Exception as e:
            if "429" in str(e) or "TooManyRequest" in str(e):
                self.rate_limiter.report_429()
                self.rate_limit_failures += 1
                logging.warning(f"Rate limit hit during text splitting: {e}")
            else:
                logging.error(f"Error in split_text: {e}")
            
            return self._header_based_chunking(text)
    
    def _header_based_chunking(self, text):
        """Fallback chunking method that respects header boundaries"""
        chunks = []
        
        # Find all headers (## or ####)
        header_pattern = re.compile(r'(^|\n)(#{2,4}\s+[^#\n]+)(\n|$)', re.MULTILINE)
        header_matches = list(header_pattern.finditer(text))
        
        if not header_matches:
            # If no headers, fall back to paragraph-based chunking
            return self._fallback_chunking(text)
        
        # Process content by headers
        last_pos = 0
        current_chunk = ""
        
        for i, match in enumerate(header_matches):
            # If this is the first header and there's content before it,
            # add that content as a separate chunk
            if i == 0 and match.start() > 0:
                chunks.append(text[:match.start()].strip())
            
            header = match.group(2)
            header_start = match.start()
            
            # If we've accumulated content, check if adding this section would exceed the limit
            if current_chunk:
                # If this is a level 2 header (##) or we'd exceed the size limit,
                # finish the current chunk before this header
                if header.startswith("##") or len(current_chunk) + (match.start() - last_pos) > 3000:  # Increased from 2000
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
            
            # Add the header to the current chunk
            current_chunk += text[header_start:match.end()]
            last_pos = match.end()
            
            # If this is the last header, add all remaining content
            if i == len(header_matches) - 1:
                current_chunk += text[match.end():].strip()
                chunks.append(current_chunk.strip())
            else:
                # Otherwise, add content until the next header
                next_header_start = header_matches[i + 1].start()
                current_chunk += text[match.end():next_header_start]
                last_pos = next_header_start
        
        # If we have accumulated content that hasn't been added to chunks, add it
        if current_chunk and current_chunk not in chunks:
            chunks.append(current_chunk.strip())
        
        # If chunks are still too large, further split them
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > 3000:  # Increased from 2000
                # Split by paragraphs, preserving headers
                sub_chunks = self._split_large_chunk_by_paragraphs(chunk, max_size=3000)  # Increased from default
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks