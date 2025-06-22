import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import asyncio
import requests
import random
import time
import json
from dotenv import load_dotenv

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import SparseVector, Distance
from fastembed import SparseTextEmbedding

# Constants
QDRANT_HOST = "localhost"  # If running Docker on the same machine
QDRANT_PORT = 6333  # Default HTTP API port
COLLECTION_NAME = "SAP_ABAP_CODE_DOCUMENTATION_3_CLASSES_TEST"
OPENAI_VECTOR_SIZE = 1536  # Size of OpenAI ada-002 embeddings
PROGRESS_FILE = "embedding_progress.json"  # File to track progress
BATCH_SIZE = 300  # Number of documents to process in each batch
MAX_CONCURRENT = 10  # Maximum concurrent API calls
BATCH_PAUSE = 10  # Seconds to pause between batches

# OpenAI embedding wrapper class with rate limiting handling
class OpenAIEmbeddingWrapper:
    def __init__(self, model_instances=None, model="text-embedding-ada-002"):
        if model_instances is None:
            model_instances = ["embed_1", "embed_2", "embed_3"]
        self.model_instances = model_instances
        self.model = model
        
        # Import here to avoid dependency if not using OpenAI
        try:
            from gen_ai_hub.proxy.native.openai import AsyncOpenAI
            self.AsyncOpenAI = AsyncOpenAI
            self.requests = requests
            # Load environment variables
            load_dotenv('rag.env')  # Update the path to your .env file
        except ImportError:
            print("Warning: gen_ai_hub, requests, or dotenv not installed. OpenAI embeddings won't work.")
        
        # Client cache
        self.client_cache = {}

    def get_auth_token(self):
        """Get authentication token from SAP AI Core"""
        auth_url = os.getenv('AICORE_AUTH_URL')
        client_id = os.getenv('AICORE_CLIENT_ID')
        client_secret = os.getenv('AICORE_CLIENT_SECRET')
        response = self.requests.post(
            auth_url,
            data={
                'grant_type': 'client_credentials',
                'client_id': client_id,
                'client_secret': client_secret
            },
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        if response.status_code == 200:
            token_data = response.json()
            return token_data.get('access_token')
        else:
            raise Exception(f"Auth failed: {response.status_code} - {response.text}")

    def get_client(self, model_instance):
        """Get or create client for a model instance"""
        if model_instance in self.client_cache:
            return self.client_cache[model_instance]
        # Get deployment ID for this instance
        deployment_id = os.getenv(f'AICORE_DEPLOYMENT_ID_{model_instance}')
        if not deployment_id:
            raise ValueError(f"No deployment ID found for {model_instance}")
        # Get auth token
        token = self.get_auth_token()
        # Create client
        client = self.AsyncOpenAI(
            base_url=os.getenv('AICORE_BASE_URL'),
            api_key="dummy", # Not used with SAP AI Core
            default_headers={
                "Authorization": f"Bearer {token}",
                "AI-Resource-Group": os.getenv('AICORE_RESOURCE_GROUP', 'default'),
                "AI-Deployment-ID": deployment_id
            }
        )
        # Cache the client
        self.client_cache[model_instance] = client
        return client

    async def _get_embedding_async(self, text, model_instance, max_retries=5):
        """Get embedding asynchronously with retries and backoff"""
        retries = 0
        while retries < max_retries:
            try:
                # Ensure text is a string and remove newlines
                if not isinstance(text, str):
                    raise ValueError("Expected text to be a string")
                text = text.replace("\n", " ")
                # Get client for this model instance
                client = self.get_client(model_instance)
                response = await client.embeddings.create(
                    input=text,
                    model=self.model
                )
                return response.data[0].embedding
            except Exception as e:
                if "TooManyRequest" in str(e) or "429" in str(e):
                    retries += 1
                    # Exponential backoff - wait longer after each retry
                    wait_time = 5 ** retries + random.uniform(0, 1)
                    print(f"Rate limited on {model_instance}. Retry {retries}/{max_retries} after {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                else:
                    # For non-rate-limit errors, try other instances
                    print(f"Error with {model_instance}: {e}")
                    for next_instance in self.model_instances:
                        if next_instance != model_instance:
                            try:
                                print(f"Retrying with {next_instance}")
                                embedding = await self._get_embedding_async(text, next_instance, max_retries)
                                return embedding
                            except Exception as e2:
                                print(f"Error on retry: {e2}")
                    return None
        
        print(f"Max retries exceeded for {model_instance}")
        return None

    async def _process_batch_async(self, texts, instances):
        """Process a batch of texts with controlled concurrency"""
        results = [None] * len(texts)
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        
        async def process_text(idx, text, instance):
            async with semaphore:
                return idx, await self._get_embedding_async(text, instance)
        
        # Create tasks for all texts
        tasks = []
        for i, text in enumerate(texts):
            instance = instances[i % len(instances)]
            tasks.append(process_text(i, text, instance))
        
        # Execute and collect results
        completed = 0
        for future in asyncio.as_completed(tasks):
            idx, embedding = await future
            results[idx] = embedding
            completed += 1
            if completed % 100 == 0 or completed == len(texts):
                print(f"  Completed {completed}/{len(texts)} embeddings ({completed/len(texts)*100:.1f}%)")
        
        return results

    def encode(self, texts):
        """
        Encode texts to embeddings with controlled concurrency.
        Args:
            texts: A string or list of strings to encode
        Returns:
            A list of embeddings (one per text)
        """
        # Ensure texts is always a list
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return []
            
        try:
            # Get the current event loop or create a new one if none exists
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Prepare instance assignments for each text
            instances = [self.model_instances[i % len(self.model_instances)] for i in range(len(texts))]
            
            # Process in batches with controlled concurrency
            print(f"Generating embeddings for {len(texts)} documents with max concurrency {MAX_CONCURRENT}...")
            start_time = time.time()
            
            embeddings = loop.run_until_complete(self._process_batch_async(texts, instances))
            
            elapsed = time.time() - start_time
            print(f"✓ Embeddings completed in {elapsed:.2f} seconds ({len(texts)/elapsed:.2f} docs/sec)")
            
        except Exception as e:
            print(f"Error in OpenAI encoding: {e}")
            embeddings = [None] * len(texts)
                
        return embeddings


class DocumentUploader:
    def __init__(self):
        """Initialize the document uploader for SAP ABAP code documentation"""
        print("Initializing document uploader for SAP ABAP code documentation...")
        
        # Initialize BM25 model
        try:
            print("Loading BM25 model...")
            self.sparse_model = SparseTextEmbedding("Qdrant/bm25")
            print("✓ BM25 model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading BM25 model: {e}")
            self.sparse_model = None
        
        # Initialize OpenAI model
        try:
            print("Loading OpenAI model...")
            self.openai_model = OpenAIEmbeddingWrapper()
            print("✓ OpenAI model wrapper initialized")
        except Exception as e:
            print(f"✗ Error loading OpenAI model: {e}")
            self.openai_model = None
        
        # Initialize Qdrant client
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            
    def initialize_qdrant(self):
        """Initialize Qdrant collection with proper vector configurations"""
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        
        # Check if collection exists and optionally recreate it
        recreate = True  # Set to True to recreate the collection with correct config
        if self.client.collection_exists(COLLECTION_NAME):
            if recreate:
                print(f"Deleting existing collection {COLLECTION_NAME}...")
                self.client.delete_collection(COLLECTION_NAME)
                print(f"✓ Collection deleted")
                time.sleep(1)  # Give server time to process
            else:
                print(f"✓ Collection {COLLECTION_NAME} already exists")
                return
        
        # Create collection with on-disk vector storage and HNSW index
        print(f"Creating collection {COLLECTION_NAME}...")
        try:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config={
                    "OpenAI": models.VectorParams(
                        size=OPENAI_VECTOR_SIZE,
                        distance=Distance.COSINE,
                        on_disk=True  # Store vectors on disk instead of in memory
                    )
                },
                sparse_vectors_config={
                    "BM25": models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
                optimizers_config=models.OptimizersConfigDiff(
                    memmap_threshold=10000  # Lower threshold for converting to memmap
                ),
                hnsw_config=models.HnswConfigDiff(
                    on_disk=True  # Store HNSW index on disk
                ),
                on_disk_payload=True  # Store payload on disk
            )
            print(f"✓ Collection {COLLECTION_NAME} created successfully")
            
            # Verify collection was created
            collection_info = self.client.get_collection(COLLECTION_NAME)
            print(f"Collection status: {collection_info.status}")
            print(f"Collection config: {collection_info.config}")
            
        except Exception as e:
            print(f"Error creating collection: {e}")
            raise
    
    def get_existing_ids(self):
        """Get IDs of documents already in Qdrant"""
        existing_ids = set()
        try:
            offset = 0
            limit = 1000
            while True:
                results = self.client.scroll(
                    collection_name=COLLECTION_NAME,
                    offset=offset,
                    limit=limit,
                    with_payload=False,
                    with_vectors=False
                )
                points = results[0]
                if not points:
                    break
                existing_ids.update(p.id for p in points)
                offset += limit
                print(f"  Found {len(existing_ids)} existing documents so far...")
            print(f"✓ Found {len(existing_ids)} existing documents in Qdrant")
        except Exception as e:
            print(f"Error checking existing documents: {e}")
        return existing_ids
    
    def load_progress(self):
        """Load progress from file"""
        if os.path.exists(PROGRESS_FILE):
            try:
                with open(PROGRESS_FILE, 'r') as f:
                    progress = json.load(f)
                print(f"✓ Loaded progress: {progress['processed_count']}/{progress['total_count']} documents processed")
                return progress
            except Exception as e:
                print(f"Error loading progress: {e}")
        return {"processed_count": 0, "last_index": -1, "total_count": 0}
    
    def save_progress(self, progress):
        """Save progress to file"""
        try:
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(progress, f)
            print(f"✓ Progress saved: {progress['processed_count']}/{progress['total_count']} documents")
        except Exception as e:
            print(f"Error saving progress: {e}")

    def process_sparse_vector(self, sparse_vec):
        """
        Process a sparse vector from fastembed into Qdrant SparseVector format
        
        Args:
            sparse_vec: Sparse vector from fastembed (could be dict, generator, etc.)
                
        Returns:
            SparseVector object for Qdrant
        """
        if sparse_vec is None:
            return None
            
        indices = []
        values = []
        
        try:
            # For generator objects (most common with newer fastembed versions)
            if hasattr(sparse_vec, '__iter__') and not isinstance(sparse_vec, dict):
                # Convert generator to list of tuples
                sparse_items = list(sparse_vec)
                for item in sparse_items:
                    if isinstance(item, tuple) and len(item) == 2:
                        idx, val = item
                        indices.append(int(idx))
                        values.append(float(val))
            # For dictionary-like objects
            elif isinstance(sparse_vec, dict) or hasattr(sparse_vec, 'items'):
                for idx, val in sparse_vec.items():
                    indices.append(int(idx))
                    values.append(float(val))
        except Exception as e:
            print(f"Error processing sparse vector: {e}")
            return None
        
        if not indices:  # Return None if no indices found
            return None
            
        return SparseVector(indices=indices, values=values)

    def embed_and_store_dataset(self, csv_path):
        """
        Process a CSV dataset, generate embeddings, and store in Qdrant with resumption
        
        Args:
            csv_path: Path to the CSV file containing the dataset
        """
        print(f"Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        print(f"Dataset loaded: {total_rows} rows")
        
        # Verify metadata columns
        document_column = 'doc'
        metadata_columns = ['title', 'code_snippet', 'length', 'filename']
        for col in [document_column] + metadata_columns:
            if col not in df.columns:
                print(f"Warning: Expected column '{col}' not found in dataset")
        
        # Get existing IDs in Qdrant
        print("Checking for existing documents in Qdrant...")
        existing_ids = self.get_existing_ids()
        
        # Load progress
        progress = self.load_progress()
        if progress["total_count"] == 0:
            progress["total_count"] = total_rows
        
        # Start from last processed index + 1
        start_index = progress["last_index"] + 1
        
        # Process in batches
        try:
            for batch_start in range(start_index, total_rows, BATCH_SIZE):
                batch_end = min(batch_start + BATCH_SIZE, total_rows)
                batch_number = batch_start // BATCH_SIZE + 1
                total_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
                
                print(f"\n===== Processing batch {batch_number}/{total_batches}: rows {batch_start} to {batch_end-1} =====")
                
                # Process batch with error handling
                try:
                    self._process_batch(df, batch_start, batch_end, existing_ids, progress)
                except KeyboardInterrupt:
                    print("\n⚠️ Process interrupted by user. Progress saved.")
                    return
                except Exception as e:
                    print(f"\n⚠️ Error processing batch {batch_number}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Save progress up to the last successful batch
                    progress["last_index"] = batch_start - 1
                    self.save_progress(progress)
                    print(f"⚠️ Progress saved. You can restart from batch {batch_number}.")
                    
                    # Ask whether to continue or abort
                    if input("Continue processing? (y/n): ").lower() != 'y':
                        return
                    
                # Pause between batches to avoid overwhelming the API
                if batch_end < total_rows:
                    print(f"Pausing for {BATCH_PAUSE} seconds before next batch...")
                    time.sleep(BATCH_PAUSE)
            
            print(f"\n===== Processing complete! =====")
            print(f"Processed {progress['processed_count']}/{progress['total_count']} documents")
        
        except Exception as e:
            print(f"\n⚠️ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            self.save_progress(progress)
            print(f"⚠️ Progress saved at index {progress['last_index']}. You can restart from there.")

    def _process_batch(self, df, batch_start, batch_end, existing_ids, progress):
        """Process a single batch of data"""
        document_column = 'doc'
        
        # Skip batch if all IDs already exist
        batch_ids = list(range(batch_start, batch_end))
        if all(id in existing_ids for id in batch_ids):
            print(f"Skipping batch - all documents already exist in Qdrant")
            progress["processed_count"] += len(batch_ids)
            progress["last_index"] = batch_end - 1
            self.save_progress(progress)
            return
        
        # Get batch of data
        batch_df = df.iloc[batch_start:batch_end].copy()
        
        # Extract the document column
        documents = batch_df[document_column].tolist()
        
        # Generate OpenAI embeddings
        print(f"Generating OpenAI embeddings for {len(documents)} documents...")
        openai_embeddings = self.openai_model.encode(documents)
        
        # Generate BM25 sparse vectors
        print(f"Generating BM25 sparse vectors...")
        start_time = time.time()
        bm25_vectors = []
        for doc in tqdm(documents, desc="BM25 embedding"):
            sparse_vector = self.sparse_model.embed(doc)
            bm25_vectors.append(sparse_vector)
        elapsed = time.time() - start_time
        print(f"✓ BM25 vectors completed in {elapsed:.2f} seconds ({len(documents)/elapsed:.2f} docs/sec)")
        
        # Prepare points for Qdrant
        print("Preparing points for upload...")
        points = []
        
        # Process each document
        for i in tqdm(range(len(documents)), desc="Preparing points"):
            # Skip if OpenAI embedding is None
            if openai_embeddings[i] is None:
                print(f"Skipping index {batch_start + i} due to missing OpenAI embedding")
                continue
            
            # Skip if this ID already exists in Qdrant
            if batch_start + i in existing_ids:
                print(f"Skipping index {batch_start + i} - already exists in Qdrant")
                continue
            
            # Extract and process metadata
            try:
                # Prepare payload
                row = batch_df.iloc[i]
                payload = {
                    "text": row[document_column] if document_column in row else "",
                    "title": row["title"] if "title" in row else "",
                    "code_snippet": row["code_snippet"] if "code_snippet" in row else "",
                    "length": int(row["length"]) if "length" in row and not pd.isna(row["length"]) else 0,
                    "filename": row["filename"] if "filename" in row else ""
                }
                
                # Convert numpy types to native Python types for JSON serialization
                for key, value in payload.items():
                    if hasattr(value, 'item'):
                        payload[key] = value.item()
                    elif isinstance(value, (np.int64, np.int32)):
                        payload[key] = int(value)
                    elif isinstance(value, (np.float64, np.float32)):
                        payload[key] = float(value)
                
                # Process BM25 sparse vector
                bm25_sparse = self.process_sparse_vector(bm25_vectors[i])
                
                # Create point with the correct format
                if bm25_sparse is not None:
                    # Only include sparse_vectors when we have valid sparse vectors
                    points.append(models.PointStruct(
                        id=batch_start + i,
                        vector={"OpenAI": openai_embeddings[i]},
                        sparse_vectors={"BM25": bm25_sparse},
                        payload=payload
                    ))
                else:
                    # Skip the sparse_vectors parameter entirely when it's None
                    points.append(models.PointStruct(
                        id=batch_start + i,
                        vector={"OpenAI": openai_embeddings[i]},
                        payload=payload
                    ))
                
            except Exception as e:
                print(f"Error processing document {batch_start + i}: {e}")
                continue
        
        # Upload points
        if points:
            print(f"Uploading {len(points)} points to Qdrant...")
            try:
                self.client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points,
                    wait=True  # Wait for the server to process the points
                )
                print(f"✓ Successfully uploaded {len(points)} documents to Qdrant")
            except Exception as e:
                print(f"Error uploading points to Qdrant: {e}")
                import traceback
                traceback.print_exc()
                raise  # Re-raise to handle in the main method
        else:
            print("No new points to upload in this batch")
        
        # Update progress
        progress["processed_count"] += len(documents)
        progress["last_index"] = batch_end - 1
        self.save_progress(progress)
        
        # Update existing IDs to include the newly added points
        for point in points:
            existing_ids.add(point.id)

def main():
    # Path to your CSV dataset
    csv_path = "/home/user/Desktop/output_chunk_file_classes.csv"
    
    # Create document uploader
    uploader = DocumentUploader()
    
    # Initialize Qdrant
    uploader.initialize_qdrant()
    
    # Upload dataset with resumption support
    uploader.embed_and_store_dataset(csv_path)
    
    print("Document upload process finished. The hybrid search system is now ready to use.")


if __name__ == "__main__":
    main()