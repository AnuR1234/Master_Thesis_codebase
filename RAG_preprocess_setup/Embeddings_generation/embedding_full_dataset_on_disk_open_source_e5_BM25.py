import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import json
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Qdrant imports
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.models import SparseVector, Distance
from fastembed import SparseTextEmbedding

# Constants
QDRANT_HOST = "localhost"  # If running Docker on the same machine
QDRANT_PORT = 6333  # Default HTTP API port
COLLECTION_NAME = "SAP_ABAP_CODE_DOCUMENTATION_E5_BM25_REPORTS"  # Updated collection name
E5_VECTOR_SIZE = 1024  # Size of E5-large-v2 embeddings
PROGRESS_FILE = "embedding_progress_reports.json"  # File to track progress
BATCH_SIZE = 128  # Smaller batch size to process documents
BATCH_PAUSE = 5  # Seconds to pause between batches

# Embedding time measurement utility
def measure_embedding_time(embedding_function, documents):
    """Measure time taken to generate embeddings"""
    start_time = time.time()
    embeddings = embedding_function(documents)
    end_time = time.time()
    return embeddings, end_time - start_time

# Main document processing class
class DocumentUploader:
    def __init__(self):
        """Initialize the document uploader"""
        print("Initializing document uploader for SAP ABAP code documentation...")
        
        # Initialize BM25 model
        try:
            print("Loading BM25 model...")
            self.sparse_model = SparseTextEmbedding("Qdrant/bm25")
            print("✓ BM25 model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading BM25 model: {e}")
            raise
        
        # Initialize E5-large-v2 model instead of BGE-M3
        try:
            print("Loading E5-large-v2 model with CPU...")
            self.e5_model = SentenceTransformer('intfloat/e5-large-v2', device='cpu')
            
            # Test the model with a simple input
            test_text = "Test sentence"
            test_embedding = self.e5_model.encode(
                ["query: " + test_text], 
                batch_size=1, 
                normalize_embeddings=True, 
                device='cpu'
            )
            print(f"✓ E5-large-v2 model loaded successfully")
            print(f"  Embedding dimension: {test_embedding.shape[1]}")
            
            # Update global vector size
            global E5_VECTOR_SIZE
            E5_VECTOR_SIZE = test_embedding.shape[1]
        except Exception as e:
            print(f"✗ Error loading E5-large-v2 model: {e}")
            raise
        
        # Initialize Qdrant client
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            
    def initialize_qdrant(self):
        """Initialize Qdrant collection with proper vector configurations"""
        print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        
        # Check if collection exists and optionally recreate it
        recreate = True  # Set to True to recreate the collection
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
                    "E5": models.VectorParams(  # Changed from "BGE" to "E5"
                        size=E5_VECTOR_SIZE,
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
        """
        if sparse_vec is None:
            return None
            
        indices = []
        values = []
        
        try:
            # For generator objects
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
                    
                # Pause between batches to avoid overwhelming the system
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
        
        # Generate E5-large-v2 embeddings
        print(f"Generating E5-large-v2 embeddings for {len(documents)} documents...")
        try:
            # Create prefixed texts with "query: " for E5 model
            prefixed_texts = ["query: " + text for text in documents]
            
            # Generate embeddings
            e5_embeddings, e5_time = measure_embedding_time(
                lambda x: self.e5_model.encode(
                    x, 
                    batch_size=4, 
                    normalize_embeddings=True, 
                    device='cpu'
                ),
                prefixed_texts
            )
            print(f"✓ E5-large-v2 embeddings generated in {e5_time:.2f} seconds ({len(documents)/e5_time:.2f} docs/sec)")
            print(f"  Embedding dimension: {e5_embeddings.shape[1]}")
            print(f"  Number of embeddings: {len(e5_embeddings)}")
        except Exception as e:
            print(f"Error generating E5 embeddings: {e}")
            import traceback
            traceback.print_exc()
            raise
        
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
                
                # Get E5 embedding and ensure it's a list
                e5_embedding = e5_embeddings[i].tolist() if isinstance(e5_embeddings[i], np.ndarray) else e5_embeddings[i]
                
                # Process BM25 sparse vector
                bm25_sparse = self.process_sparse_vector(bm25_vectors[i])
                
                # Create point with the correct format - with error checking
                try:
                    if bm25_sparse is not None:
                        points.append(models.PointStruct(
                            id=batch_start + i,
                            vector={"E5": e5_embedding},  # Changed from "BGE" to "E5"
                            sparse_vectors={"BM25": bm25_sparse},
                            payload=payload
                        ))
                    else:
                        points.append(models.PointStruct(
                            id=batch_start + i,
                            vector={"E5": e5_embedding},  # Changed from "BGE" to "E5"
                            payload=payload
                        ))
                except Exception as e:
                    print(f"Error creating point for document {batch_start + i}: {e}")
                    print(f"E5 embedding type: {type(e5_embedding)}, length: {len(e5_embedding) if isinstance(e5_embedding, list) else 'N/A'}")
                    continue
                
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
    # Make sure required packages are installed
    try:
        import sentence_transformers
        print(f"✓ sentence_transformers version {sentence_transformers.__version__} is installed")
    except ImportError:
        print("✗ sentence_transformers is not installed! Installing now...")
        import subprocess
        subprocess.check_call(["pip", "install", "sentence-transformers"])
        print("✓ sentence_transformers installed successfully")
        import sentence_transformers
    
    # Path to your CSV dataset
    csv_path = "/home/user/Desktop/embedding_openai_bm25/output_chunk_file_reports.csv"
    
    # Create document uploader
    uploader = DocumentUploader()
    
    # Initialize Qdrant
    uploader.initialize_qdrant()
    
    # Upload dataset with resumption support
    uploader.embed_and_store_dataset(csv_path)
    
    print("Document upload process finished. The hybrid search system is now ready to use.")


if __name__ == "__main__":
    main()