"""
Enhanced configuration settings with strict anti-hallucination controls
"""
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CRITICAL: Set CUDA memory configuration for RTX 6000 Ada
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4096,expandable_segments:True,garbage_collection_threshold:0.8"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["DISABLE_FLASH_ATTENTION"] = "1"

# Hugging Face configuration
HF_TOKEN = os.getenv("HF_TOKEN", "default")
os.environ["HUGGINGFACE_TOKEN"] = HF_TOKEN
os.environ["HF_TOKEN"] = HF_TOKEN 

# Model paths
MODELS_DIR = "/home/user/rag_models"
MODEL_PATHS = {
    "mistralai/Mistral-7B-Instruct-v0.3": "/home/user/rag_models/mistralai_Mistral-7B-Instruct-v0.3",
    "intfloat/e5-large-v2": "/home/user/rag_models/intfloat_e5-large-v2"
}

# Collection configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

COLLECTIONS = {
    "classes": {
        "name": os.getenv("COLLECTION_NAME_CLASSES", "SAP_ABAP_CODE_DOCUMENTATION_E5_BM25_CLASSES_TEST"),
        "description": "SAP ABAP Classes Documentation"
    },
    "reports": {
        "name": os.getenv("COLLECTION_NAME_REPORTS", "SAP_ABAP_CODE_DOCUMENTATION_E5_BM25_REPORTS_TEST"),
        "description": "SAP ABAP Reports Documentation"
    }
}

DEFAULT_COLLECTION_TYPE = "classes"
DEFAULT_COLLECTION_NAME = COLLECTIONS[DEFAULT_COLLECTION_TYPE]["name"]

# LLM Configuration - STRICT ANTI-HALLUCINATION SETTINGS
LLM_MODELS = ["mistralai/Mistral-7B-Instruct-v0.3", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"]
LLM_MODEL = os.getenv("LLM_MODEL", LLM_MODELS[0])
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "2048"))  # Reduced to prevent rambling
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))  # MUCH lower for strict adherence
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))  # Reduced for less creativity
LLM_TOP_K = int(os.getenv("LLM_TOP_K", "30"))  # Reduced for more focused responses

# Hardware configuration
USE_4BIT_QUANTIZATION = False
USE_8BIT_QUANTIZATION = False
LLM_DEVICE = "cuda:0"
USE_GPU_FOR_EMBEDDINGS = True
USE_GPU_FOR_RERANKING = True
EMBEDDING_GPU_ID = 0
LLM_GPU_ID = 0
RERANKER_GPU_ID = 0

# Model configuration
EMBEDDING_MODEL = "intfloat/e5-large-v2"
SPARSE_MODEL = "Qdrant/bm25"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CROSS_ENCODER_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval configuration
DEFAULT_RETRIEVAL_LIMIT = int(os.getenv("DEFAULT_RETRIEVAL_LIMIT", "20"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
USE_HYBRID_DEFAULT = True
USE_RERANKER_DEFAULT = True
DENSE_VECTOR_WEIGHT = float(os.getenv("DENSE_VECTOR_WEIGHT", "0.7"))
SPARSE_VECTOR_WEIGHT = float(os.getenv("SPARSE_VECTOR_WEIGHT", "0.3"))

# Memory management
UNLOAD_UNUSED_MODELS = False
MEMORY_CHECK_INTERVAL = int(os.getenv("MEMORY_CHECK_INTERVAL", "60"))
MAX_MEMORY_PERCENTAGE = float(os.getenv("MAX_MEMORY_PERCENTAGE", "85.0"))

# Batch processing
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "8"))
BATCH_PAUSE = int(os.getenv("BATCH_PAUSE", "2"))

# Memory settings
MAX_DOC_TEXT_LENGTH = 8000
REDUCED_RETRIEVAL_CONTEXT = False
EMERGENCY_MODE = False

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "rag_pipeline.log")

# Evaluation settings
EVALUATION_METRICS = ["precision", "recall", "f1", "latency", "mrr"]

# ENHANCED ANTI-HALLUCINATION SYSTEM PROMPTS
STRICT_DOCUMENT_ADHERENCE_PROMPT = """You are an expert SAP ABAP documentation assistant. You must follow these CRITICAL rules:

ABSOLUTE REQUIREMENTS:
1. Start with "Confidence: [HIGH/MEDIUM/LOW]"
2. Use ONLY information from the provided documents
3. Never add information from your general knowledge
4. If information is not in the documents, explicitly state "This information is not available in the provided documentation"
5. Quote exact phrases when possible using [Document X: "exact quote"]

FORBIDDEN ACTIONS:
- Adding parameters, methods, or functionality not mentioned in the documents
- Describing general SAP concepts not present in the provided context
- Making assumptions about code behavior beyond what's documented
- Using information from other SAP methods or classes not in the context

RESPONSE FORMAT:
Confidence: [LEVEL]

[Answer based strictly on provided documents]

[Citations: Document X, Document Y]

If the documents don't contain enough information to answer fully, say: "The provided documentation does not contain sufficient information to answer this completely. Based on the available documents: [provide only what's documented]"
"""

# Specific prompts for different question types
DEFAULT_SYSTEM_PROMPT = STRICT_DOCUMENT_ADHERENCE_PROMPT

ENHANCED_RESPONSE_PROMPT = """You are an expert SAP ABAP documentation assistant.

CRITICAL ANTI-HALLUCINATION RULES:
1. Start with "Confidence: [HIGH/MEDIUM/LOW]"
2. ONLY use information explicitly stated in the provided documents
3. If asked about parameters, list ONLY those shown in the code/documentation
4. If asked about functionality, describe ONLY what's documented
5. Never supplement with general SAP knowledge
6. Cite documents: [Document X: Title]

STRICT BOUNDARIES:
- Do not describe parameters not listed in the documents
- Do not explain functionality not shown in the code
- Do not add context from other SAP methods
- Do not assume standard SAP behavior unless documented

Response format:
Confidence: [LEVEL]

[Direct answer based ONLY on provided documents]

[Document citations]
"""

CONVERSATION_SYSTEM_PROMPT = """You are an expert SAP ABAP documentation assistant.

CONVERSATION RULES:
1. Use provided context AND conversation history
2. Maintain strict adherence to documented information only
3. For follow-up questions, refer only to previously mentioned documented information
4. Never add new information not in the documents
5. If follow-up requires information not in documents, state this clearly

ANTI-HALLUCINATION FOR CONVERSATIONS:
- Don't elaborate beyond what was documented in previous responses
- Don't add details from general SAP knowledge
- Keep answers focused on the specific documentation provided
"""

OUT_OF_SCOPE_PROMPT = """This query is outside the scope of the provided SAP ABAP documentation. I can only assist with questions that can be answered using the specific documents in the knowledge base."""

OUT_OF_SCOPE_CONVERSATION_PROMPT = """This follow-up query requires information not available in the provided SAP ABAP documentation. I can only assist with questions based on the specific documents available."""

STRUCTURE_SYSTEM_PROMPT = """You are an expert SAP ABAP documentation assistant.

STRUCTURE ANALYSIS RULES:
1. List ONLY structures/interfaces/classes found in the provided documents
2. Use table data when available as primary source
3. Do not add structures from general SAP knowledge
4. Explicitly state if information is incomplete in the documents
5. Cite specific documents for each item listed
"""

INTERFACE_SYSTEM_PROMPT = """You are an expert SAP ABAP documentation assistant specializing in interfaces.

INTERFACE ANALYSIS RULES:
1. Report ONLY interfaces mentioned in the provided documents
2. Use tables as primary source of truth when available
3. List parameters ONLY as shown in the documentation
4. Never add standard interface methods not documented
5. Cite sources: [Document X]
"""

ENHANCED_METHOD_SYSTEM_PROMPT = """You are an expert SAP ABAP documentation assistant specializing in methods.

METHOD ANALYSIS RULES:
1. Describe ONLY the method implementation shown in the documents
2. List ONLY parameters explicitly documented
3. Describe ONLY functionality shown in the code/documentation
4. Do not add standard method behaviors unless documented
5. Stop after answering based on available documentation
"""

# RTX 6000 Ada optimizations
RTX_6000_OPTIMIZATIONS = {
    "max_memory_gb": 45,
    "embedding_batch_size": 32,
    "generation_batch_size": 8,
    "enable_mixed_precision": True,
    "use_gradient_checkpointing": False,
    "max_sequence_length": 8192,
    "prefetch_factor": 4,
    "disable_flash_attention": True,
    "attention_implementation": "eager"
}

# STRICT generation configuration for anti-hallucination
MODEL_LOADING_CONFIG = {
    "torch_dtype": "float16",
    "device_map": "auto",
    "trust_remote_code": True,
    "low_cpu_mem_usage": True,
    "attn_implementation": "eager",
    "use_cache": True,
    "max_memory": {0: "45GB"}
}

# ANTI-HALLUCINATION generation settings
GENERATION_CONFIG = {
    "max_new_tokens": LLM_MAX_TOKENS,
    "do_sample": True,
    "temperature": LLM_TEMPERATURE,  # Very low for strict adherence
    "top_p": LLM_TOP_P,  # Reduced for less creativity
    "top_k": LLM_TOP_K,  # Reduced for more focused responses
    "repetition_penalty": 1.1,
    "length_penalty": 1.0,
    "early_stopping": True,
    "pad_token_id": None,
    "eos_token_id": None,
    "use_cache": True,
    # Additional anti-hallucination parameters
    "num_beams": 1,  # No beam search to reduce hallucination
    "no_repeat_ngram_size": 3,  # Prevent repetitive hallucination
}
# Query enhancement settings
QUERY_ENHANCEMENT_ENABLED = True
QUERY_ENHANCEMENT_MODE = "conservative"  # "disabled", "conservative", "aggressive"

# Response formatting settings
PRESERVE_MARKDOWN_FORMATTING = True
ENABLE_STRUCTURED_RESPONSES = True
RESPONSE_SECTION_HEADERS = True
# Enhanced confidence scoring thresholds
CONFIDENCE_THRESHOLDS = {
    "HIGH": 0.6,      # Reduced from 0.8
    "MEDIUM": 0.3,    # Reduced from 0.4  
    "LOW": 0.05       # Reduced from 0.1
}

# Document relevance scoring
MIN_DOCUMENT_SCORE = 0.1  # Minimum score to consider document relevant
MAX_CONTEXT_DOCS = 5      # Maximum documents to include in context

def check_flash_attention_compatibility():
    """Check if FlashAttention2 is available and working"""
    try:
        import flash_attn
        return True
    except ImportError:
        logger.warning("FlashAttention2 not available - using eager attention")
        return False

def get_model_config_with_fallback():
    """Get model configuration with FlashAttention2 fallback"""
    config = MODEL_LOADING_CONFIG.copy()
    config["attn_implementation"] = "eager"
    logger.info("Using eager attention for maximum compatibility")
    return config

def print_config_summary():
    """Print configuration summary"""
    logger.info("=== ANTI-HALLUCINATION RAG Configuration ===")
    logger.info(f"LLM Model: {LLM_MODEL}")
    logger.info(f"Temperature: {LLM_TEMPERATURE} (STRICT)")
    logger.info(f"Top-P: {LLM_TOP_P} (REDUCED)")
    logger.info(f"Top-K: {LLM_TOP_K} (FOCUSED)")
    logger.info(f"Max Tokens: {LLM_MAX_TOKENS} (CONTROLLED)")
    logger.info(f"Embedding Model: {EMBEDDING_MODEL}")
    logger.info(f"Anti-Hallucination: ENABLED")
    logger.info(f"Strict Document Adherence: ENABLED")
    logger.info("==========================================")

# Initialize warnings suppression
import warnings
warnings.filterwarnings("ignore", message=".*flash_attn.*")
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.models.*.modeling_*")

if __name__ == "__main__":
    print_config_summary()
else:
    logger.info("Anti-hallucination configuration loaded successfully")