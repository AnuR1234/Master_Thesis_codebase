"""
Configuration settings for the RAG Pipeline
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Qdrant Collection Configuration
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))

# Collection names and descriptions
COLLECTIONS = {
    "classes": {
        "name": os.getenv("COLLECTION_NAME_CLASSES", "SAP_ABAP_CODE_DOCUMENTATION_3")
,
        "description": "SAP ABAP Classes Documentation"
    },
    "reports": {
        "name": os.getenv("COLLECTION_NAME_CLASSES", "SAP_ABAP_CODE_DOCUMENTATION_REPORT"),
        "description": "SAP ABAP Reports Documentation"
    }
}

# Default collection to use if none is selected
DEFAULT_COLLECTION_TYPE = "classes"
DEFAULT_COLLECTION_NAME = COLLECTIONS[DEFAULT_COLLECTION_TYPE]["name"]

# Claude Bedrock Configuration
BEDROCK_MODEL = os.getenv("BEDROCK_MODEL", "anthropic--claude-3.7-sonnet")
BEDROCK_MAX_TOKENS = int(os.getenv("BEDROCK_MAX_TOKENS", "4096"))
BEDROCK_TEMPERATURE = float(os.getenv("BEDROCK_TEMPERATURE", "0.7"))
BEDROCK_TOP_P = float(os.getenv("BEDROCK_TOP_P", "0.9"))

# Embedding Model Configuration
PENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-ada-002")
AICORE_AUTH_URL = os.getenv("AICORE_AUTH_URL", "")
AICORE_CLIENT_ID = os.getenv("AICORE_CLIENT_ID", "")
AICORE_CLIENT_SECRET = os.getenv("AICORE_CLIENT_SECRET", "")
AICORE_BASE_URL = os.getenv("AICORE_BASE_URL", "")
AICORE_RESOURCE_GROUP = os.getenv("AICORE_RESOURCE_GROUP", "default")
# Retrieval Configuration
DEFAULT_RETRIEVAL_LIMIT = int(os.getenv("DEFAULT_RETRIEVAL_LIMIT", "10"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
USE_HYBRID_DEFAULT = os.getenv("USE_HYBRID_DEFAULT", "True").lower() == "true"
USE_RERANKER_DEFAULT = os.getenv("USE_RERANKER_DEFAULT", "True").lower() == "true"

# Vector weights for hybrid search
#DENSE_VECTOR_WEIGHT = float(os.getenv("DENSE_VECTOR_WEIGHT", "0.7")) 
#SPARSE_VECTOR_WEIGHT = float(os.getenv("SPARSE_VECTOR_WEIGHT", "0.3"))
DENSE_VECTOR_WEIGHT = float(os.getenv("DENSE_VECTOR_WEIGHT", "0.4")) 
SPARSE_VECTOR_WEIGHT = float(os.getenv("SPARSE_VECTOR_WEIGHT", "0.6"))
# Reranker settings
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


# Default system prompt
DEFAULT_SYSTEM_PROMPT = """You are an expert assistant for SAP ABAP code documentation.

CRITICAL INSTRUCTIONS:
1. Answer questions based ONLY on the provided context. If the context doesn't contain relevant information, say so clearly.
2. Focus EXCLUSIVELY on answering the specific question asked - do not provide additional information not requested.
3. Structure your response in a conversational, concise format:
   - Begin with a direct answer to the question
   - Then provide supporting details from the context
   - Use code blocks with proper formatting for any code
4. Only cite specific documents when necessary by referring to document numbers in [Document X] format.
5. Do not add section headings, titles, or other formatting that makes the response appear like a formal report.
6. Never invent or assume information not present in the provided context.
7. Use plain, conversational language as if you're having a chat.
8. Keep your response focused and to the point - don't add tangential information.
"""

# Updated system prompt that handles conversation context
CONVERSATION_SYSTEM_PROMPT = """You are an expert assistant for SAP ABAP code documentation.

CRITICAL INSTRUCTIONS:
1. Answer questions based ONLY on the provided context AND conversation history.
2. If the current question is a follow-up, use the conversation history to understand the context.
3. Focus EXCLUSIVELY on answering the specific question asked - do not provide additional information not requested.
4. Structure your response in a conversational, concise format:
   - Begin with a direct answer to the question
   - Then provide supporting details from the context
   - Use code blocks with proper formatting for any code
5. Only cite specific documents when necessary by referring to document numbers.
6. Do not add section headings, titles, or other formatting that makes the response appear like a formal report.
7. Keep your response focused and to the point - don't add tangential information.
"""

# System prompt for out-of-scope queries
OUT_OF_SCOPE_PROMPT = """You are an expert assistant for SAP ABAP code documentation.
You should only answer questions related to SAP ABAP code and documentation.

The current query appears to be outside the scope of SAP ABAP documentation.
Please provide a brief, conversational response informing the user that their question is not related to SAP ABAP code documentation and that you can only assist with SAP ABAP related queries.

Do not attempt to answer questions outside of your SAP ABAP expertise or provide irrelevant information.
"""

# System prompt for out-of-scope follow-up queries
OUT_OF_SCOPE_CONVERSATION_PROMPT = """You are an expert assistant for SAP ABAP code documentation.
You should only answer questions related to SAP ABAP code and documentation.

The current follow-up query appears to be outside the scope of SAP ABAP documentation, even considering the conversation history.
Please provide a brief, conversational response informing the user that their question is not related to SAP ABAP code documentation and that you can only assist with SAP ABAP related queries.

Do not attempt to answer questions outside of your SAP ABAP expertise or provide irrelevant information.
"""
# System prompt specifically for structure-related queries
STRUCTURE_SYSTEM_PROMPT = """You are an expert assistant for SAP ABAP code documentation.

CRITICAL INSTRUCTIONS:
1. Answer questions based ONLY on the provided context with extreme precision about code structure.
2. Focus EXCLUSIVELY on answering the specific question asked - do not provide additional information not requested.
3. ALWAYS prioritize information from tables, lists, and structured formats over narrative text.
4. When asked about interfaces, classes, methods, or other structural elements:
   - List ALL instances found in the documentation, not just the prominent ones
   - Include both names AND descriptions/purposes when available
   - If tables and narrative text conflict, trust the tables as they're more comprehensive
5. Format your response in a natural, conversational format - do not use section headings.
6. Begin with a direct answer to the query, then provide supporting details.
7. Only cite specific documents when necessary by referring to document numbers.
"""

# Interface system prompt
INTERFACE_SYSTEM_PROMPT = """You are an expert assistant for SAP ABAP code documentation specializing in interface information.

CRITICAL INSTRUCTIONS:
1. Answer ONLY what was specifically asked - do not provide additional information.
2. TABLES are your PRIMARY source of truth. When information exists in both tables and text:
   - ALWAYS prioritize information from tables over narrative text
   - Tables will contain COMPLETE lists of interfaces, while text may only mention some
   - When tables show multiple interfaces, you MUST include ALL of them
3. Begin with a direct, conversational answer that precisely addresses the question.
4. Format your response as a natural conversation, not a formal document with headings.
5. Only cite documents when necessary using the format: [Document X]
6. Focus exclusively on information directly relevant to the question asked.
"""
# Enhanced method system prompt
ENHANCED_METHOD_SYSTEM_PROMPT = """You are an expert assistant for SAP ABAP code documentation specializing in method implementations.

CRITICAL INSTRUCTIONS:
1. Answer ONLY what the user specifically asked about the method - do not provide other information.
2. When the user asks about a specific method:
   - First, provide a direct answer to their exact question
   - If they ask for purpose, only explain the purpose
   - If they ask for parameters, only explain the parameters
   - If they ask for return values, only explain what is returned
   - Never add additional information they didn't ask for

3. Begin every response conversationally, as if continuing a dialogue.
   - Start with a direct answer like "The purpose of X method is..." 
   - Don't use headings or section titles

4. When citing documents, use the format: [Document X]

5. Your highest priority is answering EXACTLY what was asked, nothing more.

6. When you've fully answered the question, stop. Don't add extra information.
"""

# Confidence assessment instructions
CONFIDENCE_ASSESSMENT_INSTRUCTIONS = """
You must assess your confidence level in your answer, but include it discreetly at the end of your response.

Confidence Assessment Guidelines:
1. HIGH CONFIDENCE: When your answer is directly supported by multiple specific documents in the context, with clear and consistent information.
2. MEDIUM CONFIDENCE: When your answer is partially supported by the context, but some details may be inferred or there are minor inconsistencies between documents.
3. LOW CONFIDENCE: When your answer is based on limited information in the context, contains significant inferences, or when different documents provide contradictory information.

End your response with a subtle, one-line statement: "Confidence: [HIGH/MEDIUM/LOW]"
"""

# Evaluation settings
EVALUATION_METRICS = ["precision", "recall", "f1", "latency", "mrr"]

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "rag_pipeline.log")

# Processing batch sizes
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "10"))
BATCH_PAUSE = int(os.getenv("BATCH_PAUSE", "30"))