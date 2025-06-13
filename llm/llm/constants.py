from enum import Enum


EMBEDDINGS_MODEL_NAME = "all-MiniLM-L6-v2"
# LLM_NAME = "google_gemma-3-27b-it-qat-Q5_K_M.gguf"
LLM_NAME = "gemma-3-12b-it-q4_0.gguf"


class ChainType(Enum):
    RAG = "rag"
    ITERATIONAL = "iterational"
    ONLY = "only"
