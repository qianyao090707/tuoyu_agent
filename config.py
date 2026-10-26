# ========== 对话模型配置 ==========
MODEL_CONFIG = {
    "model_name": "qwen-plus",
    "temperature": 0.1
}

# ========== RAG 知识库配置 ==========
RAG_CONFIG = {
    "pdf_dir": "data/docs",
    "index_save_dir": "data/rag_index",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "embedding_model": "text-embedding-v2",
    "retrieve_k": 4
}
