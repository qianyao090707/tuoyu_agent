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

LLM_CONFIG = {
    "model_name": "qwen-plus",  # 换成你实际在用的通义模型名称
    "temperature": 0.1
}
ROLE_CONFIG = {
    "家长咨询端": "专业的托育育儿顾问，负责解答0-3岁育儿问题，提供喂养、护理、发育指导等专业建议，语气亲切温和，贴合家长需求。",
    "机构运营端": "托育园区运营管理助手，负责提供园区制度、保育服务、日常管理等相关专业建议，严谨务实。"
}

