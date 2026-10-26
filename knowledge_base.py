import os
import jieba
import pickle
import hashlib
import logging
from typing import List, Optional, Dict
from collections import defaultdict

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, Docx2txtLoader

from config import RAG_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KnowledgeBase:
    """知识库：文档加载、切块、向量索引构建 + 双路召回 + RRF精排"""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.cfg = RAG_CONFIG
        self.chunk_size = int(self.cfg["chunk_size"])
        self.chunk_overlap = int(self.cfg["chunk_overlap"])
        self.doc_dir = self.cfg["pdf_dir"]
        self.index_dir = self.cfg["index_save_dir"]

        # 嵌入模型
        self.embeddings = DashScopeEmbeddings(
            model=self.cfg["embedding_model"],
            dashscope_api_key=self.api_key
        )

        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", " ", ""]
        )

        # 向量库与BM25检索器
        self.vector_store: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None

        # RRF 精排常量（工业界标准取值）
        self.rrf_k = 60

        # 初始化：有索引则加载，无则构建
        self._init_index()

    def _load_documents(self) -> List[Document]:
        """加载目录下所有 Word 文档"""
        if not os.path.exists(self.doc_dir):
            raise FileNotFoundError(f"文档目录不存在：{self.doc_dir}")

        loader = DirectoryLoader(
            self.doc_dir,
            glob="**/*.docx",
            loader_cls=Docx2txtLoader,
            show_progress=True
        )
        docs = loader.load()
        logger.info(f"检测到 {len(docs)} 个知识库文档")

        if len(docs) == 0:
            raise ValueError("DOCX目录为空，请将知识库Word文档放入./data/docs目录")

        # 语义切块
        split_docs = self.text_splitter.split_documents(docs)
        logger.info(f"文档切块完成，共 {len(split_docs)} 个文本块")
        return split_docs

    def _get_dir_hash(self) -> str:
        """计算文档目录哈希，判断是否需要重建索引"""
        hash_md5 = hashlib.md5()
        for root, _, files in os.walk(self.doc_dir):
            for file in sorted(files):
                if file.endswith(".docx"):
                    file_path = os.path.join(root, file)
                    hash_md5.update(file.encode())
                    hash_md5.update(str(os.path.getmtime(file_path)).encode())
        return hash_md5.hexdigest()

    def _init_index(self):
        """初始化索引：优先加载本地，不存在或文档更新则重建"""
        hash_file = os.path.join(self.index_dir, "doc_hash")
        current_hash = self._get_dir_hash()

        # 检查本地索引是否可用
        if os.path.exists(self.index_dir) and os.path.exists(hash_file):
            try:
                with open(hash_file, "r") as f:
                    old_hash = f.read().strip()
                if old_hash == current_hash:
                    # 加载本地向量索引
                    self.vector_store = FAISS.load_local(
                        self.index_dir,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                    # 加载BM25索引
                    bm25_path = os.path.join(self.index_dir, "bm25_retriever.pkl")
                    with open(bm25_path, "rb") as f:
                        self.bm25_retriever = pickle.load(f)
                    logger.info("本地索引加载成功")
                    return
            except Exception as e:
                logger.warning(f"本地索引加载失败，开始重建索引: {e}")

        # 重建索引
        logger.info("文档索引无效或已更新，开始重建索引")
        docs = self._load_documents()
        self._build_vector_index(docs)
        self._build_bm25_retriever(docs)
        self._save_index(current_hash)
        logger.info("索引构建完成")

    def _build_vector_index(self, docs: List[Document]):
        """构建FAISS向量索引"""
        self.vector_store = FAISS.from_documents(docs, self.embeddings)

    def _build_bm25_retriever(self, docs: List[Document]):
        """构建BM25关键词检索器（召回量翻倍，给精排留候选池）"""
        # 中文分词预处理
        for doc in docs:
            doc.page_content = " ".join(jieba.lcut(doc.page_content))
        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = int(self.cfg["retrieve_k"]) * 2

    def _save_index(self, doc_hash: str):
        """保存索引到本地"""
        os.makedirs(self.index_dir, exist_ok=True)
        self.vector_store.save_local(self.index_dir)

        bm25_path = os.path.join(self.index_dir, "bm25_retriever.pkl")
        with open(bm25_path, "wb") as f:
            pickle.dump(self.bm25_retriever, f)

        hash_file = os.path.join(self.index_dir, "doc_hash")
        with open(hash_file, "w") as f:
            f.write(doc_hash)

    def _rrf_rank(self, doc_lists: List[List[Document]]) -> List[Document]:
        """
        RRF（倒数排名融合）精排算法
        公式：score = sum( 1 / (rrf_k + 排名) )
        """
        scores: Dict[str, float] = defaultdict(float)
        doc_map: Dict[str, Document] = {}

        for doc_list in doc_lists:
            for rank, doc in enumerate(doc_list, start=1):
                doc_id = doc.page_content
                doc_map[doc_id] = doc
                scores[doc_id] += 1.0 / (self.rrf_k + rank)

        # 按分数降序排序
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[doc_id] for doc_id, _ in sorted_docs]

    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """
        双路召回 + RRF精排
        1. BM25关键词召回 + 向量语义召回
        2. RRF算法融合两路结果，输出精排后的 top_k
        """
        if not self.vector_store or not self.bm25_retriever:
            raise RuntimeError("知识库未初始化")

        # 查询分词，和BM25预处理对齐
        query_cut = " ".join(jieba.lcut(query))

        # 1. 两路分别召回（候选池翻倍）
        bm25_docs = self.bm25_retriever.invoke(query_cut)
        vector_retriever = self.vector_store.as_retriever(
            search_kwargs={"k": int(self.cfg["retrieve_k"]) * 2}
        )
        vector_docs = vector_retriever.invoke(query)

        # 2. RRF 精排融合
        ranked_docs = self._rrf_rank([bm25_docs, vector_docs])

        # 3. 返回最终 top_k
        return ranked_docs[:top_k]
