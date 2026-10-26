from typing import List
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.documents import Document

from knowledge_base import KnowledgeBase
from config import MODEL_CONFIG


class QueryAgent:
    """查询智能体：关键词优化与知识库精准召回"""

    def __init__(self, api_key: str, kb: KnowledgeBase):
        self.llm = ChatTongyi(
            model=MODEL_CONFIG["model_name"],
            dashscope_api_key=api_key,
            temperature=0.1
        )
        self.kb = kb
        self.prompt = PromptTemplate.from_template("""
你是专业检索专员。请根据用户问题和驳回原因（如果有），提取1-3个最精准的检索关键词，用空格分隔。
用户问题：{query}
驳回原因（无则忽略）：{reject_reason}
检索关键词：
""")

    def search(self, query: str, reject_reason: str = "", top_k: int = 3) -> List[Document]:
        keywords = self.llm.invoke(
            self.prompt.format(query=query, reject_reason=reject_reason)
        ).content
        return self.kb.retrieve(keywords, top_k=top_k)


class GenerateAgent:
    """生成智能体：基于召回知识库内容生成合规回答"""

    def __init__(self, api_key: str, system_prompt: str):
        self.llm = ChatTongyi(
            model=MODEL_CONFIG["model_name"],
            dashscope_api_key=api_key,
            temperature=0.1
        )
        self.system_prompt = system_prompt
        self.prompt = PromptTemplate.from_template("""
{system_prompt}

请严格参考以下知识库内容回答用户问题，禁止编造知识库中没有的信息。
如果知识库内容不足以回答问题，请如实说明，不要自行虚构内容。

【知识库参考内容】
{context}

【用户问题】
{query}

请给出专业、清晰的回答：
""")

    def generate(self, query: str, context_docs: List[Document]) -> str:
        context = "\n\n".join([doc.page_content for doc in context_docs])
        full_prompt = self.prompt.format(
            system_prompt=self.system_prompt,
            context=context,
            query=query
        )
        response = self.llm.invoke(full_prompt)
        return response.content


class MultiAgentOrchestrator:
    """多智能体调度器：协调查询召回、生成应答全流程"""

    def __init__(self, api_key: str, role_config: dict):
        self.api_key = api_key
        self.role_config = role_config
        self.system_prompt = role_config["prompt"]

        # 初始化 RAG 知识库
        self.kb = KnowledgeBase(api_key=api_key)

        # 初始化各职能 Agent
        self.query_agent = QueryAgent(api_key=api_key, kb=self.kb)
        self.generate_agent = GenerateAgent(api_key=api_key, system_prompt=self.system_prompt)

    def run(self, query: str) -> str:
        # 1. 查询 Agent 优化关键词、召回知识库片段
        retrieve_docs = self.query_agent.search(query, top_k=3)

        # 2. 生成 Agent 基于召回内容输出最终回答
        answer = self.generate_agent.generate(query, retrieve_docs)

        return answer
