import logging
from typing import List, Tuple
from langchain_community.llms import Tongyi
from langchain_core.documents import Document

from knowledge_base import KnowledgeBase
from config import LLM_CONFIG, ROLE_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultiAgentOrchestrator:
    """
    托育场景多智能体助手
    核心流程：ReAct推理判断 → 双路召回+RRF精排 → 内容生成 → 合规审核
    支持角色：家长咨询端 / 机构运营端
    """

    def __init__(self, api_key: str, role: str = "家长咨询端", role_config=None):
        self.api_key = api_key
        self.role = role

        # 初始化知识库（双路召回 + RRF精排）
        self.kb = KnowledgeBase(api_key)

        # 初始化大模型
        self.llm = Tongyi(
            model=LLM_CONFIG["model_name"],
            dashscope_api_key=api_key,
            temperature=float(LLM_CONFIG["temperature"]),
            top_p=0.8
        )

        # 角色约束：优先用传入的 role_config，没有就从配置文件读取
        if role_config is not None:
            self.role_desc = role_config
        else:
            self.role_desc = ROLE_CONFIG.get(role, ROLE_CONFIG["家长咨询端"])

        # 审核最大重试次数
        self.max_audit_retry = 2

    def _build_context(self, docs: List[Document]) -> str:
        """将检索到的文档片段拼接为上下文文本"""
        return "\n\n".join([f"【参考片段{i+1}】{doc.page_content}" for i, doc in enumerate(docs)])

    def _react_judge(self, query: str, docs: List[Document]) -> Tuple[bool, str]:
        """
        ReAct 第一步：推理判断
        判断当前召回文档是否足以回答用户问题；若不足，生成更精准的二次检索关键词
        返回：(是否需要二次检索, 二次检索关键词)
        """
        context = self._build_context(docs)

        prompt = f"""
你是专业的信息评估助手。请基于以下知识库参考片段，判断能否完整、准确地回答用户问题。
【用户角色】{self.role}
【用户问题】{query}
【知识库参考片段】
{context}

【要求】
1. 仅基于给出的参考片段判断，禁止自行补充知识
2. 若参考片段能覆盖问题核心要点，判定为「充分」
3. 若信息不足，请提炼出更精准的检索关键词，用于二次查找

【输出格式，严格按下面两行输出，不要多余内容】
是否充分：是/否
补充检索关键词：无/你的关键词
        """.strip()

        try:
            result = self.llm.invoke(prompt).strip()
            # 解析结果，容错处理
            is_sufficient = "是" in result.split("是否充分：")[-1].split("\n")[0]
            keyword_part = result.split("补充检索关键词：")[-1].strip().split("\n")[0]
            need_second_retrieve = not is_sufficient and keyword_part and keyword_part != "无"
            second_keyword = keyword_part if need_second_retrieve else ""

            logger.info(f"ReAct判断结果：充分={is_sufficient}，二次检索关键词={second_keyword}")
            return need_second_retrieve, second_keyword

        except Exception as e:
            logger.warning(f"ReAct判断异常，跳过二次检索：{e}")
            return False, ""

    def _generate_answer(self, query: str, docs: List[Document]) -> str:
        """基于最终召回文档生成专业回答"""
        context = self._build_context(docs)

        prompt = f"""
你是一名{self.role_desc}。请严格基于下方知识库参考片段回答用户问题。
【角色要求】
- 回答必须全部来自参考片段，禁止编造任何不在片段中的信息
- 语言专业、温和、有条理，符合托育场景规范
- 若参考片段没有相关内容，请明确告知「暂无相关规范说明，建议咨询园区工作人员」

【用户问题】{query}
【知识库参考片段】
{context}
        """.strip()

        answer = self.llm.invoke(prompt).strip()
        return answer

    def _audit_answer(self, query: str, answer: str, docs: List[Document]) -> Tuple[bool, str]:
        """
        审核智能体：校验回答是否合规、是否超纲、是否存在幻觉
        返回：(是否通过, 审核意见)
        """
        context = self._build_context(docs)

        prompt = f"""
你是合规审核助手。请校验下面的回答是否严格基于知识库参考片段，是否符合角色要求。
【用户问题】{query}
【角色】{self.role}
【知识库参考片段】
{context}
【待审核回答】
{answer}

【审核标准】
1. 回答内容是否全部来自参考片段，无编造、无幻觉
2. 是否符合{self.role}的专业表述规范
3. 是否存在无关内容、错误建议

【输出格式，严格按下面两行输出】
是否通过：是/否
审核意见：简要说明原因
        """.strip()

        try:
            result = self.llm.invoke(prompt).strip()
            is_pass = "是" in result.split("是否通过：")[-1].split("\n")[0]
            opinion = result.split("审核意见：")[-1].strip()
            return is_pass, opinion
        except Exception as e:
            logger.warning(f"审核异常，默认通过：{e}")
            return True, "审核异常，默认放行"

    def run(self, query: str) -> str:
        """
        完整问答主入口：ReAct推理 → 召回精排 → 生成 → 审核（最多两轮复核）
        """
        # ========== 第一轮：初始召回 ==========
        first_docs = self.kb.retrieve(query, top_k=3)
        logger.info(f"初始召回完成，共{len(first_docs)}个片段")

        # ========== ReAct 推理-行动循环 ==========
        need_retrieve, second_keyword = self._react_judge(query, first_docs)
        final_docs = first_docs

        if need_retrieve and second_keyword:
            logger.info(f"执行ReAct二次检索，关键词：{second_keyword}")
            second_docs = self.kb.retrieve(second_keyword, top_k=2)
            # 合并两次召回结果，去重
            final_docs = first_docs + [
                doc for doc in second_docs
                if doc.page_content not in [d.page_content for d in first_docs]
            ]

        # ========== 生成 + 审核闭环 ==========
        current_docs = final_docs
        final_answer = ""

        for retry in range(self.max_audit_retry):
            answer = self._generate_answer(query, current_docs)
            is_pass, opinion = self._audit_answer(query, answer, current_docs)

            if is_pass:
                final_answer = answer
                logger.info(f"第{retry+1}轮审核通过")
                break
            else:
                logger.warning(f"第{retry+1}轮审核不通过：{opinion}，退回重查")
                # 审核不通过，扩充召回数量重新检索
                current_docs = self.kb.retrieve(query, top_k=5)
        else:
            final_answer = "抱歉，未能生成符合规范的回答，请换一种方式提问或咨询园区工作人员。"
            logger.error("多轮审核均未通过")

        return final_answer
