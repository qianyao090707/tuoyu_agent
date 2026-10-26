import streamlit as st
from multi_agent import MultiAgentOrchestrator

# 页面基础配置
st.set_page_config(page_title="托育智辅", page_icon="🧸", layout="wide")

# ====================== 角色配置 ======================
ROLE_CONFIG = {
    "家长咨询端": {
        "prompt": "你是园区智能育儿顾问，负责解答0-3岁育儿问题、介绍托育服务、预约免费体验课，语气亲切专业，贴合家长需求。",
        "scene": "consult"
    },
    "机构运营端": {
        "prompt": "你是园区运营辅助助手，负责提供园区运营、家长沟通话术、托育管理相关的专业建议，严谨实用。",
        "scene": "operation"
    }
}

# ====================== 会话状态初始化 ======================
if "agent" not in st.session_state:
    st.session_state.agent = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_role" not in st.session_state:
    st.session_state.current_role = "家长咨询端"


# ====================== 统一问答处理函数 ======================
def handle_user_query(query: str):
    """统一处理用户提问：加入对话 + 调用模型生成回复"""
    st.session_state.messages.append({"role": "user", "content": query})

    if not st.session_state.agent:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ 请先在左侧输入API密钥，点击「初始化智能体」后再提问"
        })
        return

    try:
        with st.spinner("正在思考..."):
            answer = st.session_state.agent.run(query)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"调用出错：{str(e)}"
        })


# ====================== 侧边栏配置 ======================
with st.sidebar:
    st.header("系统配置")

    api_key = st.text_input("DashScope API密钥", type="password", placeholder="请输入API Key")

    if st.button("初始化智能体", use_container_width=True):
        if not api_key.strip():
            st.error("请先输入API密钥")
        else:
            try:
                current_role_config = ROLE_CONFIG[st.session_state.current_role]
                st.session_state.agent = MultiAgentOrchestrator(
                    api_key=api_key.strip(),
                    role_config=current_role_config
                )
                st.success("智能体初始化成功")
                st.session_state.messages = []
            except Exception as e:
                st.error(f"初始化失败：{str(e)}")

    st.divider()

    st.subheader("角色切换")
    role_list = list(ROLE_CONFIG.keys())
    new_role = st.radio(
        "选择当前角色端",
        role_list,
        index=role_list.index(st.session_state.current_role)
    )
    if new_role != st.session_state.current_role:
        st.session_state.current_role = new_role
        st.session_state.agent = None
        st.session_state.messages = []
        st.info("角色已切换，请重新点击「初始化智能体」生效")
        st.rerun()

    st.divider()
    if st.button("重置对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ====================== 主页面 ======================
st.title("园区智能育儿顾问")
st.caption("本工具仅作育儿与运营参考，不替代专业儿科诊疗与官方保育资质")

st.info(f"您好！当前是【{st.session_state.current_role}】，初始化后即可开始咨询~")

st.write("💡 常见问题示例：")
col1, col2 = st.columns(2)
with col1:
    if st.button("10月龄宝宝加新辅食后轻微红疹怎么处理？", use_container_width=True):
        handle_user_query("10月龄宝宝加新辅食后轻微红疹怎么处理？")
        st.rerun()
with col2:
    if st.button("宝宝10个月，准备送托育，你们辅食怎么安排？", use_container_width=True):
        handle_user_query("宝宝10个月，准备送托育，你们辅食怎么安排？")
        st.rerun()

st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("请输入您的问题")
if user_input:
    handle_user_query(user_input)
    st.rerun()
