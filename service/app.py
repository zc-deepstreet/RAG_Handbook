import sys
import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from retrieval.retriever import retrieve_docs, build_context
from generation.generator import generate_answer, build_prompt
from generation.prompt import PROMPT_TEMPLATE
import dotenv
from langchain_openai import ChatOpenAI
import os
from evaluation.rag_evaluator import evaluate_rag_system
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# --- 0. 路径与基础设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 确保路径下包含这些自定义模块
try:
    from retrieval.retriever import retrieve_docs, build_context
    from evaluation.rag_evaluator import evaluate_rag_system
except ImportError:
    pass

# 【定义核心变量】
WELCOME_MSG = "我是北京交通大学学生手册小助手，可以帮你解答有关学习规章制度的问题。请问有什么我可以帮你的吗？"
# --- 1. 页面配置与 UI 优化 CSS ---
st.set_page_config(page_title="北交大学生手册助手", page_icon="🏫", layout="wide")

st.markdown("""
    <style>
        /* 1. 全局字体与侧边栏背景 */
        html, body, [class*="st-"] {
            font-family: "PingFang SC", "Microsoft YaHei", sans-serif !important;
        }
        [data-testid="stSidebar"] {
            background-color: #003366;
        }

        /* 2. 侧边栏组件间距优化 */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.5rem !important;
        }

        /* 3. 系统概况描述框样式 */
        .system-info-card {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 15px;
            margin: 10px 0 20px 0;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        .system-info-card h3 {
            color: white !important;
            font-size: 1.1rem !important;
            margin-bottom: 8px !important;
        }
        .system-info-card p {
            color: rgba(255, 255, 255, 0.8) !important;
            font-size: 0.9rem !important;
            line-height: 1.5 !important;
        }
        .tip-box {
            background-color: rgba(173, 216, 230, 0.1);
            border-radius: 8px;
            padding: 10px;
            margin-top: 10px;
            font-size: 0.85rem;
            color: #ADD8E6;
            border-left: 3px solid #ADD8E6;
        }

        /* 4. 侧边栏标题 */
        .sidebar-title {
            color: rgba(255, 255, 255, 0.6) !important;
            font-size: 0.85rem !important;
            font-weight: 600 !important;
            text-transform: uppercase;
            margin: 20px 0 8px 5px !important;
        }

        /* 5. 按钮基础样式：彻底去框、无背景、全宽度 */
        [data-testid="stSidebar"] .stButton button {
            background-color: transparent !important;
            border: none !important;
            box-shadow: none !important;
            color: #ffffff !important;
            width: 100% !important;
            height: 42px !important;
            padding: 0px 15px !important;
            border-radius: 10px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: flex-start !important;
            text-align: left !important;
            transition: all 0.2s;
        }

        /* 6. 开启新对话按钮专属加粗 */
        button[key="new_chat_btn"] {
            background-color: rgba(255, 255, 255, 0.1) !important;
            margin-bottom: 10px !important;
        }
        button[key="new_chat_btn"] p {
            font-weight: 700 !important;
        }

        /* 7. 历史对话按钮文字：溢出省略 */
        [data-testid="stSidebar"] .stButton button div p {
            color: inherit !important;
            font-size: 0.95rem !important;
            margin: 0 !important;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        /* 8. 选中状态高亮：DeepSeek 浅蓝风格 */
        [data-testid="stSidebar"] .stButton button:hover {
            background-color: rgba(255, 255, 255, 0.1) !important;
        }

        /* 分割线 */
        hr {
            margin: 20px 0 !important;
            opacity: 0.1 !important;
        }
    </style>
    """, unsafe_allow_html=True)


# --- 2. 初始化模型 ---
@st.cache_resource
def init_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    db_path = os.path.join(project_root, "chroma_db")
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

    dotenv.load_dotenv()
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0.3)

    rerank_model_name = "BAAI/bge-reranker-base"
    rerank_tokenizer = AutoTokenizer.from_pretrained(rerank_model_name)
    rerank_model = AutoModelForSequenceClassification.from_pretrained(
        rerank_model_name
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rerank_model.to(device)
    rerank_model.eval()
    return vector_db, llm, rerank_model, rerank_tokenizer


vector_db, llm, rerank_model, rerank_tokenizer = init_models()

# --- 3. 状态管理 ---
if "sessions" not in st.session_state:
    st.session_state.sessions = [{"name": "新对话", "messages": [{"role": "assistant", "content": WELCOME_MSG}]}]
if "active_session_idx" not in st.session_state:
    st.session_state.active_session_idx = 0
if "eval_buffer" not in st.session_state:
    st.session_state.eval_buffer = []

# --- 4. 侧边栏内容 ---
with st.sidebar:
    # 学校 Logo
    st.image("https://www.bjtu.edu.cn/images/logo.png", use_container_width=True)

    # 系统概况描述块 (仿照图片)
    st.markdown("""
    <div class="system-info-card">
        <h3>系统概况</h3>
        <p>欢迎使用！本助手基于 <b>GPT-4o / RAG</b> 架构，专门为您解答《学生手册》相关问题。</p>
        <div class="tip-box">
            💡 提示：关于绩点、奖学金、处分的规定，系统已深度学习。
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 【1. 开启新对话】
    if st.button("✨ 开启新对话", key="new_chat_btn"):
        st.session_state.sessions.insert(0, {"name": "新对话",
                                             "messages": [{"role": "assistant", "content": WELCOME_MSG}]})
        st.session_state.active_session_idx = 0
        st.rerun()

    # 【2. 历史对话列表】
    st.markdown('<p class="sidebar-title">最近对话</p>', unsafe_allow_html=True)
    for idx, session in enumerate(st.session_state.sessions):
        is_active = (idx == st.session_state.active_session_idx)

        # 选中高亮逻辑
        if is_active:
            st.markdown(f"""<style>button[key="s_btn_{idx}"] {{ 
                background-color: #ADD8E6 !important; 
                color: #000000 !important; 
                font-weight: 600 !important; 
            }}</style>""", unsafe_allow_html=True)

        # 移除了删除按钮列，直接展示全宽对话按钮
        if st.button(session['name'], key=f"s_btn_{idx}"):
            st.session_state.active_session_idx = idx
            st.rerun()

    st.markdown("---")

    # 【3. 系统操作】
    if st.button("🧹 清空当前对话内容", key="clear_chat_btn"):
        st.session_state.sessions[st.session_state.active_session_idx]["messages"] = [
            {"role": "assistant", "content": WELCOME_MSG}]
        st.rerun()

    if st.button("🚀 运行 RAG 评估", key="run_eval_btn"):
        if st.session_state.eval_buffer:
            with st.spinner("评估中..."):
                eval_result = evaluate_rag_system(st.session_state.eval_buffer, llm, vector_db._embedding_function)
                st.dataframe(eval_result.to_pandas(), use_container_width=True)

# --- 5. 主界面渲染 ---
current_session = st.session_state.sessions[st.session_state.active_session_idx]
st.title("🏫 北京交通大学学生手册助手")

# 聊天记录显示
for msg in current_session["messages"]:
    icon = "🤖" if msg["role"] == "assistant" else "👤"
    with st.chat_message(msg["role"], avatar=icon):
        st.markdown(msg["content"])

# --- 6. 聊天输入逻辑 ---
if prompt := st.chat_input("请输入您的问题..."):
    if current_session["name"] == "新对话":
        current_session["name"] = (prompt[:12] + '..') if len(prompt) > 12 else prompt

    current_session["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        # ① 检索
        # docs = retrieve_docs(vector_db,
        #                      prompt,
        #                      llm=llm,
        #                      use_multi_query=True,
        #                      use_hyde=True)
        docs = retrieve_docs(
            vector_db,
            prompt,
            llm=llm,
            k=30,
            fetch_k=60,
            use_multi_query=True,
            use_hyde=True,
            use_rrf=True,
            use_model_rerank=True,
            rerank_model=rerank_model,
            rerank_tokenizer=rerank_tokenizer,
            final_top_n=6,
        )
        # ② 构建上下文
        context = build_context(docs)

        def stream_response():
            full_prompt = build_prompt(PROMPT_TEMPLATE, prompt, context)
            for chunk in llm.stream(full_prompt):
                yield chunk.content

        # ③ 生成回答
        full_answer = st.write_stream(stream_response())
        current_session["messages"].append({"role": "assistant", "content": full_answer})

        # 记录评估数据
        st.session_state.eval_buffer.append({
            "query": prompt,
            "answer": full_answer,
            "contexts": [d.page_content for d in docs]
        })
