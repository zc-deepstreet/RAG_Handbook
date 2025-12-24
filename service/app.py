import streamlit as st
import os
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from retrieval.retriever import retrieve_docs, build_context
from generation.generator import generate_answer
from generation.prompt import PROMPT_TEMPLATE
import dotenv
from langchain_openai import ChatOpenAI
import os
from evaluation.rag_evaluator import evaluate_rag_system


# --- 1. 页面配置与自定义样式 ---
st.set_page_config(page_title="北交大学生手册助手", page_icon="🏫", layout="centered")

# 注入 CSS 修改侧边栏颜色为深蓝色 (#003366 是深蓝系颜色)
# --- 1. 页面配置与自定义样式 ---
st.set_page_config(page_title="北交大学生手册助手", page_icon="🏫", layout="centered")

# 注入 CSS：修改侧边栏背景为深蓝，并确保按钮文字为黑色
st.markdown("""
    <style>
        /* 修改侧边栏背景颜色 */
        [data-testid="stSidebar"] {
            background-color: #003366;
        }

        /* 修改侧边栏内的标题和普通文字为白色 */
        [data-testid="stSidebar"] .stText, 
        [data-testid="stSidebar"] p, 
        [data-testid="stSidebar"] h1, 
        [data-testid="stSidebar"] h2, 
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] span {
            color: white !important;
        }

        /* 核心修改：强制侧边栏按钮背景为纯白，字体颜色为黑色 */
        [data-testid="stSidebar"] .stButton button {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: none;
            font-weight: bold;
        }

        /* 鼠标悬停在按钮上时的效果（可选，增加交互感） */
        [data-testid="stSidebar"] .stButton button:hover {
            background-color: #eeeeee !important;
            color: #000000 !important;
        }
    </style>
    """, unsafe_allow_html=True)

# --- 2. 加载后端模型 (保持不变) ---
@st.cache_resource
def init_models():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'}
    )
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(BASE_DIR, "chroma_db")
    if not os.path.exists(db_path):
        return None, None
    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

    dotenv.load_dotenv()  #加载当前目录下的 .env 文件
    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
    os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")
    # 创建大模型实例
    llm = ChatOpenAI(model="gpt-4o-mini")  # 默认使用
    # llm = OllamaLLM(model="deepseek-r1:8b", temperature=0.1)
    return vector_db, llm


vector_db, llm = init_models()

# --- 3. 页面标题 ---
st.title("🏫 北京交通大学")
st.subheader("学生手册智能咨询助手")
st.markdown("---")

# --- 4. 侧边栏内容 (深蓝背景) ---
with st.sidebar:
    # 建议：如果你本地有校徽图片，可以用 st.image("logo.png")
    # 这里先用北交大官网的透明底校徽链接（如果链接失效请替换为本地路径）
    st.image("https://www.bjtu.edu.cn/images/logo.png", use_container_width=True)

    st.markdown("### 系统概况")
    st.write("欢迎使用！本助手基于 DeepSeek-R1 推理模型，专门为您解答《学生手册》相关问题。")

    st.markdown("---")
    st.info("💡 **提示**：关于绩点、奖学金、处分的规定，系统已深度学习。")

    if st.button("清空对话记录"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.markdown("### 📊 系统评估")

    if st.button("运行 RAG 评估"):
        if len(st.session_state.eval_buffer) == 0:
            st.warning("⚠️ 当前还没有可评估的问答记录")
        else:
            with st.spinner("正在评估系统性能，请稍候..."):
                try:
                    eval_result = evaluate_rag_system(
                        eval_records=st.session_state.eval_buffer,
                        llm=llm,
                        embeddings=vector_db._embedding_function,
                    )
                    st.success("评估完成！")
                    df = eval_result.to_pandas()
                    st.dataframe(
                        df[
                            [
                                "nv_context_relevance",
                                "answer_relevancy",
                                "faithfulness",
                                "nv_response_groundedness",
                            ]
                        ],
                        use_container_width=True,
                    )

                except Exception as e:
                    st.error(f"评估失败：{e}")

# --- 5. 对话逻辑 (保持不变) ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "同学你好！我是北京交通大学学生手册小助手，有什么我可以帮你的吗？"}
    ]

if "eval_buffer" not in st.session_state:
    st.session_state.eval_buffer = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("请输入您的问题..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("正在检索校规并思考..."):
            try:
                # ① 检索
                docs = retrieve_docs(vector_db, user_query)
                # ② 构建上下文
                context = build_context(docs)
                # ③ 生成回答
                answer = generate_answer(
                    llm,
                    PROMPT_TEMPLATE,
                    user_query,
                    context
                )
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
                # ④ 评估指标
                st.session_state.eval_buffer.append({
                    "query": user_query,
                    "answer": answer,
                    "contexts": [doc.page_content for doc in docs],
                })
            except Exception as e:
                st.error(f"出错啦: {e}")