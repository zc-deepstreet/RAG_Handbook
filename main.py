import os
# 导入最新的 Ollama 接口，消除警告
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import dotenv
from langchain_openai import ChatOpenAI
import os


def run_bjtu_assistant():
    # 1. 初始化设置
    print("系统启动中，正在加载北京交通大学知识库...")

    # 显卡配置：确保使用你的 4070
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={'device': 'cpu'}
    )

    # 2. 加载向量数据库
    db_path = "chroma_db"
    if not os.path.exists(db_path):
        print(f"错误：找不到数据库文件夹 {db_path}，请先运行 data_process.py")
        return

    vector_db = Chroma(persist_directory=db_path, embedding_function=embeddings)

    # 3. 调用大模型（使用gpt-4o-mini）
    dotenv.load_dotenv()  #加载当前目录下的 .env 文件

    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
    os.environ['OPENAI_BASE_URL'] = os.getenv("OPENAI_BASE_URL")

    # 创建大模型实例
    llm = ChatOpenAI(model="gpt-4o-mini")  # 默认使用
    #llm = OllamaLLM(model="deepseek-r1:8b", temperature=0.1)  # 低温使回答更严谨、保守。它每次都会倾向于选择概率最高的词

    # 4. 定义增强型 Prompt 模板（设定规则）
    # 加入了语义推导指令
    template = """你现在是【北京交通大学学生手册小助手】。

    任务说明：
    1. 请根据下方提供的【参考资料】回答学生的问题。
    2. 资料中的用词可能与提问不同（如“绩点”对应“GPA”，“违禁”对应“违规”），请进行语义关联。
    3. 如果资料中确实没有相关规定，请回答：“抱歉，在《北京交通大学学生手册》中未查询到相关条文。”
    4. 严禁编造不存在的校规，回答应保持礼貌和专业。

    【参考资料】：
    {context}

    【学生问题】：{question}

    【助手的详细回答】："""

    prompt_temp = PromptTemplate.from_template(template)

    # 5. 开场白
    print("\n" + "=" * 50)
    print("我是北京交通大学学生手册小助手，请问您有什么问题需要帮助？")
    print("=" * 50)

    while True:
        user_query = input("\n同学提问 (输入 quit 退出)：")
        if user_query.lower() in ['quit', 'exit', '退出']:
            print("再见，祝你学习顺利！")
            break

        if not user_query.strip():
            continue

        print("正在为您翻阅手册并进行深度推理...")

        try:
            # --- 核心改进：MMR 检索 (最大边际相关性) ---
            # fetch_k=20：先在大数据库里撒网，捞出 20 条语义相关的片段。
            # k=6：在这 20 条里进行“二次筛选”，选出 6 条既相关、彼此之间内容又不重复 的片段。
            docs = vector_db.max_marginal_relevance_search(
                user_query,
                k=6,
                fetch_k=20
            )

            # 如果想展示系统找到了哪些依据，可以开启下面的 print
            # print(f"DEBUG: 检索到 {len(docs)} 个片段。")

            # 拼接参考上下文
            context_text = ""
            for i, doc in enumerate(docs):
                context_text += f"\n--- 资料片段 {i + 1} ---\n{doc.page_content}\n"

            # 构建最终 Prompt
            formatted_prompt = prompt_temp.format(context=context_text, question=user_query)

            # 6. 调用大模型生成回答
            response = llm.invoke(formatted_prompt)

            # 输出最终结果
            print(f"\n[助手回复]：\n{response}")

        except Exception as e:
            print(f"遇到一点小麻烦：{e}")


if __name__ == "__main__":
    run_bjtu_assistant()