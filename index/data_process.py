import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def build_knowledge_base():
    # 1. 路径配置 (使用 r 保证路径正确)
    pdf_path = r"E:\大模型RAG\data\handbook.pdf"
    db_path = "../chroma_db"

    if not os.path.exists(pdf_path):
        print(f"错误：找不到文件 {pdf_path}，请检查路径和文件名！")
        return

    # 2. 加载 PDF
    print("正在加载 PDF...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    # 3. 切分文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)
    print(f"切分完成，共得到 {len(split_docs)} 个文本块。")

    # 4. 初始化嵌入模型
    # BGE 模型是目前中文 RAG 的首选
    print("正在初始化嵌入模型 (BAAI/bge-small-zh-v1.5)...")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

    # 5. 构建并持久化向量数据库
    print("正在构建数据库并写入硬盘...")
    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,    #加工工具：告诉数据库用哪个模型把文字转成向量
        persist_directory=db_path
    )
    print(f"恭喜！数据库已成功保存至 {db_path}")


if __name__ == "__main__":
    build_knowledge_base()