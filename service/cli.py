# service/cli.py
from retrieval.retriever import retrieve_docs, build_context
from generation.generator import generate_answer
from generation.prompt import PROMPT_TEMPLATE


def run_cli(vector_db, llm):
    eval_buffer = []

    print("我是北京交通大学学生手册小助手，请问您有什么问题？")

    while True:
        question = input("\n提问（quit 退出）：")
        if question.lower() in ["quit", "exit", "退出"]:
            break

        docs = retrieve_docs(vector_db, question)
        context = build_context(docs)
        answer = generate_answer(llm, PROMPT_TEMPLATE, question, context)

        print("\n[助手回答]\n", answer)
