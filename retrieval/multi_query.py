# retrieval/multi_query.py
# 检索前处理：多查询优化

from langchain_core.prompts import PromptTemplate


MULTI_QUERY_PROMPT = PromptTemplate.from_template("""
你是一名【信息检索查询优化助手】。

你的任务是：  
基于用户的原始问题，生成 {n} 条用于向量检索的查询语句，
以便从《北京交通大学学生手册》这类高校规章制度文本中检索相关条款。

生成要求：
1. 每条查询应与原问题语义相关，但应从不同的表述角度进行改写
   （例如：制度性表述、书面条款式表述、规范性描述等）
2. 尽量使用 正式、规范、制度化 的语言，避免口语化或对话式表达
3. 不要引入原问题中未明确提及的事实、条件或假设
4. 不要给出解释、分析、编号或多余说明文字
5. 每行仅输出一条完整的检索查询语句

用户原始问题：
{question}

生成的检索查询：
""")


def generate_multi_queries(llm, question, n=5):
    """
    使用 LLM 生成多条检索查询
    """
    prompt = MULTI_QUERY_PROMPT.format(
        question=question,
        n=n
    )

    response = llm.invoke(prompt)

    # 转为字符串并按行切分
    queries = [
        q.strip()
        for q in response.content.split("\n")
        if q.strip()
    ]

    # 去重 + 保留原始问题
    queries = list(dict.fromkeys([question] + queries))
    print("多查询生成的查询语句：")
    print(response.content)

    return queries
