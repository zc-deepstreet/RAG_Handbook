# retrieval/hyde.py

from langchain_core.prompts import PromptTemplate


HYDE_PROMPT = PromptTemplate.from_template(
 """
你是一名高校学生管理规章文本生成助手。

请基于学生提出的问题，生成一段 可能原样出现在《北京交通大学学生手册》中的制度性条文片段，
用于向量检索辅助，而不是用于直接回答学生问题。

生成要求（非常重要）：
1. 使用第三人称、客观、中性的制度语言
2. 不要使用“以下”“建议”“应当如何做”等总结或指导性语句
3. 不要分点、不编号、不下结论
4. 不要出现“补救措施”“解决办法”等回答式措辞
5. 内容应描述“在某种学业情形下，学校通常如何管理或处理”，而非对学生提出建议
6. 文本风格应接近高校学生手册中的原始条款表述

学生问题：
{question}

假设的学生手册条文片段：
"""
)


def generate_hypothetical_doc(llm, question):
    """
    使用 HyDE 生成假设文档，用于向量检索
    """
    prompt = HYDE_PROMPT.format(question=question)
    response = llm.invoke(prompt)
    print()
    print("假设文档嵌入生成的回答语句：")
    print(response.content)

    return response.content.strip()
