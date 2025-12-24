# generation/generator.py

def generate_answer(llm, prompt_template, question, context):
    """
    生成模块：
    - Prompt 注入
    - LLM 调度
    - （需要再加）重述用户消息 + 用户对话记录（多轮对话记忆） + 拒答策略
    """
    prompt = prompt_template.format(
        context=context,
        question=question
    )
    response = llm.invoke(prompt)
    return response
