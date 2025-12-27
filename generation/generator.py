# generation/generator.py
import re
from typing import List, Dict, Generator

# 添加引用、用户问题重述、上下文记忆、优化提示词，拒绝回答策略

def rephrase_user_query(question: str, conversation_history: List[Dict] = None) -> str:
    """
    重述用户消息，使其更符合检索需求
    """
    # 简单的重述规则，可以根据需要扩展为LLM调用
    rephrase_rules = {
        r'怎么(.*)绩点': '绩点计算规则',
        r'如何(.*)奖学金': '奖学金申请条件',
        r'怎样(.*)处分': '学生处分规定',
        r'什么时候(.*)考试': '考试时间安排',
        r'哪里(.*)成绩': '成绩查询方式'
    }

    for pattern, replacement in rephrase_rules.items():
        if re.search(pattern, question):
            return replacement

    # 如果没有匹配规则，返回原问题
    return question


def is_student_manual_related(question: str) -> bool:
    """
    改进版：提高学业相关词汇的优先级
    """
    question_lower = question.lower()

    # 1. 学业核心词汇（最高优先级）- 只要包含就认为是相关的
    core_academic_keywords = [
        '挂科', '不及格', '成绩', '学分', '绩点', '学业', '考试', '处分', '违纪',
        '毕业', '学位', '选课', '考勤', '学籍', '转专业', '休学', '退学',
        '补考', '重修', '学业警告', '学术不端', '作弊', '违规', '处罚'
    ]

    # 只要包含任何一个核心学业词汇，立即判断为相关
    for keyword in core_academic_keywords:
        if keyword in question_lower:
            return True

    # 2. 严格的不相关词汇（中等优先级）- 只在没有学业词汇时生效
    strong_unrelated_keywords = [
        '天气', '新闻', '股票', '投资', '电影', '音乐', '政治', '经济',
        '国际', '旅游', '美食', '健康', '美容', '时尚', '电视剧', '综艺'
    ]

    # 3. 一般相关词汇（较低优先级）
    general_related_keywords = [
        '宿舍', '图书馆', '实验室', '实习', '毕业论文', '学位证', '毕业证',
        '学费', '助学金', '贷款', '勤工助学', '社团', '学生会', '志愿活动'
    ]

    # 4. 如果包含严格不相关词汇且没有学业核心词汇，才判断为不相关
    has_unrelated = any(keyword in question_lower for keyword in strong_unrelated_keywords)
    if has_unrelated:
        # 虽然有看似不相关的词汇，但如果问题较长可能还是相关的
        return len(question_lower.strip()) > 8  # 给长问题更多机会

    # 5. 检查一般相关词汇
    if any(keyword in question_lower for keyword in general_related_keywords):
        return True

    # 6. 语义模式匹配（提高对复杂问题的识别）
    academic_patterns = [
        r'.*导致.*挂科', r'.*影响.*成绩', r'.*有什么.*后果', r'.*会.*处分',
        r'.*如果.*会.*', r'.*要是.*会.*', r'.*什么.*规定', r'.*如何.*处理',
        r'.*会不会.*影响', r'.*有没有.*关系'
    ]

    for pattern in academic_patterns:
        if re.search(pattern, question_lower):
            return True

    # 7. 默认策略：给模糊问题机会
    return len(question_lower.strip()) > 5

def build_prompt_with_history(prompt_template, question, context, conversation_history=None):
    """
    简化版提示词构建：移除引用参数
    """
    # 构建对话历史上下文
    history_context = ""
    if conversation_history:
        # 只保留最近的对话作为历史，可以自己设置
        recent_history = conversation_history[-10:]  # 目前是最近5轮（每轮2条消息）
        for msg in recent_history:
            role = "用户" if msg["role"] == "user" else "助手"
            history_context += f"{role}: {msg['content']}\n"

    # 简化版提示词构建（移除citations参数）
    prompt = prompt_template.format(
        question=question,
        context=context,
        conversation_history=history_context if history_context else "无"
    )

    return prompt


def generate_answer(llm, prompt_template, question, context, conversation_history=None):
    """
    简化版生成函数：移除引用功能
    """
    # 1. 问题相关性判断
    if not is_student_manual_related(question):
        return "抱歉，我是北京交通大学学生手册助手，主要解答与学生规章制度相关的问题。如果您有其他类型的问题，建议咨询相关部门。"

    # 2. 重述用户问题
    rephrased_question = rephrase_user_query(question)

    # 3. 构建提示词（不再包含引用信息）
    prompt = build_prompt_with_history(
        prompt_template,
        rephrased_question,
        context,
        conversation_history
    )

    response = llm.invoke(prompt)
    return response.content


def generate_answer_stream(llm, prompt_template, question, context, conversation_history=None):
    """
    简化版流式生成函数：移除引用功能
    """
    # 1. 问题相关性判断
    if not is_student_manual_related(question):
        yield "抱歉，我是北京交通大学学生手册助手，主要解答与学生规章制度相关的问题。如果您有其他类型的问题，建议咨询相关部门。"
        return

    # 2. 重述用户问题
    rephrased_question = rephrase_user_query(question, conversation_history)

    # 3. 构建提示词（不再包含引用信息）
    prompt = build_prompt_with_history(
        prompt_template,
        rephrased_question,
        context,
        conversation_history
    )

    # 4. 流式生成
    response_stream = llm.stream(prompt)

    for chunk in response_stream:
        if hasattr(chunk, 'content'):
            yield chunk.content


def build_prompt(prompt_template, question, context):
    """
    保持原有的简单提示词构建函数（兼容性）
    """
    return prompt_template.format(question=question, context=context)