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


def calculate_academic_weight(question: str) -> float:
    """
    计算问题中学业相关词汇的权重
    """
    question_lower = question.lower()

    # 定义词汇权重
    keyword_weights = {
        # 高权重学业词汇
        '挂科': 2.0, '不及格': 2.0, '处分': 1.8, '违纪': 1.8, '作弊': 1.8,
        '成绩': 1.5, '学分': 1.5, '毕业': 1.5, '学位': 1.5,
        # 中等权重
        '考试': 1.2, '选课': 1.2, '考勤': 1.2, '学业': 1.2,
        # 低权重但相关
        '宿舍': 0.8, '图书馆': 0.8, '实习': 0.8, '奖学金': 0.8
    }

    total_weight = 0
    for keyword, weight in keyword_weights.items():
        if keyword in question_lower:
            total_weight += weight

    return total_weight


def is_student_manual_related_weighted(question: str) -> bool:
    """
    基于权重判断的相关性函数
    """
    # 计算学业权重
    academic_weight = calculate_academic_weight(question)

    # 如果学业权重足够高，直接认为是相关的
    if academic_weight >= 1.0:
        return True

    question_lower = question.lower()

    # 严格不相关检查
    strong_unrelated = ['天气', '新闻', '股票', '投资', '电影', '音乐']
    if any(keyword in question_lower for keyword in strong_unrelated):
        return False

    # 中等学业权重也给机会
    if academic_weight >= 0.5:
        return True

    # 语义模式匹配
    academic_patterns = [r'.*导致.*挂科', r'.*影响.*成绩', r'.*有什么.*后果']
    for pattern in academic_patterns:
        if re.search(pattern, question_lower):
            return True

    return len(question_lower.strip()) > 5


def format_citations(docs: List) -> str:
    """
    格式化引用来源信息 - 修复文件路径显示问题
    """
    if not docs:
        return ""

    citations = []
    for i, doc in enumerate(docs, 1):
        # 尝试从文档元数据中提取来源信息
        source = getattr(doc, 'metadata', {}).get('source', '未知章节')
        page = getattr(doc, 'metadata', {}).get('page', '')

        # 修复：清理文件路径，提取有意义的章节名称
        source_info = clean_source_path(source, str(doc.page_content))

        citation_text = f"[{i}]《北京交通大学学生手册》{source_info}"
        if page:
            citation_text += f" 第{page}条"
        citations.append(citation_text)

    return "  \n".join(citations)


def clean_source_path(source: str, content: str) -> str:
    """
    清理文件路径，返回有意义的章节名称
    """
    # 如果源信息包含文件路径特征，进行清理
    if 'handbook.pdf' in source or '大模型RAG' in source or '\\' in source:
        # 基于内容分析返回有意义的章节名称
        content_lower = content.lower()

        if any(keyword in content_lower for keyword in ['绩点', '学分', '成绩']):
            return "学业成绩管理规定"
        elif any(keyword in content_lower for keyword in ['奖学金', '助学金', '资助']):
            return "奖学金与资助政策"
        elif any(keyword in content_lower for keyword in ['处分', '违纪', '处罚']):
            return "学生违纪处分规定"
        elif any(keyword in content_lower for keyword in ['毕业', '学位']):
            return "毕业与学位授予规定"
        elif any(keyword in content_lower for keyword in ['宿舍', '住宿']):
            return "学生宿舍管理规定"
        elif any(keyword in content_lower for keyword in ['请假', '考勤']):
            return "学生考勤与请假规定"
        else:
            return "学生管理相关规定"  # 默认名称
    else:
        # 如果不是文件路径，直接使用源信息
        return source


def build_prompt_with_history(prompt_template, question, context, conversation_history=None, citations=""):
    """
    构建包含对话历史和引用信息的提示词
    """
    # 构建对话历史上下文
    history_context = ""
    if conversation_history:
        # 只保留最近3轮对话作为历史
        recent_history = conversation_history[-6:]  # 最近3轮（每轮2条消息）
        for msg in recent_history:
            role = "用户" if msg["role"] == "user" else "助手"
            history_context += f"{role}: {msg['content']}\n"

    # 完整的提示词构建
    prompt = prompt_template.format(
        question=question,
        context=context,
        conversation_history=history_context if history_context else "无",
        citations=citations
    )

    return prompt


def generate_answer(llm, prompt_template, question, context, conversation_history=None, docs=None):
    """
    增强的生成函数：支持多轮对话和引用标注
    """
    # 1. 问题相关性判断
    if not is_student_manual_related(question):
        return "抱歉，我是北京交通大学学生手册助手，主要解答与学生规章制度相关的问题。如果您有其他类型的问题，建议咨询相关部门。"

    # 2. 重述用户问题（可选，根据需要开启）
    rephrased_question = rephrase_user_query(question)

    # 3. 格式化引用信息
    citations_text = format_citations(docs) if docs else ""

    # 4. 构建增强提示词
    prompt = build_prompt_with_history(
        prompt_template,
        rephrased_question,  # 使用重述后的问题
        context,
        conversation_history,
        citations_text
    )

    response = llm.invoke(prompt)
    return response.content


def generate_answer_stream(llm, prompt_template, question, context, conversation_history=None, docs=None):
    """
    增强的流式生成函数
    """
    # 1. 问题相关性判断
    if not is_student_manual_related_weighted(question):
        yield "抱歉，我是北京交通大学学生手册助手，主要解答与学生规章制度相关的问题。如果您有其他类型的问题，建议咨询相关部门。"
        return

    # 2. 重述用户问题
    rephrased_question = rephrase_user_query(question, conversation_history)

    # 3. 格式化引用信息
    citations_text = format_citations(docs) if docs else ""

    # 4. 构建增强提示词
    prompt = build_prompt_with_history(
        prompt_template,
        rephrased_question,
        context,
        conversation_history,
        citations_text
    )

    # 5. 流式生成
    response_stream = llm.stream(prompt)
    full_response = ""

    for chunk in response_stream:
        if hasattr(chunk, 'content'):
            content = chunk.content
            full_response += content
            yield content
        else:
            yield chunk

    # 6. 在流式输出后添加引用信息（如果需要单独显示）
    if citations_text and "引用" not in full_response:
        yield f"\n\n---\n**参考来源：**\n{citations_text}"


def build_prompt(prompt_template, question, context):
    """
    保持原有的简单提示词构建函数（兼容性）
    """
    return prompt_template.format(question=question, context=context)