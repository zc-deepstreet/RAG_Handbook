from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    ContextRelevance,
    answer_relevancy,
    faithfulness,
    ResponseGroundedness,
)


def build_ragas_dataset(eval_records):
    """
    将 RAG 推理过程中的数据，整理为 RAGAS 所需的数据格式
    """
    data = {
        "user_input": [r["query"] for r in eval_records],
        "response": [r["answer"] for r in eval_records],
        "retrieved_contexts": [r["contexts"] for r in eval_records],
    }
    return Dataset.from_dict(data)


def evaluate_rag_system(
        eval_records,
        llm,
        embeddings,
):
    """
    使用 RAGAS 对 BJTU 学生手册 RAG 系统进行评估
    """

    dataset = build_ragas_dataset(eval_records)

    metrics = [
        ContextRelevance(),      # 检索是否相关
        answer_relevancy,        # 回答是否对题
        faithfulness,            # 是否忠实于资料
        ResponseGroundedness(),  # 是否有依据
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    return result
