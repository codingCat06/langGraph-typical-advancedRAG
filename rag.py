"""
RAG - Entry Point
LangGraph 워크플로우 호출
"""

from typing import Generator, Dict, Any

from workflows import run_query, stream_query
from settings import logger


def get_response(question: str) -> str:
    """동기 응답"""
    result = run_query(question)
    return result.get("answer", "답변을 생성할 수 없습니다.")


def stream_response(question: str) -> Generator[str, None, None]:
    """스트리밍 응답"""
    for chunk in stream_query(question):
        yield chunk


def get_response_with_metadata(question: str) -> Dict[str, Any]:
    """
    메타데이터 포함 응답
    
    Returns:
        {
            answer: str,
            evidence: list,
            flow_log: list,
            sufficiency: dict,
            hallucination: dict,
            ...
        }
    """
    result = run_query(question)
    return {
        "answer": result.get("answer", ""),
        "evidence": result.get("evidence", []),
        "flow_log": result.get("flow_log", []),
        "sufficiency": result.get("sufficiency"),
        "hallucination": result.get("hallucination"),
        "iterations": result.get("iteration", 1),
        "query_plan": result.get("query_plan"),
    }


# 하위 호환성을 위한 alias
retrieve_with_sources = None  # 기존 함수 제거됨


if __name__ == "__main__":
    # 테스트
    question = "Transformer 모델의 self-attention이란?"
    print(f"Q: {question}")
    print()
    print("A:", get_response(question))
