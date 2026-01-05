"""
Query Flow - RAG 질의 워크플로우
LangGraph StateGraph 조립
"""

from typing import Dict, Any, Generator, Literal

from langgraph.graph import StateGraph, END

from graph.state import RAGState
from graph.cache import cache_lookup, cache_store
from graph.query import (
    plan_query,
    retrieve_chunks,
    expand_neighbors,
    rerank_chunks,
    judge_sufficiency,
    generate_answer,
)
from graph.grader import grade_hallucination
from settings import NodeLogger, logger


log = NodeLogger("QueryFlow")


# === Conditional Edge Functions ===

def check_cache_hit(state: RAGState) -> Literal["cache_hit", "plan"]:
    """캐시 히트 여부 확인"""
    if state.get("cache_hit"):
        return "cache_hit"
    return "plan"


def should_loop(state: RAGState) -> Literal["loop", "generate"]:
    """반복 여부 결정"""
    if state.get("should_continue"):
        return "loop"
    return "generate"


def handle_hallucination(state: RAGState) -> Literal["rewrite", "done"]:
    """환각 처리 분기"""
    hallu = state.get("hallucination")
    if hallu and not hallu.grounded and state.get("should_continue"):
        return "rewrite"
    return "done"


# === Build Graph ===

def build_query_graph() -> StateGraph:
    """Query 워크플로우 그래프 빌드"""
    
    builder = StateGraph(RAGState)
    
    # === Add Nodes ===
    builder.add_node("cache_lookup", cache_lookup)
    builder.add_node("plan", plan_query)
    builder.add_node("retrieve", retrieve_chunks)
    builder.add_node("expand", expand_neighbors)
    builder.add_node("rerank", rerank_chunks)
    builder.add_node("judge", judge_sufficiency)
    builder.add_node("generate", generate_answer)
    builder.add_node("grade", grade_hallucination)
    builder.add_node("cache_store", cache_store)
    
    # === Define Edges ===
    
    # Start -> Cache Lookup
    builder.set_entry_point("cache_lookup")
    
    # Cache Lookup -> (hit: end, miss: plan)
    builder.add_conditional_edges(
        "cache_lookup",
        check_cache_hit,
        {
            "cache_hit": END,
            "plan": "plan"
        }
    )
    
    # Plan -> Retrieve
    builder.add_edge("plan", "retrieve")
    
    # Retrieve -> Expand
    builder.add_edge("retrieve", "expand")
    
    # Expand -> Rerank
    builder.add_edge("expand", "rerank")
    
    # Rerank -> Judge
    builder.add_edge("rerank", "judge")
    
    # Judge -> (loop: retrieve, done: generate)
    builder.add_conditional_edges(
        "judge",
        should_loop,
        {
            "loop": "retrieve",
            "generate": "generate"
        }
    )
    
    # Generate -> Grade
    builder.add_edge("generate", "grade")
    
    # Grade -> (rewrite: generate, done: cache_store)
    builder.add_conditional_edges(
        "grade",
        handle_hallucination,
        {
            "rewrite": "generate",
            "done": "cache_store"
        }
    )
    
    # Cache Store -> End
    builder.add_edge("cache_store", END)
    
    return builder.compile()


# === Graph Instance ===
_query_graph = None


def get_query_graph():
    """싱글톤 그래프 인스턴스"""
    global _query_graph
    if _query_graph is None:
        _query_graph = build_query_graph()
    return _query_graph


# === Public API ===

def run_query(query: str) -> Dict[str, Any]:
    """
    쿼리 실행 (동기)
    
    Returns:
        {answer, evidence, flow_log, ...}
    """
    log.enter(f"query='{query[:50]}...'")
    
    initial_state: RAGState = {
        "query": query,
        "iteration": 0,
        "max_iters": 3,
        "candidates": [],
        "pool_high": [],
        "pool_all": [],
        "missing_aspects": [],
        "cache_hit": False,
        "should_continue": False,
        "flow_log": [],
    }
    
    graph = get_query_graph()
    result = graph.invoke(initial_state, config={"recursion_limit": 50})
    
    # Flow log 출력
    flow_log = result.get("flow_log", [])
    logger.info("=== Flow Log ===")
    for entry in flow_log:
        logger.info(f"  {entry}")
    
    log.exit(f"answer_len={len(result.get('answer', ''))}")
    
    return result


def stream_query(query: str) -> Generator[str, None, None]:
    """
    쿼리 실행 (스트리밍)
    
    Note: 현재는 전체 실행 후 답변 반환
    TODO: 실제 스트리밍 구현
    """
    result = run_query(query)
    
    answer = result.get("answer", "")
    if answer:
        # 청크 단위로 yield (시뮬레이션)
        chunk_size = 20
        for i in range(0, len(answer), chunk_size):
            yield answer[i:i+chunk_size]
    else:
        yield "관련 문서를 찾을 수 없습니다."
