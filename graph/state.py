"""
RAG State - LangGraph 상태 정의
"""

from typing import TypedDict, List, Dict, Any, Optional

from models.query import QueryPlan
from models.answer import AnswerRecord, SufficiencyResult, HallucinationResult


class RetrievedChunk(TypedDict):
    """검색된 청크"""
    chunk_id: str
    doc_id: str
    text: str
    score: float
    weighted_score: float
    metadata: Dict[str, Any]


class RAGState(TypedDict, total=False):
    """RAG 워크플로우 상태"""
    
    # === Input ===
    query: str
    
    # === Query Planning ===
    query_plan: Optional[QueryPlan]
    current_subquery: Optional[str]
    subquery_index: int
    
    # === Retrieval ===
    candidates: List[RetrievedChunk]      # 현재 검색 결과
    pool_high: List[RetrievedChunk]       # 높은 점수 유지 풀
    pool_all: List[RetrievedChunk]        # 모든 검색 결과 (중복 제거)
    
    # === Loop Control ===
    iteration: int
    max_iters: int
    missing_aspects: List[str]
    
    # === Sufficiency ===
    sufficiency: Optional[SufficiencyResult]
    
    # === Answer ===
    answer: Optional[str]
    evidence: List[Dict[str, Any]]
    
    # === Hallucination ===
    hallucination: Optional[HallucinationResult]
    
    # === Cache ===
    cache_hit: bool
    cached_answer: Optional[str]
    
    # === Flow Control ===
    should_continue: bool
    error: Optional[str]
    
    # === Logging ===
    flow_log: List[str]  # 노드 흐름 기록
