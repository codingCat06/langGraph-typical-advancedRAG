"""
Query Models - 쿼리 계획
"""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field
import uuid


class StopCondition(BaseModel):
    """반복 종료 조건"""
    min_sufficiency: float = 0.7
    max_iters: int = 3


class QueryPlan(BaseModel):
    """
    Self-Query Planner 출력 (Soft)
    """
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    raw_query: str = ""  # LLM이 반환 안 할 수 있음
    normalized_query: str = ""
    intent_hint: Literal["explain", "howto", "troubleshoot", "compare", "extract", "unknown"] = "unknown"
    answer_style: Literal["short", "normal", "detailed"] = "normal"
    constraints: List[str] = Field(default_factory=list)  # 예: ["cite_pages", "include_code", "korean"]
    rewrites: List[str] = Field(default_factory=list)
    subqueries: List[str] = Field(default_factory=list)
    stop_condition: StopCondition = Field(default_factory=StopCondition)
    
    @classmethod
    def from_raw(cls, query: str) -> "QueryPlan":
        """단순 쿼리에서 기본 QueryPlan 생성"""
        return cls(raw_query=query, normalized_query=query)
