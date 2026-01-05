"""
Planner Node - Self-Query Planner
질문을 QueryPlan으로 구조화
"""

from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate

from graph.state import RAGState
from models.query import QueryPlan
from llm import get_llm_client
from settings import NodeLogger, MAX_ITERATIONS, MIN_SUFFICIENCY


log = NodeLogger("Planner")


PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query planning assistant. Analyze the user's question and create a structured query plan.

Output JSON with these fields:
- normalized_query: cleaned/clarified version of the query
- intent_hint: one of "explain", "howto", "troubleshoot", "compare", "extract", "unknown"
- answer_style: one of "short", "normal", "detailed"
- constraints: list of constraints like "cite_pages", "include_code", "korean"
- rewrites: list of alternative phrasings
- subqueries: list of sub-questions to answer (if complex query)

Example output:
{{
  "normalized_query": "Transformer 모델의 self-attention 메커니즘",
  "intent_hint": "explain",
  "answer_style": "detailed",
  "constraints": ["korean", "cite_pages"],
  "rewrites": ["self-attention이란", "transformer attention 작동 원리"],
  "subqueries": []
}}"""),
    ("human", "Query: {query}")
])


def plan_query(state: RAGState) -> Dict[str, Any]:
    """
    질문을 QueryPlan으로 구조화
    """
    log.enter(f"query='{state['query'][:50]}...'")
    
    query = state["query"]
    
    try:
        client = get_llm_client()
        
        # JSON 스키마로 파싱 시도
        plan = client.invoke_json(
            prompt=PLANNER_PROMPT,
            variables={"query": query},
            schema=QueryPlan,
            max_retries=1
        )
        
        # 기본값 설정
        plan.raw_query = query
        plan.stop_condition.min_sufficiency = MIN_SUFFICIENCY
        plan.stop_condition.max_iters = MAX_ITERATIONS
        
        log.success(f"intent={plan.intent_hint}, subqueries={len(plan.subqueries)}")
        
        return {
            "query_plan": plan,
            "max_iters": MAX_ITERATIONS,
            "iteration": 0,
            "flow_log": state.get("flow_log", []) + [f"[Planner] intent={plan.intent_hint}"]
        }
        
    except Exception as e:
        log.fail(str(e))
        
        # 실패 시 기본 plan 생성
        plan = QueryPlan.from_raw(query)
        
        return {
            "query_plan": plan,
            "max_iters": MAX_ITERATIONS,
            "iteration": 0,
            "error": str(e),
            "flow_log": state.get("flow_log", []) + [f"[Planner] fallback (error: {str(e)[:30]})"]
        }
