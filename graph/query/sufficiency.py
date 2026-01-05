"""
Sufficiency Node - 충분성 판단
"""

from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from graph.state import RAGState
from models.answer import SufficiencyResult
from llm import get_llm_client
from settings import MIN_SUFFICIENCY, NodeLogger


log = NodeLogger("Sufficiency")


class SufficiencyOutput(BaseModel):
    """LLM 출력 스키마"""
    score: float
    verdict: str
    notes: str
    missing_aspects: list[str]


SUFFICIENCY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a sufficiency judge. Given a query and retrieved context, 
determine if the context is sufficient to answer the query.

Output JSON:
{{
  "score": 0.0-1.0 (how sufficient the context is),
  "verdict": "enough" or "not_enough",
  "notes": "brief explanation",
  "missing_aspects": ["list", "of", "missing", "info"]  // empty if enough
}}

Be strict: if key information is missing, give low score."""),
    ("human", """Query: {query}

Context:
{context}

Judge sufficiency:""")
])


def judge_sufficiency(state: RAGState) -> Dict[str, Any]:
    """
    충분성 판단
    - score < min_sufficiency면 반복 필요
    """
    log.enter()
    
    pool_high = state.get("pool_high", [])
    iteration = state.get("iteration", 0)
    max_iters = state.get("max_iters", 3)
    
    if not pool_high:
        log.skip("No context to judge")
        result = SufficiencyResult(
            score=0.0,
            verdict="not_enough",
            notes="No context retrieved",
            missing_aspects=["all"]
        )
        next_iter = iteration + 1
        should_continue = next_iter < max_iters
        return {
            "sufficiency": result,
            "should_continue": should_continue,
            "iteration": next_iter,
            "flow_log": state.get("flow_log", []) + [f"[Sufficiency] no context, iter={next_iter}/{max_iters}"]
        }
    
    # 컨텍스트 구성
    query_text = state.get("query_plan")
    if query_text:
        query_text = query_text.normalized_query or state["query"]
    else:
        query_text = state["query"]
    
    context = "\n\n---\n\n".join([c["text"][:500] for c in pool_high[:5]])
    
    try:
        client = get_llm_client()
        output = client.invoke_json(
            prompt=SUFFICIENCY_PROMPT,
            variables={"query": query_text, "context": context},
            schema=SufficiencyOutput,
            max_retries=1
        )
        
        result = SufficiencyResult(
            score=output.score,
            verdict="enough" if output.score >= MIN_SUFFICIENCY else "not_enough",
            notes=output.notes,
            missing_aspects=output.missing_aspects
        )
        
        next_iter = iteration + 1
        should_continue = result.verdict == "not_enough" and next_iter < max_iters
        
        if should_continue:
            log.loop(next_iter, f"score={result.score:.2f} < {MIN_SUFFICIENCY}, iter={next_iter}/{max_iters}")
        else:
            log.success(f"score={result.score:.2f}, verdict={result.verdict}")
        
        return {
            "sufficiency": result,
            "missing_aspects": result.missing_aspects,
            "should_continue": should_continue,
            "iteration": next_iter,
            "flow_log": state.get("flow_log", []) + [
                f"[Sufficiency] score={result.score:.2f}, iter={next_iter}/{max_iters}"
            ]
        }
        
    except Exception as e:
        log.fail(str(e))
        # 실패 시 충분하다고 가정하고 진행
        result = SufficiencyResult(score=0.5, verdict="enough", notes=f"Error: {str(e)}")
        return {
            "sufficiency": result,
            "should_continue": False,
            "error": str(e),
            "flow_log": state.get("flow_log", []) + [f"[Sufficiency] ERROR, proceeding anyway"]
        }
