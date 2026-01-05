"""
Hallucination Grader Node - LangSmith Hub 기반 환각 검사
"""

from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from graph.state import RAGState
from models.answer import HallucinationResult
from llm import get_llm_client
from settings import (
    HALLUCINATION_CHECK_ENABLED,
    HALLUCINATION_PROMPT_ID,
    HALLUCINATION_ON_FAIL,
    NodeLogger
)


log = NodeLogger("HalluGrader")


class HallucinationOutput(BaseModel):
    """LLM 출력 스키마"""
    grounded: bool
    score: float = 1.0
    reason: str = ""


# 기본 프롬프트 (LangSmith Hub 대신 사용)
HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a hallucination detector. Given a question, context, and answer,
determine if the answer is grounded in the context.

Output JSON:
{{
  "grounded": true/false (is the answer fully supported by context?),
  "score": 0.0-1.0 (how grounded is the answer?),
  "reason": "explanation"
}}

Be strict: any claim not directly supported by context = not grounded."""),
    ("human", """Question: {question}

Context:
{context}

Answer: {answer}

Judge if the answer is grounded:""")
])


def grade_hallucination(state: RAGState) -> Dict[str, Any]:
    """
    환각 검사
    - grounded=False면 rewrite 또는 loop back
    """
    log.enter()
    
    if not HALLUCINATION_CHECK_ENABLED:
        log.skip("Hallucination check disabled")
        return {
            "hallucination": HallucinationResult(grounded=True),
            "flow_log": state.get("flow_log", []) + ["[HalluGrader] disabled"]
        }
    
    answer = state.get("answer")
    pool_high = state.get("pool_high", [])
    
    if not answer or not pool_high:
        log.skip("No answer or context to grade")
        return {
            "hallucination": HallucinationResult(grounded=True),
            "flow_log": state.get("flow_log", []) + ["[HalluGrader] skipped"]
        }
    
    # 컨텍스트 구성
    context = "\n\n---\n\n".join([c["text"][:400] for c in pool_high[:3]])
    
    query_text = state.get("query_plan")
    if query_text:
        query_text = query_text.normalized_query or state["query"]
    else:
        query_text = state["query"]
    
    try:
        # TODO: LangSmith Hub에서 프롬프트 pull
        # from langchain import hub
        # prompt = hub.pull(HALLUCINATION_PROMPT_ID)
        
        client = get_llm_client()
        output = client.invoke_json(
            prompt=HALLUCINATION_PROMPT,
            variables={
                "question": query_text,
                "context": context,
                "answer": answer[:500]
            },
            schema=HallucinationOutput,
            max_retries=1
        )
        
        result = HallucinationResult(
            grounded=output.grounded,
            score=output.score,
            reason=output.reason
        )
        
        if result.grounded:
            log.success(f"grounded=True, score={result.score:.2f}")
        else:
            log.loop(1, f"grounded=False, reason={result.reason[:30]}")
        
        # 환각 감지 시 처리
        should_rewrite = not result.grounded and HALLUCINATION_ON_FAIL == "rewrite_once"
        should_loop_back = not result.grounded and HALLUCINATION_ON_FAIL == "loop_back"
        
        return {
            "hallucination": result,
            "should_continue": should_loop_back and state.get("iteration", 0) < state.get("max_iters", 3),
            "flow_log": state.get("flow_log", []) + [
                f"[HalluGrader] grounded={result.grounded}, score={result.score:.2f}"
            ]
        }
        
    except Exception as e:
        log.fail(str(e))
        return {
            "hallucination": HallucinationResult(grounded=True, reason=f"Error: {str(e)}"),
            "error": str(e),
            "flow_log": state.get("flow_log", []) + [f"[HalluGrader] ERROR"]
        }
