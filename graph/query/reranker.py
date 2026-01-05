"""
Reranker Node - LLM 기반 재순위
"""

from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate

from graph.state import RAGState, RetrievedChunk
from llm import get_llm_client
from settings import RERANK_TOP_K, NodeLogger


log = NodeLogger("Reranker")


RERANK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a relevance judge. Given a query and a list of document chunks, 
rank them by relevance to the query.

Output JSON array of chunk indices (0-based) in order of relevance, most relevant first.
Only include the top {top_k} most relevant chunks.

Example output: [2, 0, 4]"""),
    ("human", """Query: {query}

Chunks:
{chunks}

Output the indices of the top {top_k} most relevant chunks as JSON array:""")
])


def rerank_chunks(state: RAGState) -> Dict[str, Any]:
    """
    LLM 기반 재순위
    - pool_all에서 top_k 선택
    - pool_high 업데이트
    """
    log.enter()
    
    pool_all = state.get("pool_all", [])
    
    if not pool_all:
        log.skip("No chunks to rerank")
        return {"pool_high": [], "flow_log": state.get("flow_log", []) + ["[Reranker] skipped (no chunks)"]}
    
    if len(pool_all) <= RERANK_TOP_K:
        # 이미 충분히 적으면 그대로 사용
        log.skip(f"Only {len(pool_all)} chunks, using all")
        return {
            "pool_high": pool_all,
            "flow_log": state.get("flow_log", []) + [f"[Reranker] using all {len(pool_all)} chunks"]
        }
    
    # 청크 텍스트 준비
    query = state.get("query_plan")
    if query:
        query_text = query.normalized_query or state["query"]
    else:
        query_text = state["query"]
    
    chunks_text = "\n\n".join([
        f"[{i}] {c['text'][:300]}..." for i, c in enumerate(pool_all[:10])  # 최대 10개만
    ])
    
    try:
        client = get_llm_client()
        response = client.invoke(
            prompt=RERANK_PROMPT,
            variables={
                "query": query_text,
                "chunks": chunks_text,
                "top_k": RERANK_TOP_K
            }
        )
        
        # JSON 파싱 시도
        import json
        try:
            # [0, 2, 1] 형태 추출
            import re
            match = re.search(r'\[[\d,\s]+\]', response)
            if match:
                indices = json.loads(match.group())
            else:
                indices = list(range(RERANK_TOP_K))
        except:
            indices = list(range(RERANK_TOP_K))
        
        # 인덱스로 pool_high 구성
        pool_high: List[RetrievedChunk] = []
        for idx in indices[:RERANK_TOP_K]:
            if 0 <= idx < len(pool_all):
                pool_high.append(pool_all[idx])
        
        log.success(f"Reranked to {len(pool_high)} chunks")
        
        return {
            "pool_high": pool_high,
            "flow_log": state.get("flow_log", []) + [f"[Reranker] selected {len(pool_high)} chunks"]
        }
        
    except Exception as e:
        log.fail(str(e))
        # 실패 시 score 순 상위 선택
        sorted_pool = sorted(pool_all, key=lambda x: x.get("weighted_score", 0), reverse=True)
        pool_high = sorted_pool[:RERANK_TOP_K]
        
        return {
            "pool_high": pool_high,
            "error": str(e),
            "flow_log": state.get("flow_log", []) + [f"[Reranker] fallback (error)"]
        }
