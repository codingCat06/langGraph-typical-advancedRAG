"""
Cache Lookup Node - Semantic Cache 조회
"""

from typing import Dict, Any

from graph.state import RAGState
from store import get_chroma_store
from settings import CACHE_ENABLED, CACHE_THRESHOLD, NodeLogger


log = NodeLogger("CacheLookup")


def cache_lookup(state: RAGState) -> Dict[str, Any]:
    """
    Semantic Cache 조회
    - 유사한 과거 질문의 답변 재사용
    """
    log.enter()
    
    if not CACHE_ENABLED:
        log.skip("Cache disabled")
        return {
            "cache_hit": False,
            "flow_log": state.get("flow_log", []) + ["[CacheLookup] disabled"]
        }
    
    query = state["query"]
    
    try:
        # TODO: 별도 cache collection에서 검색
        # 현재는 placeholder
        
        # 임시: 항상 cache miss
        log.exit("Cache miss (placeholder)")
        return {
            "cache_hit": False,
            "cached_answer": None,
            "flow_log": state.get("flow_log", []) + ["[CacheLookup] miss"]
        }
        
        # 실제 구현 예시:
        # cache_store = get_chroma_store(collection="query_cache")
        # results = cache_store.similarity_search(query, k=1)
        # if results and results[0]["score"] >= CACHE_THRESHOLD:
        #     cached = results[0]["metadata"]
        #     if not cached.get("stale_flag"):
        #         return {"cache_hit": True, "cached_answer": cached["answer"]}
        
    except Exception as e:
        log.fail(str(e))
        return {
            "cache_hit": False,
            "error": str(e),
            "flow_log": state.get("flow_log", []) + [f"[CacheLookup] ERROR"]
        }
