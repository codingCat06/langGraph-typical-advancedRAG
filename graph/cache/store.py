"""
Cache Store Node - 답변 캐싱
"""

from typing import Dict, Any

from graph.state import RAGState
from models.cache import CacheRecord
from store.metadata import get_metadata_store
from settings import CACHE_ENABLED, NodeLogger


log = NodeLogger("CacheStore")


def cache_store(state: RAGState) -> Dict[str, Any]:
    """
    답변을 캐시에 저장
    """
    log.enter()
    
    if not CACHE_ENABLED:
        log.skip("Cache disabled")
        return {"flow_log": state.get("flow_log", []) + ["[CacheStore] disabled"]}
    
    answer = state.get("answer")
    query_plan = state.get("query_plan")
    
    if not answer or not query_plan:
        log.skip("No answer or query_plan to cache")
        return {"flow_log": state.get("flow_log", []) + ["[CacheStore] skipped"]}
    
    try:
        # TODO: query embedding을 cache collection에 저장
        # 현재는 metadata store에만 저장
        
        cache = CacheRecord(
            query_id=query_plan.query_id,
            cached_answer_id="",  # TODO: answer_id 연결
        )
        
        meta_store = get_metadata_store()
        meta_store.add_cache(cache)
        
        log.success(f"Cached query_id={query_plan.query_id[:8]}...")
        return {
            "flow_log": state.get("flow_log", []) + [f"[CacheStore] saved"]
        }
        
    except Exception as e:
        log.fail(str(e))
        return {
            "error": str(e),
            "flow_log": state.get("flow_log", []) + [f"[CacheStore] ERROR"]
        }
