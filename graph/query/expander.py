"""
Expander Node - 이웃 Chunk 확장
페이지 ±1 또는 이전/다음 chunk 포함
"""

from typing import Dict, Any, List

from graph.state import RAGState, RetrievedChunk
from store import get_chroma_store
from settings import NodeLogger


log = NodeLogger("Expander")


def expand_neighbors(state: RAGState) -> Dict[str, Any]:
    """
    검색된 chunk의 이웃 확장
    - 같은 doc_id 내에서 page ±1 chunk 포함
    """
    log.enter()
    
    candidates = state.get("candidates", [])
    pool_all = state.get("pool_all", [])
    
    if not candidates:
        log.skip("No candidates to expand")
        return {"flow_log": state.get("flow_log", []) + ["[Expander] skipped (no candidates)"]}
    
    # 확장 대상 수집: (doc_id, page) 쌍
    expansion_targets = set()
    for c in candidates:
        doc_id = c.get("doc_id")
        page = c.get("metadata", {}).get("page", 1)
        if doc_id:
            # 현재 페이지 ±1
            expansion_targets.add((doc_id, page - 1))
            expansion_targets.add((doc_id, page + 1))
    
    # 이미 있는 chunk 제외
    existing_keys = {
        (c.get("doc_id"), c.get("metadata", {}).get("page"))
        for c in pool_all
    }
    new_targets = expansion_targets - existing_keys
    
    if not new_targets:
        log.skip("No new neighbors to fetch")
        return {"flow_log": state.get("flow_log", []) + ["[Expander] no new neighbors"]}
    
    # TODO: Chroma에서 metadata filter로 이웃 검색
    # 현재는 간단히 스킵 (실제 구현 시 filter 사용)
    log.debug(f"Would expand {len(new_targets)} neighbor pages (not implemented)")
    
    log.exit(f"expansion_targets={len(new_targets)}")
    return {
        "flow_log": state.get("flow_log", []) + [f"[Expander] targets={len(new_targets)} (placeholder)"]
    }
