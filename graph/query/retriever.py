"""
Retriever Node - ANN 검색
Chroma에서 유사도 검색 수행
"""

from typing import Dict, Any, List

from graph.state import RAGState, RetrievedChunk
from store import get_chroma_store
from settings import RETRIEVAL_TOP_K, NodeLogger


log = NodeLogger("Retriever")


def retrieve_chunks(state: RAGState) -> Dict[str, Any]:
    """
    ANN 검색 수행
    - subquery가 있으면 각각 검색
    - 결과를 candidates에 저장
    """
    log.enter()
    
    query_plan = state.get("query_plan")
    iteration = state.get("iteration", 0)
    
    # 검색 쿼리 결정
    if query_plan and query_plan.subqueries and state.get("subquery_index", 0) < len(query_plan.subqueries):
        # subquery 모드
        idx = state.get("subquery_index", 0)
        search_query = query_plan.subqueries[idx]
        log.debug(f"Using subquery[{idx}]: {search_query[:30]}...")
    elif query_plan and query_plan.normalized_query:
        search_query = query_plan.normalized_query
    else:
        search_query = state["query"]
    
    # 반복 시 missing_aspects 기반 추가 검색
    missing_aspects = state.get("missing_aspects", [])
    if iteration > 0 and missing_aspects:
        # 부족한 측면을 쿼리에 추가
        additional = " ".join(missing_aspects[:2])
        search_query = f"{search_query} {additional}"
        log.debug(f"Added missing aspects: {additional}")
    
    try:
        store = get_chroma_store()
        results = store.similarity_search(
            query=search_query,
            k=RETRIEVAL_TOP_K
        )
        
        # RetrievedChunk로 변환
        candidates: List[RetrievedChunk] = []
        for r in results:
            metadata = r["metadata"]
            text = r["doc"].page_content
            
            # chunk_id 없으면 텍스트 해시로 대체
            chunk_id = metadata.get("chunk_id")
            if not chunk_id:
                import hashlib
                chunk_id = hashlib.md5(text[:100].encode()).hexdigest()[:12]
            
            # doc_id 없으면 source URL 해시로 대체
            doc_id = metadata.get("doc_id")
            if not doc_id:
                source = metadata.get("source", "")
                import hashlib
                doc_id = hashlib.md5(source.encode()).hexdigest()[:12]
            
            chunk: RetrievedChunk = {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "text": text,
                "score": r["score"],
                "weighted_score": r["weighted_score"],
                "metadata": metadata
            }
            candidates.append(chunk)
        
        # 기존 pool_all과 병합 (중복 제거 - chunk_id 기반)
        pool_all = state.get("pool_all", [])
        seen_ids = {c["chunk_id"] for c in pool_all if c.get("chunk_id")}
        for c in candidates:
            cid = c.get("chunk_id", "")
            if cid and cid not in seen_ids:
                pool_all.append(c)
                seen_ids.add(cid)
            elif not cid:
                pool_all.append(c)  # ID 없으면 그냥 추가
        
        log.success(f"Retrieved {len(candidates)} chunks, pool_all={len(pool_all)}")
        
        return {
            "candidates": candidates,
            "pool_all": pool_all,
            "flow_log": state.get("flow_log", []) + [
                f"[Retriever] iter={iteration}, found={len(candidates)}"
            ]
        }
        
    except Exception as e:
        log.fail(str(e))
        return {
            "candidates": [],
            "error": str(e),
            "flow_log": state.get("flow_log", []) + [f"[Retriever] ERROR: {str(e)[:30]}"]
        }
