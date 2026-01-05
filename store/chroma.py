"""
ChromaStore - Chroma 벡터 DB 래퍼
"""

import math
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from settings import (
    CHROMA_DIR,
    EMBEDDING_MODEL,
    USE_TIME_WEIGHT,
    TIME_WEIGHT_ALPHA,
    TIME_WEIGHT_TAU_DAYS,
    NodeLogger,
)


log = NodeLogger("ChromaStore")


class ChromaStore:
    """Chroma 벡터 DB 래퍼"""
    
    def __init__(self, collection_name: str = "langchain"):
        # 기존 vectordb.py와 호환: langchain 기본 컬렉션 사용
        self.collection_name = collection_name
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        self._vectorstore: Optional[Chroma] = None
    
    @property
    def vectorstore(self) -> Chroma:
        """Lazy initialization"""
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                persist_directory=str(CHROMA_DIR),
                embedding_function=self.embeddings
            )
        return self._vectorstore
    
    def add_texts(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> List[str]:
        """텍스트 추가 및 ID 반환"""
        log.enter(f"Adding {len(texts)} texts")
        ids = self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        log.exit(f"Added {len(ids)} documents")
        return ids
    
    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        유사도 검색 + Time Weighting 적용
        
        Returns:
            List of {doc, score, weighted_score}
        """
        log.enter(f"query='{query[:30]}...', k={k}")
        
        # score 포함 검색
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
            filter=filter
        )
        
        scored_results = []
        for doc, score in results:
            weighted_score = self._apply_time_weight(score, doc.metadata)
            scored_results.append({
                "doc": doc,
                "score": score,
                "weighted_score": weighted_score,
                "metadata": doc.metadata
            })
        
        # weighted_score로 재정렬
        scored_results.sort(key=lambda x: x["weighted_score"], reverse=True)
        
        log.exit(f"Found {len(scored_results)} results")
        return scored_results
    
    def _apply_time_weight(self, score: float, metadata: Dict[str, Any]) -> float:
        """
        Time Weighting 적용
        w(age) = 1 + alpha * exp(-age_days / tau_days)
        
        Note: 기존 데이터에 ingested_at이 없으면 weight=1 (영향 없음)
        """
        if not USE_TIME_WEIGHT:
            return score
        
        # ingested_at 또는 vectorized_at 사용
        ingested_at_str = metadata.get("ingested_at") or metadata.get("vectorized_at")
        if not ingested_at_str:
            return score  # 기존 데이터는 time weight 미적용
        
        try:
            if isinstance(ingested_at_str, str):
                # 다양한 포맷 처리
                for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                    try:
                        ingested_at = datetime.strptime(ingested_at_str, fmt)
                        break
                    except:
                        continue
                else:
                    return score
            else:
                ingested_at = ingested_at_str
            
            age_days = (datetime.now() - ingested_at).days
            weight = 1 + TIME_WEIGHT_ALPHA * math.exp(-age_days / TIME_WEIGHT_TAU_DAYS)
            return score * weight
        except Exception:
            return score
    
    def delete_by_doc_id(self, doc_id: str) -> None:
        """doc_id로 문서 삭제 (레거시: source URL로도 삭제)"""
        log.enter(f"Deleting doc_id={doc_id}")
        try:
            # 먼저 doc_id로 시도
            self.vectorstore._collection.delete(where={"doc_id": doc_id})
        except Exception:
            pass
        
        try:
            # 레거시: source URL로도 시도 (doc_id가 URL 해시일 수 있음)
            self.vectorstore._collection.delete(where={"source": doc_id})
            log.success("Deleted")
        except Exception as e:
            log.fail(str(e))
    
    def delete_by_source(self, source_url: str) -> None:
        """source URL로 문서 삭제 (레거시 호환)"""
        log.enter(f"Deleting source={source_url[:30]}...")
        try:
            self.vectorstore._collection.delete(where={"source": source_url})
            log.success("Deleted")
        except Exception as e:
            log.fail(str(e))
    
    def delete_collection(self) -> None:
        """전체 컬렉션 삭제"""
        log.enter("Deleting entire collection")
        try:
            self.vectorstore.delete_collection()
            self._vectorstore = None
            log.success("Collection deleted")
        except Exception as e:
            log.fail(str(e))
    
    def get_retriever(self, k: int = 5):
        """LangChain Retriever 반환"""
        return self.vectorstore.as_retriever(search_kwargs={"k": k})


# 싱글톤 인스턴스
_default_store: Optional[ChromaStore] = None


def get_chroma_store() -> ChromaStore:
    """기본 ChromaStore 인스턴스 반환"""
    global _default_store
    if _default_store is None:
        _default_store = ChromaStore()
    return _default_store
