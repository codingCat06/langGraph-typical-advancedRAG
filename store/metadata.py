"""
MetadataStore - JSON 기반 메타데이터 관리
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from settings import STORE_FILE, NodeLogger
from models.document import DocumentRecord
from models.cache import CacheRecord, DocumentEvent


log = NodeLogger("MetadataStore")


class MetadataStore:
    """JSON 기반 메타데이터 저장소"""
    
    def __init__(self, store_file: Path = STORE_FILE):
        self.store_file = store_file
        self._data: Optional[Dict[str, Any]] = None
    
    @property
    def data(self) -> Dict[str, Any]:
        """Lazy load"""
        if self._data is None:
            self._data = self._load()
        return self._data
    
    def _load(self) -> Dict[str, Any]:
        """파일에서 로드 (레거시 포맷 자동 감지)"""
        if self.store_file.exists():
            with open(self.store_file, "r", encoding="utf-8") as f:
                raw = json.load(f)
            
            # 레거시 포맷 감지 (URL이 키인 경우)
            if raw and not any(k in raw for k in ["documents", "cache", "events"]):
                log.enter("Detected legacy format, migrating...")
                return self._migrate_legacy(raw)
            
            return raw
        return {
            "documents": {},
            "cache": {},
            "events": [],
        }
    
    def _migrate_legacy(self, legacy: Dict[str, Any]) -> Dict[str, Any]:
        """
        레거시 포맷 마이그레이션 (URL -> doc_id)
        기존 ChromaDB 데이터는 그대로 유지됨
        """
        import hashlib
        
        documents = {}
        for url, data in legacy.items():
            # URL 해시를 doc_id로 사용
            doc_id = hashlib.md5(url.encode()).hexdigest()[:12]
            
            documents[doc_id] = {
                "doc_id": doc_id,
                "source_url": url,
                "doc_sha256": "",
                "ingested_at": data.get("uploaded_at", ""),
                "parser": {"name": "marker-pdf", "version": "unknown"},
                "chunker": {"name": "RecursiveCharacterTextSplitter", "version": "langchain", "params_hash": ""},
                "num_pages": data.get("page_count", 0),
                "parsed_file": data.get("parsed_file"),
                "vectorized": data.get("vectorized", False),
                "vectorized_at": data.get("vectorized_at"),
                # 레거시 필드 유지
                "request_id": data.get("request_id"),
                "status": data.get("status"),
                "fetched_at": data.get("fetched_at"),
            }
        
        log.success(f"Migrated {len(documents)} documents")
        
        return {
            "documents": documents,
            "cache": {},
            "events": [],
            "_legacy_migrated": True,
        }
    
    def save(self) -> None:
        """파일에 저장"""
        with open(self.store_file, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2, default=str)
    
    # --- Documents ---
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """doc_id로 문서 조회"""
        return self.data.get("documents", {}).get(doc_id)
    
    def get_document_by_url(self, url: str) -> Optional[Dict[str, Any]]:
        """URL로 문서 조회 (레거시 호환)"""
        docs = self.data.get("documents", {})
        for doc_id, doc in docs.items():
            if doc.get("source_url") == url:
                return {**doc, "doc_id": doc_id}
        return None
    
    def add_document(self, doc: DocumentRecord) -> str:
        """문서 추가"""
        log.enter(f"Adding doc_id={doc.doc_id}")
        if "documents" not in self.data:
            self.data["documents"] = {}
        self.data["documents"][doc.doc_id] = doc.model_dump()
        self.save()
        log.success(f"doc_id={doc.doc_id}")
        return doc.doc_id
    
    def update_document(self, doc_id: str, updates: Dict[str, Any]) -> None:
        """문서 업데이트"""
        log.enter(f"Updating doc_id={doc_id}")
        if doc_id in self.data.get("documents", {}):
            self.data["documents"][doc_id].update(updates)
            self.save()
            log.success(f"Updated {list(updates.keys())}")
        else:
            log.fail(f"doc_id={doc_id} not found")
    
    def list_documents(self, vectorized: Optional[bool] = None) -> List[Dict[str, Any]]:
        """문서 목록"""
        docs = self.data.get("documents", {})
        result = []
        for doc_id, doc in docs.items():
            if vectorized is None or doc.get("vectorized") == vectorized:
                result.append({**doc, "doc_id": doc_id})
        return result
    
    # --- Cache ---
    def get_cache(self, cache_id: str) -> Optional[Dict[str, Any]]:
        """캐시 조회"""
        return self.data.get("cache", {}).get(cache_id)
    
    def add_cache(self, cache: CacheRecord) -> str:
        """캐시 추가"""
        log.enter(f"Adding cache_id={cache.cache_id}")
        if "cache" not in self.data:
            self.data["cache"] = {}
        self.data["cache"][cache.cache_id] = cache.model_dump()
        self.save()
        return cache.cache_id
    
    def mark_cache_stale(
        self,
        cache_id: str,
        reason: str,
        related_doc_id: Optional[str] = None
    ) -> None:
        """캐시 stale 마킹"""
        log.enter(f"Marking stale: cache_id={cache_id}, reason={reason}")
        if cache_id in self.data.get("cache", {}):
            self.data["cache"][cache_id]["stale_flag"] = True
            self.data["cache"][cache_id]["stale_reason"] = reason
            self.data["cache"][cache_id]["related_doc_id"] = related_doc_id
            self.save()
            log.success("Marked stale")
    
    def list_cache(self, stale: Optional[bool] = None) -> List[Dict[str, Any]]:
        """캐시 목록"""
        caches = self.data.get("cache", {})
        result = []
        for cache_id, cache in caches.items():
            if stale is None or cache.get("stale_flag") == stale:
                result.append({**cache, "cache_id": cache_id})
        return result
    
    # --- Events ---
    def add_event(self, event: DocumentEvent) -> str:
        """이벤트 추가"""
        log.enter(f"Adding event: {event.event_type} for doc_id={event.doc_id}")
        if "events" not in self.data:
            self.data["events"] = []
        self.data["events"].append(event.model_dump())
        self.save()
        return event.event_id
    
    def list_events(self, doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """이벤트 목록"""
        events = self.data.get("events", [])
        if doc_id:
            return [e for e in events if e.get("doc_id") == doc_id]
        return events


# 싱글톤
_default_store: Optional[MetadataStore] = None


def get_metadata_store() -> MetadataStore:
    """기본 MetadataStore 인스턴스"""
    global _default_store
    if _default_store is None:
        _default_store = MetadataStore()
    return _default_store
