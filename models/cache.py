"""
Cache Models - 캐시 및 문서 이벤트
"""

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field
import uuid

from settings import PIPELINE_VERSION


class CacheRecord(BaseModel):
    """
    Semantic Cache 레코드
    """
    cache_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str
    query_embedding_id: Optional[str] = None
    cached_answer_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    pipeline_version: str = PIPELINE_VERSION
    stale_flag: bool = False
    stale_reason: Optional[Literal["doc_updated", "doc_deleted", "new_doc_added"]] = None
    related_doc_id: Optional[str] = None


class DocumentEvent(BaseModel):
    """
    문서 변경 이벤트 (전파 트리거)
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    event_type: Literal["added", "updated", "deleted"]
    doc_id: str
    old_doc_sha256: Optional[str] = None
    new_doc_sha256: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
