"""
Document Models - 문서 및 청크 메타데이터
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class ParserInfo(BaseModel):
    """파서 정보"""
    name: str = "marker-pdf"
    version: str = "unknown"


class ChunkerInfo(BaseModel):
    """청커 정보"""
    name: str = "RecursiveCharacterTextSplitter"
    version: str = "langchain"
    params_hash: str = ""


class DocumentRecord(BaseModel):
    """
    PDF 파일 단위 메타데이터 (Hard)
    전파/버전의 기준
    """
    doc_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_sha256: str = ""
    source_url: str
    ingested_at: datetime = Field(default_factory=datetime.now)
    parser: ParserInfo = Field(default_factory=ParserInfo)
    chunker: ChunkerInfo = Field(default_factory=ChunkerInfo)
    num_pages: int = 0
    file_bytes: Optional[int] = None
    
    # 추가 필드 (store.json 호환)
    parsed_file: Optional[str] = None
    vectorized: bool = False
    vectorized_at: Optional[datetime] = None


class ChunkRecord(BaseModel):
    """
    검색/근거 단위 메타데이터 (Hard)
    """
    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    page_start: int = 1
    page_end: int = 1
    char_start: int = 0
    char_end: int = 0
    token_count: int = 0
    text_sha1: Optional[str] = None
    
    # Chroma 저장용 추가 필드
    text: str = ""
    embedding_id: Optional[str] = None


class ChunkDerived(BaseModel):
    """
    청크 파생 정보 (Soft/Derived)
    틀려도 시스템이 죽지 않는 힌트
    """
    chunk_id: str
    language: str = "mixed"  # "ko" | "en" | "mixed"
    has_code: bool = False
    has_table_like: bool = False
    chunk_embedding_id: Optional[str] = None
    doc_centroid_embedding_id: Optional[str] = None
    topic_cluster_id: Optional[int] = None
    topic_conf: Optional[float] = None
