"""
Answer Models - 답변 및 평가 결과
"""

from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
import uuid

from settings import PIPELINE_VERSION


class EvidenceItem(BaseModel):
    """답변 근거"""
    doc_id: str
    doc_sha256: str = ""
    chunk_ids: List[str] = Field(default_factory=list)
    pages: List[int] = Field(default_factory=list)


class SufficiencyResult(BaseModel):
    """충분성 판단 결과"""
    score: float = 0.0
    verdict: Literal["enough", "not_enough"] = "not_enough"
    notes: str = ""
    missing_aspects: List[str] = Field(default_factory=list)


class HallucinationResult(BaseModel):
    """환각 검사 결과"""
    grounded: bool = True
    score: Optional[float] = None
    reason: Optional[str] = None


class AnswerRecord(BaseModel):
    """
    답변 + 전파 근거
    """
    answer_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    final_answer: str = ""
    evidence: List[EvidenceItem] = Field(default_factory=list)
    sufficiency: SufficiencyResult = Field(default_factory=SufficiencyResult)
    hallucination: HallucinationResult = Field(default_factory=HallucinationResult)
    pipeline_version: str = PIPELINE_VERSION
    
    # 메타
    iterations_used: int = 1
    total_chunks_retrieved: int = 0
