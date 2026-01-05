"""
LLM Client - LLM 호출 추상화
JSON Schema 강제 및 재시도 로직 포함
"""

from typing import Any, Dict, Optional, Type, TypeVar
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

from settings import LLM_MODEL, LLM_TEMPERATURE, NodeLogger


log = NodeLogger("LLMClient")

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """LLM 호출 클라이언트"""
    
    def __init__(
        self,
        model: str = LLM_MODEL,
        temperature: float = LLM_TEMPERATURE,
    ):
        self.model = model
        self.temperature = temperature
        self._llm: Optional[ChatOpenAI] = None
    
    @property
    def llm(self) -> ChatOpenAI:
        """Lazy initialization"""
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
            )
        return self._llm
    
    def invoke(
        self,
        prompt: ChatPromptTemplate,
        variables: Dict[str, Any],
    ) -> str:
        """단순 텍스트 응답"""
        log.enter(f"model={self.model}")
        chain = prompt | self.llm
        result = chain.invoke(variables)
        log.exit(f"response_len={len(result.content)}")
        return result.content
    
    def invoke_json(
        self,
        prompt: ChatPromptTemplate,
        variables: Dict[str, Any],
        schema: Type[T],
        max_retries: int = 1,
    ) -> T:
        """
        JSON Schema 강제 응답
        실패 시 max_retries 만큼 재시도
        """
        log.enter(f"schema={schema.__name__}, retries={max_retries}")
        
        parser = JsonOutputParser(pydantic_object=schema)
        chain = prompt | self.llm | parser
        
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                result = chain.invoke(variables)
                
                # dict -> Pydantic
                if isinstance(result, dict):
                    result = schema(**result)
                
                log.success(f"Parsed {schema.__name__}")
                return result
                
            except Exception as e:
                last_error = e
                log.loop(attempt + 1, f"Parse failed: {str(e)[:50]}")
        
        log.fail(f"All retries failed: {last_error}")
        raise ValueError(f"Failed to parse {schema.__name__}: {last_error}")
    
    def stream(
        self,
        prompt: ChatPromptTemplate,
        variables: Dict[str, Any],
    ):
        """스트리밍 응답"""
        log.enter(f"model={self.model}, streaming=True")
        
        streaming_llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            streaming=True,
        )
        chain = prompt | streaming_llm
        
        for chunk in chain.stream(variables):
            yield chunk.content
        
        log.exit("Stream complete")


# 싱글톤
_default_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """기본 LLMClient 인스턴스"""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client
