"""
Generator Node - 답변 생성
"""

from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate

from graph.state import RAGState
from llm import get_llm_client
from settings import NodeLogger


log = NodeLogger("Generator")


GENERATOR_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a knowledgeable assistant. Answer the question based on the provided documents.

IMPORTANT RULES:
1. When you use information from a document, cite it INLINE like: "...내용... [파일명 p.페이지]"
2. Answer in Korean
3. If unsure based on the context, say so
4. Be comprehensive but concise

Documents:
{context}"""),
    ("human", "{query}")
])


def generate_answer(state: RAGState) -> Dict[str, Any]:
    """
    최종 답변 생성
    """
    log.enter()
    
    pool_high = state.get("pool_high", [])
    
    if not pool_high:
        log.skip("No context for generation")
        return {
            "answer": "관련 문서를 찾을 수 없습니다.",
            "evidence": [],
            "flow_log": state.get("flow_log", []) + ["[Generator] no context"]
        }
    
    # 컨텍스트 및 evidence 구성
    context_parts = []
    evidence: List[Dict[str, Any]] = []
    
    for c in pool_high:
        metadata = c.get("metadata", {})
        filename = metadata.get("filename", "Unknown")
        page = metadata.get("page", "?")
        
        source_label = f"[{filename} p.{page}]"
        context_parts.append(f"--- {source_label} ---\n{c['text']}")
        
        evidence.append({
            "doc_id": c.get("doc_id", ""),
            "chunk_id": c.get("chunk_id", ""),
            "filename": filename,
            "page": page,
        })
    
    context = "\n\n".join(context_parts)
    
    query_text = state.get("query_plan")
    if query_text:
        query_text = query_text.normalized_query or state["query"]
    else:
        query_text = state["query"]
    
    try:
        client = get_llm_client()
        answer = client.invoke(
            prompt=GENERATOR_PROMPT,
            variables={"context": context, "query": query_text}
        )
        
        log.success(f"Generated {len(answer)} chars")
        
        return {
            "answer": answer,
            "evidence": evidence,
            "flow_log": state.get("flow_log", []) + [f"[Generator] answer_len={len(answer)}"]
        }
        
    except Exception as e:
        log.fail(str(e))
        return {
            "answer": f"답변 생성 중 오류가 발생했습니다: {str(e)}",
            "evidence": evidence,
            "error": str(e),
            "flow_log": state.get("flow_log", []) + [f"[Generator] ERROR: {str(e)[:30]}"]
        }
