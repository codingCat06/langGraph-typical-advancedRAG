"""
VectorDB - ChromaDB 저장 및 관리
페이지 정보 추출 포함
레거시/신규 store.json 포맷 모두 지원
"""

import json
import time
import re
from typing import Dict, Any, List

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from settings import STORE_FILE, CHROMA_DIR


# =============================================================================
# Store Management (호환성 레이어)
# =============================================================================

def load_store() -> Dict[str, Any]:
    if STORE_FILE.exists():
        with open(STORE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_store(store: Dict[str, Any]) -> None:
    with open(STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def get_doc_entries(store: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    문서 항목만 추출 (레거시/신규 포맷 호환)
    Returns: {url: data} 딕셔너리
    """
    entries = {}
    
    # 신규 포맷: documents 키 아래에 저장
    if "documents" in store:
        for doc_id, data in store.get("documents", {}).items():
            if isinstance(data, dict):
                url = data.get("source_url", doc_id)
                entries[url] = data
    
    # 레거시 포맷: URL이 직접 키로 저장
    for key, data in store.items():
        if key in ("documents", "cache", "events", "_legacy_migrated"):
            continue
        if isinstance(data, dict) and key.startswith("http"):
            entries[key] = data
    
    return entries


def update_doc_entry(store: Dict[str, Any], url: str, updates: Dict[str, Any]) -> None:
    """
    문서 항목 업데이트 (레거시/신규 포맷 호환)
    """
    # 레거시: 루트 레벨에 있으면 그곳 업데이트
    if url in store:
        store[url].update(updates)
        return
    
    # 신규: documents 아래에서 찾기
    if "documents" in store:
        for doc_id, data in store["documents"].items():
            if data.get("source_url") == url:
                store["documents"][doc_id].update(updates)
                return
    
    # 못 찾으면 레거시 방식으로 추가
    if url not in store:
        store[url] = {}
    store[url].update(updates)


# =============================================================================
# Page Extraction
# =============================================================================

def extract_page_from_content(content: str, position: int) -> int:
    """
    마크다운 내용에서 position 이전까지의 마지막 페이지 번호 추출
    <span id="page-X-Y"> 형식에서 X가 페이지 번호 (0-indexed)
    """
    text_before = content[:position]
    page_matches = re.findall(r'<span id="page-(\d+)-\d+">', text_before)
    
    if page_matches:
        return int(page_matches[-1]) + 1  # 0-indexed -> 1-indexed
    return 1  # 기본값


def split_with_page_info(content: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Dict]:
    """
    텍스트를 청크로 분할하면서 각 청크의 페이지 정보 추출
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks_with_info = []
    current_pos = 0
    
    chunks = splitter.split_text(content)
    
    for chunk in chunks:
        chunk_start = content.find(chunk[:50], current_pos)
        if chunk_start == -1:
            chunk_start = current_pos
        
        page_num = extract_page_from_content(content, chunk_start)
        
        chunks_with_info.append({
            "text": chunk,
            "page": page_num
        })
        
        current_pos = chunk_start + len(chunk) - chunk_overlap
    
    return chunks_with_info


# =============================================================================
# Functions
# =============================================================================

def get_pending() -> List[Dict[str, Any]]:
    """vectorized=False, status=success인 항목"""
    store = load_store()
    entries = get_doc_entries(store)
    
    return [
        {"url": url, **data}
        for url, data in entries.items()
        if data.get("status") == "success" 
        and data.get("parsed_file")
        and not data.get("vectorized", False)
    ]


def update() -> int:
    """pending 문서를 ChromaDB에 저장 (페이지 정보 포함)"""
    pending = get_pending()
    
    if not pending:
        print("[INFO] 벡터화할 문서가 없습니다.")
        return 0
    
    print(f"[UPDATE] {len(pending)}개 문서 처리 중...")
    
    all_chunks = []
    all_metadatas = []
    processed_urls = []
    
    for doc in pending:
        parsed_file = doc.get("parsed_file")
        url = doc.get("url")
        
        filename = url.split("/")[-1] if "/" in url else url
        
        try:
            with open(parsed_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            chunks_info = split_with_page_info(content)
            
            for i, chunk_info in enumerate(chunks_info):
                all_chunks.append(chunk_info["text"])
                all_metadatas.append({
                    "source": url,
                    "filename": filename,
                    "page": chunk_info["page"],
                    "chunk_index": i,
                    "total_chunks": len(chunks_info),
                    "page_count": doc.get("page_count"),
                })
            
            processed_urls.append(url)
            print(f"  ✅ {filename}: {len(chunks_info)} chunks")
            
        except Exception as e:
            print(f"  ❌ {e}")
    
    if not all_chunks:
        return 0
    
    print(f"[CHROMA] {len(all_chunks)}개 청크 저장 중...")
    
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    if CHROMA_DIR.exists():
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings
        )
        vectorstore.add_texts(texts=all_chunks, metadatas=all_metadatas)
    else:
        Chroma.from_texts(
            texts=all_chunks,
            embedding=embeddings,
            metadatas=all_metadatas,
            persist_directory=str(CHROMA_DIR)
        )
    
    # Mark as vectorized
    store = load_store()
    for url in processed_urls:
        update_doc_entry(store, url, {
            "vectorized": True,
            "vectorized_at": time.strftime("%Y-%m-%d %H:%M:%S")
        })
    save_store(store)
    
    print(f"[SUCCESS] {len(all_chunks)}개 청크 저장 완료")
    return len(all_chunks)


def reset(urls: List[str] = None) -> None:
    """vectorized 상태 리셋 + ChromaDB에서 삭제"""
    store = load_store()
    entries = get_doc_entries(store)
    
    # ChromaDB에서 삭제
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings
        )
        
        if urls:
            for url in urls:
                vectorstore._collection.delete(where={"source": url})
        else:
            vectorstore.delete_collection()
        
        print("[CHROMA] VectorDB에서 삭제 완료")
    except Exception as e:
        print(f"[WARN] VectorDB 삭제 중 오류: {e}")
    
    # Store 업데이트
    target_urls = urls if urls else list(entries.keys())
    for url in target_urls:
        update_doc_entry(store, url, {
            "vectorized": False,
            "vectorized_at": None
        })
    
    save_store(store)
    print("[RESET] 완료")


def list_vectorized() -> None:
    """벡터화된 문서 목록 출력"""
    store = load_store()
    entries = get_doc_entries(store)
    
    vectorized = [
        (url, data) for url, data in entries.items()
        if data.get("vectorized", False)
    ]
    
    if not vectorized:
        print("[INFO] 벡터화된 문서가 없습니다.")
        return
    
    print(f"\n{'='*50}")
    print(f"VectorDB 문서 목록 ({len(vectorized)}개)")
    print(f"{'='*50}\n")
    
    for url, data in vectorized:
        filename = url.split("/")[-1] if "/" in url else url
        print(f"📄 {filename}")
        print(f"   URL: {url[:50]}...")
        print(f"   저장일: {data.get('vectorized_at', 'N/A')}")
        print(f"   페이지: {data.get('page_count', 'N/A')}p")
        print()


def get_retriever(k: int = 3):
    """ChromaDB retriever 반환"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
    return vectorstore.as_retriever(search_kwargs={"k": k})


def get_vectorstore():
    """ChromaDB vectorstore 반환"""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=str(CHROMA_DIR),
        embedding_function=embeddings
    )
