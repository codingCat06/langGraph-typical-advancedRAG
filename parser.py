"""
Parser - PDF 파싱 요청 및 상태 관리
레거시/신규 store.json 포맷 모두 지원
"""

import json
import time
import hashlib
from typing import Dict, Any, Optional, List

import requests

from settings import (
    API_MARKER_ENDPOINT,
    STORE_FILE,
    PARSED_DIR,
    get_check_url
)


# =============================================================================
# Store Management (호환성 레이어)
# =============================================================================

def load_store() -> Dict[str, Any]:
    """로컬 스토어 로드"""
    if STORE_FILE.exists():
        with open(STORE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_store(store: Dict[str, Any]) -> None:
    """로컬 스토어 저장"""
    with open(STORE_FILE, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False, indent=2)


def get_url_hash(url: str) -> str:
    """URL의 짧은 해시 생성"""
    return hashlib.md5(url.encode()).hexdigest()[:12]


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
# Upload
# =============================================================================

def upload(file_url: str, output_format: str = "markdown") -> Optional[str]:
    """
    PDF URL을 API에 전송하여 파싱 요청
    
    Returns:
        request_id if successful, None otherwise
    """
    store = load_store()
    entries = get_doc_entries(store)
    
    # 중복 체크
    if file_url in entries:
        print(f"[CACHE] 이미 요청한 PDF입니다")
        print(f"        Status: {entries[file_url].get('status')}")
        return entries[file_url].get("request_id")
    
    print(f"[UPLOAD] PDF 파싱 요청 중...")
    
    data = {
        "file_url": file_url,
        "output_format": output_format
    }
    
    try:
        response = requests.post(API_MARKER_ENDPOINT, data=data, timeout=30)
        result = response.json()
        
        if result.get("success"):
            request_id = result.get("request_id")
            
            # 레거시 방식으로 저장 (호환성)
            store[file_url] = {
                "request_id": request_id,
                "status": "pending",
                "parsed_file": None,
                "vectorized": False,
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            save_store(store)
            
            print(f"[SUCCESS] 요청 접수됨")
            print(f"          Request ID: {request_id}")
            return request_id
        else:
            print(f"[ERROR] API 호출 실패: {result.get('error')}")
            return None
            
    except requests.exceptions.Timeout:
        print("[TIMEOUT] 서버가 깨어나는 중일 수 있습니다 (Cold Start)")
        print("          잠시 후 다시 시도해 주세요.")
        return None
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


# =============================================================================
# Status
# =============================================================================

def status() -> None:
    """
    상태 확인 (vectorized된 항목 제외)
    pending → success 자동 fetch
    """
    store = load_store()
    entries = get_doc_entries(store)
    
    if not entries:
        print("[INFO] 등록된 PDF가 없습니다.")
        return
    
    # vectorized가 아닌 항목만
    not_vectorized = {url: data for url, data in entries.items()
                      if not data.get("vectorized", False)}
    
    if not not_vectorized:
        print("[INFO] 모든 PDF가 vectorized 되었습니다.")
        return
    
    print(f"\n{'='*50}")
    print(f"PDF 상태 ({len(not_vectorized)}개)")
    print(f"{'='*50}\n")
    
    for url, data in not_vectorized.items():
        current_status = data.get("status", "unknown")
        request_id = data.get("request_id")
        
        # pending이면 API에서 확인
        if current_status == "pending" and request_id:
            print(f"🔄 Checking: {url[:40]}...")
            
            try:
                check_url = get_check_url(request_id)
                response = requests.get(check_url, timeout=30)
                result = response.json()
                api_status = result.get("status")
                
                if api_status == "complete" and result.get("success"):
                    # 파싱 결과 저장
                    markdown = result.get("markdown", "")
                    url_hash = get_url_hash(url)
                    parsed_file = PARSED_DIR / f"{url_hash}.md"
                    
                    with open(parsed_file, "w", encoding="utf-8") as f:
                        f.write(markdown)
                    
                    update_doc_entry(store, url, {
                        "status": "success",
                        "parsed_file": str(parsed_file),
                        "page_count": result.get("page_count"),
                        "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    current_status = "success"
                    print(f"   ✅ Success!")
                    
                elif api_status == "complete" and not result.get("success"):
                    update_doc_entry(store, url, {
                        "status": "failed",
                        "error": result.get("error")
                    })
                    current_status = "failed"
                    print(f"   ❌ Failed")
                else:
                    print(f"   ⏳ Processing...")
                    
            except Exception as e:
                print(f"   ⚠️ Check failed: {e}")
        
        # 상태 출력
        icon = {"pending": "⏳", "success": "✅", "failed": "❌"}.get(current_status, "❓")
        print(f"{icon} [{current_status.upper()}] {url[:50]}...")
        print()
    
    save_store(store)
