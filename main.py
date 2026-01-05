"""
RAG Pipeline CLI
"""

import sys


def main():
    if len(sys.argv) < 2:
        print_help()
        return
    
    command = sys.argv[1]
    
    if command == "upload":
        from parser import upload
        if len(sys.argv) < 3:
            print("Usage: python main.py upload <pdf_url>")
            return
        upload(sys.argv[2])
        
    elif command == "status":
        from parser import status
        status()
        
    elif command == "update":
        from vectordb import update
        update()
        
    elif command == "chat":
        from chat import chat
        chat()
        
    elif command == "list":
        from vectordb import list_vectorized
        list_vectorized()
        
    elif command == "reset":
        from vectordb import reset
        urls = sys.argv[2:] if len(sys.argv) > 2 else None
        reset(urls)
    
    elif command == "delete-all":
        # 전체 초기화: ChromaDB + store.json
        confirm = input("모든 데이터를 삭제합니다 (ChromaDB + store.json). 계속하시겠습니까? [y/N]: ")
        if confirm.lower() == 'y':
            import shutil
            from settings import CHROMA_DIR, STORE_FILE
            
            # ChromaDB 삭제
            if CHROMA_DIR.exists():
                shutil.rmtree(CHROMA_DIR)
                print(f"[DELETE] ChromaDB 삭제됨: {CHROMA_DIR}")
            
            # store.json 초기화
            if STORE_FILE.exists():
                STORE_FILE.unlink()
                print(f"[DELETE] store.json 삭제됨: {STORE_FILE}")
            
            print("[완료] 모든 데이터가 삭제되었습니다.")
            print("       parsed 파일은 유지됩니다. 삭제하려면: python main.py delete-parsed")
        else:
            print("[취소됨]")
    
    elif command == "delete-parsed":
        # parsed 파일 삭제
        confirm = input("모든 parsed 파일을 삭제합니다. 계속하시겠습니까? [y/N]: ")
        if confirm.lower() == 'y':
            import shutil
            from settings import PARSED_DIR
            
            if PARSED_DIR.exists():
                for f in PARSED_DIR.iterdir():
                    f.unlink()
                print(f"[DELETE] Parsed 파일 삭제됨: {PARSED_DIR}")
            print("[완료]")
        else:
            print("[취소됨]")
    
    elif command == "reingest":
        # 전체 재벡터화: reset vectorized flags -> update
        from vectordb import reset, update
        print("[REINGEST] vectorized 플래그 리셋...")
        reset(None)  # 모든 문서 리셋
        print("[REINGEST] 재벡터화 시작...")
        update()
        print("[완료]")
        
    else:
        print(f"Unknown command: {command}")
        print_help()


def print_help():
    print("""
RAG Pipeline CLI

Commands:
  upload <url>    PDF 파싱 요청
  status          상태 확인 (자동 fetch)
  update          VectorDB 저장 (pending -> vectorized)
  list            VectorDB 문서 목록
  chat            RAG 채팅 (CLI)
  reset [url]     vectorized 리셋 (VectorDB 삭제)
  
  delete-all      전체 초기화 (ChromaDB + store.json)
  delete-parsed   Parsed 파일 삭제
  reingest        전체 재벡터화 (reset + update)

Streamlit:
  streamlit run streamlit_chat.py
""")


if __name__ == "__main__":
    main()

