"""
Chat - RAG 채팅 CLI
LangGraph 워크플로우 호출
"""

from rag import stream_response, get_response_with_metadata
from settings import logger


def chat():
    """CLI 채팅"""
    print("\n" + "="*60)
    print("RAG Chat (LangGraph) | 종료: 'quit' | 상세: 'debug'")
    print("="*60 + "\n")
    
    debug_mode = False
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Bye!")
                break
            
            if user_input.lower() == "debug":
                debug_mode = not debug_mode
                print(f"[DEBUG MODE: {'ON' if debug_mode else 'OFF'}]")
                continue
            
            print("\nAI: ", end="", flush=True)
            
            if debug_mode:
                # 전체 메타데이터 포함 응답
                result = get_response_with_metadata(user_input)
                print(result.get("answer", ""))
                
                print("\n--- Debug Info ---")
                print(f"Iterations: {result.get('iterations', 1)}")
                
                sufficiency = result.get("sufficiency")
                if sufficiency:
                    print(f"Sufficiency: {sufficiency.score:.2f} ({sufficiency.verdict})")
                
                hallucination = result.get("hallucination")
                if hallucination:
                    print(f"Hallucination: grounded={hallucination.grounded}")
                
                print("\nFlow Log:")
                for entry in result.get("flow_log", []):
                    print(f"  {entry}")
                
                print("\nEvidence:")
                for ev in result.get("evidence", [])[:3]:
                    print(f"  - {ev.get('filename')} p.{ev.get('page')}")
            else:
                # 스트리밍 응답
                for chunk in stream_response(user_input):
                    print(chunk, end="", flush=True)
            
            print("\n")
            
        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"\n[ERROR] {e}\n")
            logger.exception("Chat error")


if __name__ == "__main__":
    chat()
