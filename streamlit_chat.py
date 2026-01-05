"""
Streamlit RAG Chat - 웹 UI
LangGraph 워크플로우 사용
"""

import streamlit as st
from rag import stream_response, get_response_with_metadata

st.set_page_config(page_title="RAG Chat", page_icon="📚", layout="wide")
st.title("📚 RAG Chat (LangGraph)")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_debug" not in st.session_state:
    st.session_state.show_debug = False

# 히스토리 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("debug") and st.session_state.show_debug:
            with st.expander("Debug Info"):
                st.json(msg["debug"])

# 사용자 입력
if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # AI 응답
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        try:
            if st.session_state.show_debug:
                # Debug 모드: 전체 메타데이터
                result = get_response_with_metadata(prompt)
                full_response = result.get("answer", "")
                response_placeholder.markdown(full_response)
                
                with st.expander("🔍 Flow Log"):
                    for entry in result.get("flow_log", []):
                        st.text(entry)
                
                with st.expander("📄 Evidence"):
                    for ev in result.get("evidence", []):
                        st.text(f"• {ev.get('filename')} p.{ev.get('page')}")
                
                debug_info = {
                    "iterations": result.get("iterations"),
                    "sufficiency": str(result.get("sufficiency")),
                    "hallucination": str(result.get("hallucination")),
                }
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response,
                    "debug": debug_info
                })
            else:
                # 스트리밍 모드
                for chunk in stream_response(prompt):
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")
                
                response_placeholder.markdown(full_response)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })
                
        except Exception as e:
            st.error(f"오류: {e}")

# 사이드바
with st.sidebar:
    st.header("⚙️ Settings")
    
    st.session_state.show_debug = st.toggle("Debug Mode", st.session_state.show_debug)
    
    st.divider()
    
    if st.button("🗑️ 대화 초기화"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    
    st.subheader("📊 Pipeline Info")
    st.caption("""
    - Planner: Self-Query
    - Retrieval: ANN (Chroma)
    - Rerank: LLM-based
    - Sufficiency: Loop if needed
    - Hallucination: Grounding check
    """)
    
    st.divider()
    st.caption("CLI: `python main.py chat`")
