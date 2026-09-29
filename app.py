# app.py

import streamlit as st
import os
from rag import NaiveRAG


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Naive RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-tag {
        background-color: #e8f4f8;
        padding: 0.2rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        color: #1f77b4;
        margin-right: 0.3rem;
    }
    .chunk-box {
        background-color: #ffffff;
        color: #000000;
        padding: 1rem;
        border-left: 4px solid #1f77b4;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        font-family: monospace;
        line-height: 1.5;
    }
    .distance-badge {
        background-color: #ffc107;
        color: #000000;
        padding: 0.1rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .answer-box {
        background-color: #f0f8ff;
        color: #000000;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        font-size: 1rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)


# ---------- RAG INITIALIZATION ----------
@st.cache_resource
def get_rag():
    return NaiveRAG()

rag = get_rag()


# ---------- HEADER ----------
st.markdown('<div class="main-header">📚 Naive RAG Chatbot</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Upload PDFs, ask questions, get answers with sources. '
    'Built with Streamlit + ChromaDB + Groq.</div>',
    unsafe_allow_html=True
)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("📄 Upload Documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDFs to chat with"
    )

    if st.button("🚀 Process PDFs", type="primary", use_container_width=True):
        if uploaded_files:
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, file in enumerate(uploaded_files):
                status_text.text(f"Processing {file.name}...")

                os.makedirs("data", exist_ok=True)
                path = f"data/{file.name}"
                with open(path, "wb") as f:
                    f.write(file.getbuffer())

                num = rag.ingest(path)
                st.success(f"✅ {file.name}: {num} chunks")

                progress_bar.progress((i + 1) / len(uploaded_files))

            status_text.text("Done!")
            progress_bar.empty()
        else:
            st.warning("Please upload at least one PDF")

    st.divider()

    st.header("⚙️ Settings")
    top_k = st.slider(
        "Top-K chunks",
        min_value=1,
        max_value=10,
        value=5,
        help="How many chunks to retrieve for each query"
    )

    st.divider()

    st.caption("**Built with:**")
    st.caption("Streamlit • ChromaDB • sentence-transformers • Groq")


# ---------- CHAT SESSION ----------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ---------- DISPLAY CHAT HISTORY ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(
                f'<div class="answer-box">{msg["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.write(msg["content"])

        # Sources
        if "sources" in msg and msg["sources"]:
            with st.expander("📎 Sources"):
                for src in msg["sources"]:
                    st.markdown(
                        f'<span class="source-tag">📖 {src["source"]}</span> '
                        f'<span style="color:#999; font-size:0.8rem;">chunk {src["chunk_id"]}</span>',
                        unsafe_allow_html=True
                    )

        # Retrieved chunks
        if "chunks" in msg and msg["chunks"]:
            with st.expander("🔍 Retrieved Chunks (transparency)"):
                for i, (chunk, dist) in enumerate(zip(msg["chunks"], msg["distances"])):
                    st.markdown(
                        f'<div class="chunk-box">'
                        f'<span class="distance-badge">Chunk {i+1} • distance: {dist:.3f}</span>'
                        f'<br><br>{chunk[:500]}{"..." if len(chunk) > 500 else ""}'
                        f'</div>',
                        unsafe_allow_html=True
                    )


# ---------- CHAT INPUT ----------
if query := st.chat_input("Ask a question about your PDFs..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    # Assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = rag.generate(query, top_k=top_k)
                answer = result.get("answer", "No answer generated")
            except Exception as e:
                st.error(f"Error: {e}")
                answer = f"Error generating answer: {e}"
                result = {"sources": [], "chunks": [], "distances": []}

        # Answer dikhao
        st.markdown(
            f'<div class="answer-box"><strong>💡 Answer:</strong><br><br>{answer}</div>',
            unsafe_allow_html=True
        )

        # Sources
        if result.get("sources"):
            with st.expander("📎 Sources", expanded=True):
                for src in result["sources"]:
                    st.markdown(
                        f'<span class="source-tag">📖 {src["source"]}</span> '
                        f'<span style="color:#999; font-size:0.8rem;">chunk {src["chunk_id"]}</span>',
                        unsafe_allow_html=True
                    )

        # Retrieved chunks
        if result.get("chunks"):
            with st.expander("🔍 Retrieved Chunks (transparency)"):
                for i, (chunk, dist) in enumerate(zip(result["chunks"], result["distances"])):
                    st.markdown(
                        f'<div class="chunk-box">'
                        f'<span class="distance-badge">Chunk {i+1} • distance: {dist:.3f}</span>'
                        f'<br><br>{chunk[:500]}{"..." if len(chunk) > 500 else ""}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": result.get("sources", []),
        "chunks": result.get("chunks", []),
        "distances": result.get("distances", [])
    })