"""Streamlit UI: upload documents, ask questions, get cited answers.

Run:  streamlit run app.py
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import streamlit as st

from rag.chain import RAGChain
from rag.config import settings
from rag.ingest import SUPPORTED, delete_source, ingest_paths, ingest_url, list_sources
from rag.retriever import get_retriever

st.set_page_config(page_title="RAG Document Assistant", page_icon="📄", layout="wide")


# ----------------------------------------------------------------------------
# Cached resources
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model and index…")
def _retriever():
    return get_retriever(settings)


@st.cache_resource(show_spinner=False)
def _chain():
    return RAGChain(settings, retriever=_retriever())


def _refresh_index():
    _retriever().refresh()
    st.session_state["sources"] = list_sources(settings)


if "sources" not in st.session_state:
    st.session_state["sources"] = list_sources(settings)
if "messages" not in st.session_state:
    st.session_state["messages"] = []


# ----------------------------------------------------------------------------
# Sidebar: corpus management
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("📄 Documents")
    st.caption(
        f"LLM: `{settings.llm_provider}` · embeddings: `{settings.embedding_model.split('/')[-1]}` · "
        f"retrieval: {'hybrid (dense + BM25)' if settings.hybrid else 'dense'} · top-k {settings.top_k}"
    )

    uploads = st.file_uploader(
        "Upload PDF / DOCX / MD / TXT", type=[s.lstrip(".") for s in SUPPORTED], accept_multiple_files=True
    )
    if uploads and st.button("Ingest uploaded files", type="primary", use_container_width=True):
        with st.spinner("Chunking and embedding…"):
            tmp = Path(tempfile.mkdtemp())
            paths = []
            for f in uploads:
                p = tmp / f.name
                p.write_bytes(f.getbuffer())
                paths.append(p)
            t0 = time.time()
            stats = ingest_paths(paths, settings)
            _refresh_index()
        st.success(f"Indexed {stats['files']} file(s), {stats['pages']} page(s), {stats['chunks']} chunks in {time.time()-t0:.1f}s")
        if stats["skipped"]:
            st.warning("Skipped: " + ", ".join(stats["skipped"]))

    url = st.text_input("…or add a web page URL")
    if url and st.button("Ingest URL", use_container_width=True):
        with st.spinner("Fetching and embedding…"):
            try:
                stats = ingest_url(url, settings)
                _refresh_index()
                st.success(f"Indexed {stats['chunks']} chunks from {url}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not ingest URL: {e}")

    st.divider()
    st.subheader(f"Indexed sources ({len(st.session_state['sources'])})")
    if not st.session_state["sources"]:
        st.info("No documents yet. Upload files above or run `python scripts/ingest.py`.")
    for src, n in st.session_state["sources"].items():
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{src}**  \n<span style='color:gray;font-size:0.8em'>{n} chunks</span>", unsafe_allow_html=True)
        if c2.button("✕", key=f"del-{src}", help=f"Remove {src} from the index"):
            delete_source(src, settings)
            _refresh_index()
            st.rerun()

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# ----------------------------------------------------------------------------
# Main: chat
# ----------------------------------------------------------------------------
st.title("RAG Document Assistant")
st.caption("Ask questions about the indexed documents. Answers cite passages as [n]; expand a source to verify.")


def render_sources(sources, cited):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)} passages retrieved, {len(cited) or 'all'} cited)"):
        for s in sources:
            mark = "✅" if s.n in cited else "▫️"
            st.markdown(f"{mark} **[{s.n}] {s.label}**" + (f" · rrf {s.score}" if s.score else ""))
            st.text(s.text[:1200] + ("…" if len(s.text) > 1200 else ""))


for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            render_sources(m["sources"], m.get("cited", []))

question = st.chat_input("e.g. What is the regulatory retail portfolio limit under the Basel framework?")
if question:
    if not st.session_state["sources"]:
        st.warning("Index some documents first.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state["messages"][:-1]]
    with st.chat_message("assistant"):
        try:
            sources, tokens = _chain().stream(question, history=history)
            answer = st.write_stream(tokens)
        except Exception as e:  # noqa: BLE001
            st.error(f"LLM call failed: {e}")
            st.stop()
        import re

        cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(sources)})
        render_sources(sources, cited)

    st.session_state["messages"].append(
        {"role": "assistant", "content": answer, "sources": sources, "cited": cited}
    )
