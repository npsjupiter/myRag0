# app.py
"""
Streamlit RAG app that supports uploading multiple PDFs (accept_multiple_files=True),
building or updating a Chroma index, and chatting over all indexed docs.

Requirements (pip):
  streamlit langchain langchain-community langchain-openai chromadb openai pypdf tiktoken
"""

import os
import tempfile
import shutil
import time
from pathlib import Path
from typing import List

import streamlit as st
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# --------------- Configuration ----------------
PERSIST_DIR = os.environ.get("PERSIST_DIR", "chroma_store")  # path inside container
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVER_K = 4

# --------------- Helpers -----------------------
def ensure_api_key():
    # Prefer Streamlit secrets, then env var
    key = st.secrets.get("OPENAI_API_KEY") if hasattr(st, "secrets") else None
    key = key or os.getenv("OPENAI_API_KEY")
    if not key:
        st.error("OPENAI_API_KEY not found. Set it under Streamlit Secrets or environment variable.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = key
    return key

def save_uploaded_files(uploaded_files, target_dir: str) -> List[str]:
    """Save uploaded file-like objects to disk and return list of paths."""
    os.makedirs(target_dir, exist_ok=True)
    paths = []
    for up in uploaded_files:
        # Create a safe unique name
        out_path = os.path.join(target_dir, up.name)
        # If same name exists, add a suffix
        if os.path.exists(out_path):
            base, ext = os.path.splitext(out_path)
            timestamp = int(time.time() * 1000)
            out_path = f"{base}_{timestamp}{ext}"
        with open(out_path, "wb") as f:
            f.write(up.getbuffer())
        paths.append(out_path)
    return paths

def load_pdfs_with_metadata(paths: List[str], skip_pages: int = 0) -> List[Document]:
    """Load PDFs and attach metadata {'source': filename, 'page': n}."""
    docs = []
    for p in paths:
        loader = PyPDFLoader(p)
        pages = loader.load()
        if skip_pages:
            pages = pages[skip_pages:]
        # attach metadata per page
        filename = os.path.basename(p)
        for i, pg in enumerate(pages):
            meta = dict(pg.metadata or {})
            meta["source"] = filename
            # if loader gave page number in metadata, preserve or set one
            meta.setdefault("page", i)
            docs.append(Document(page_content=pg.page_content, metadata=meta))
    return docs

def split_and_dedup(docs: List[Document], chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    # dedupe by first 250 chars
    seen = set()
    unique = []
    for d in chunks:
        key = (d.metadata.get("source", "") , d.page_content.strip()[:250])
        if key not in seen and d.page_content and d.page_content.strip():
            seen.add(key)
            unique.append(d)
    return unique

def build_or_update_chroma(chunks: List[Document], persist_dir=PERSIST_DIR, collection_name="default_collection"):
    """
    If an existing collection exists at persist_dir, load it and add_documents.
    Otherwise, create a new Chroma with from_documents.
    """
    embeddings = OpenAIEmbeddings()
    # Try to instantiate wrapper pointed at persist_dir
    try:
        if os.path.exists(persist_dir) and os.listdir(persist_dir):
            # load existing
            vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            # add_documents expects list[Document]
            if hasattr(vectordb, "add_documents"):
                vectordb.add_documents(chunks)
            else:
                # fallback: try low-level client
                coll = getattr(vectordb, "_collection", None) or getattr(vectordb, "collection", None)
                if coll:
                    ids = [f"doc-{int(time.time()*1000)}-{i}" for i, _ in enumerate(chunks)]
                    docs = [d.page_content for d in chunks]
                    metas = [d.metadata for d in chunks]
                    coll.add(ids=ids, documents=docs, metadatas=metas)
            return vectordb
        else:
            # create new
            vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
            # persist if wrapper allows
            try:
                vectordb.persist()
            except Exception:
                pass
            return vectordb
    except Exception as e:
        st.error(f"Failed to build/update Chroma index: {e}")
        raise

def get_qa_chain_from_vectordb(vectordb):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": RETRIEVER_K})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, return_source_documents=True, output_key="answer")
    return chain

# --------------- Streamlit UI -------------------
st.set_page_config(page_title="RAG Multi-PDF Chat", layout="wide")
st.title("RAG Chat — upload multiple PDFs and chat across them")

# Ensure API key
ensure_api_key()

# Sidebar controls
st.sidebar.header("Index controls")
uploaded = st.sidebar.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)
skip_pages = st.sidebar.number_input("Skip first N pages per PDF", min_value=0, value=0, step=1)
rebuild = st.sidebar.button("Rebuild index from repo/data (delete existing)")
add_to_index = st.sidebar.button("Add uploaded PDFs to index")

# Show current persist dir info
st.sidebar.write(f"Chroma persist dir: `{PERSIST_DIR}`")
if os.path.exists(PERSIST_DIR):
    st.sidebar.write(f"Persist dir exists, size (approx): {sum(p.stat().st_size for p in Path(PERSIST_DIR).rglob('*'))/1024/1024:.2f} MB")
else:
    st.sidebar.write("No existing persist dir found (cold start).")

# Temporary storage for uploaded files
tmp_folder = os.path.join(tempfile.gettempdir(), "streamlit_rag_uploads")
os.makedirs(tmp_folder, exist_ok=True)

# Rebuild logic (delete and rebuild index)
if rebuild:
    if os.path.exists(PERSIST_DIR):
        try:
            shutil.rmtree(PERSIST_DIR)
            st.sidebar.success("Deleted existing persist dir. Now upload PDFs or ensure data/ exists to rebuild.")
        except Exception as e:
            st.sidebar.error(f"Failed to delete persist dir: {e}")

# If user uploaded files and clicked add_to_index
if uploaded and add_to_index:
    with st.spinner("Saving uploaded files..."):
        paths = save_uploaded_files(uploaded, tmp_folder)
        st.sidebar.write("Saved files:", [Path(p).name for p in paths])

    with st.spinner("Loading and processing PDFs..."):
        docs = load_pdfs_with_metadata(paths, skip_pages=skip_pages)
        st.write(f"Loaded {len(docs)} page-level documents from uploaded PDFs.")
        chunks = split_and_dedup(docs)
        st.write(f"Produced {len(chunks)} chunks to index (after split & dedup).")

        with st.spinner("Building/updating Chroma index (this calls OpenAI for embeddings)..."):
            vectordb = build_or_update_chroma(chunks, persist_dir=PERSIST_DIR)
            st.success("Index updated. You can now ask questions in the chat below.")
            # store in session_state
            st.session_state["vectordb"] = vectordb
            st.session_state.pop("qa_chain", None)  # rebuild QA chain

# If no vectordb in session, try to load existing from persist dir
if "vectordb" not in st.session_state:
    try:
        if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
            embeddings = OpenAIEmbeddings()
            st.session_state["vectordb"] = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            st.success("Loaded existing index from persist dir.")
    except Exception as e:
        st.warning(f"Could not load persist dir automatically: {e}")

# Build QA chain if vectordb exists and chain not in session
if "vectordb" in st.session_state and "qa_chain" not in st.session_state:
    try:
        st.session_state["qa_chain"] = get_qa_chain_from_vectordb(st.session_state["vectordb"])
    except Exception as e:
        st.error(f"Failed to create QA chain: {e}")

# Chat UI
st.markdown("---")
st.subheader("Chat with your documents")
if "qa_chain" in st.session_state:
    # Use chat_input for nicer UI when available
    user_query = st.chat_input("Ask a question about the indexed PDFs...")
    if user_query:
        with st.spinner("Generating answer..."):
            try:
                out = st.session_state["qa_chain"].invoke({"question": user_query})
                answer = out.get("answer") or out.get("output_text") or ""
                st.chat_message("user").write(user_query)
                st.chat_message("assistant").write(answer)
                # show sources
                srcs = out.get("source_documents", [])
                if srcs:
                    with st.expander("Sources (top 3)"):
                        for s in srcs[:3]:
                            src = s.metadata.get("source", "(no source)")
                            page = s.metadata.get("page", None)
                            st.write(f"**Source:** {src} (page: {page})")
                            st.write(s.page_content[:800] + "...")
            except Exception as e:
                st.error(f"Chain call failed: {e}")
else:
    st.info("No index loaded yet. Upload PDFs and click 'Add uploaded PDFs to index', or ensure a persist dir exists.")

# optional: show persisted collection summary
if "vectordb" in st.session_state:
    try:
        coll = getattr(st.session_state["vectordb"], "_collection", None) or getattr(st.session_state["vectordb"], "collection", None)
        if coll:
            cnt = coll.count()
            st.sidebar.write(f"Indexed chunks (approx): {cnt}")
    except Exception:
        pass
