# app.py (relevant parts only)

import os
import streamlit as st
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# CONFIG
PERSIST_DIR = os.environ.get("PERSIST_DIR", "chroma_store")   # local path
DATA_DIR = "data"   # put small PDFs here in your repo for auto-build
EMBED_MODEL = "openai"  # or "hf" depending on what you use

# Helper: load files from repo/data
def load_documents_from_data_dir(data_dir=DATA_DIR, skip_pages=0):
    docs = []
    # PDFs
    for p in os.listdir(data_dir):
        if p.lower().endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, p))
            pages = loader.load()
            if skip_pages:
                pages = pages[skip_pages:]
            # attach source
            for pg in pages:
                if not pg.metadata:
                    pg.metadata = {}
                pg.metadata["source"] = p
            docs.extend(pages)
    # TXT/MD
    for p in os.listdir(data_dir):
        if p.lower().endswith((".txt", ".md")):
            loader = TextLoader(os.path.join(data_dir, p), encoding="utf8")
            for d in loader.load():
                if not d.metadata:
                    d.metadata = {}
                d.metadata["source"] = p
                docs.append(d)
    return docs

def build_chroma_from_docs(docs, persist_dir=PERSIST_DIR):
    # clean & split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    # embeddings
    embeddings = OpenAIEmbeddings()  # uses OPENAI_API_KEY from env / secrets
    vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
    # persist only works on some wrappers; safe to call if present
    try:
        vectordb.persist()
    except Exception:
        pass
    return vectordb

# App startup: ensure OPENAI_API_KEY exists (Streamlit secrets or env)
if not (os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", None)):
    st.warning("OPENAI_API_KEY not set. Set it in Streamlit secrets or environment.")
else:
    # load or build
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        try:
            embeddings = OpenAIEmbeddings()
            vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            st.info("Loaded existing index from persist dir.")
        except Exception as e:
            st.warning("Failed to load existing index; will rebuild: " + str(e))
            docs = load_documents_from_data_dir(DATA_DIR)
            vectordb = build_chroma_from_docs(docs)
    else:
        # build from repo/data automatically on cold start
        docs = load_documents_from_data_dir(DATA_DIR)
        if not docs:
            st.info("No files in data/ to build from. Upload files or put PDFs in data/ in the repo.")
        else:
            with st.spinner("Building index from repo/data (this may take a minute)..."):
                vectordb = build_chroma_from_docs(docs)
                st.success("Index built.")

    # once vectordb exists, create retriever and chain (same as your local code)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":4})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, return_source_documents=True, output_key="answer")

    # Put the chat UI here (text_input, display)
    # ...
if 'qa_chain' not in st.session_state and 'vectordb' in locals():
    # build qa_chain once after loading vectordb
    from langchain_openai import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferMemory

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )

# UI for chat
st.subheader("Ask questions about your data")

if "qa_chain" in st.session_state:
    user_query = st.text_input("Enter your question:")
    if user_query:
        with st.spinner("Thinking..."):
            result = st.session_state.qa_chain.invoke({"question": user_query})
            st.write("### 🤖 Answer:")
            st.write(result["answer"])
            with st.expander("Sources"):
                for i, doc in enumerate(result["source_documents"], start=1):
                    st.markdown(f"**Source {i}:** {doc.metadata}")
                    st.write(doc.page_content[:400] + "…")
