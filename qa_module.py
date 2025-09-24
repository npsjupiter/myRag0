# qa_module.py
import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Load API key from Streamlit secrets or env
import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

OPENAI_API_KEY  = os.environ['OPENAI_API_KEY']
st.write(OPENAI_API_KEY)
def load_pdf(path: str, skip_pages: int = 0):
    loader = PyPDFLoader(path)
    pages = loader.load()
    return pages[skip_pages:] if skip_pages else pages

def build_vectordb(pages, persist_dir="chroma_store"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)
    embedding = OpenAIEmbeddings()  # uses OPENAI_API_KEY
    vectordb = Chroma.from_documents(chunks, embedding=embedding, persist_directory=persist_dir)
    try:
        vectordb.persist()
    except Exception:
        pass
    return vectordb

def build_qa_chain(vectordb):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # uses OPENAI_API_KEY
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k":4})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    return qa_chain
