import streamlit as st
from PyPDF2 import PdfReader
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
import os

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.title("RAG Knowledge Assistant")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    question = st.text_input("Ask a question:")
    if question:
        context = chunks[0] if chunks else ""
        llm = ChatOpenAI()
        answer = llm.invoke(f"Context:\n{context}\n\nQuestion: {question}")
        st.write(answer.content)
