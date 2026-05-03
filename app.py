import streamlit as st
import openai
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.title("RAG Knowledge Assistant")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_texts(chunks, embeddings)

    question = st.text_input("Ask a question:")
    if question:
        llm = ChatOpenAI()
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())
        answer = qa.run(question)
        st.write(answer)
