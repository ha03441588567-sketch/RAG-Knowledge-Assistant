import streamlit as st
import openai
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

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
        docs = db.similarity_search(question)
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=question)
        st.write(answer)
