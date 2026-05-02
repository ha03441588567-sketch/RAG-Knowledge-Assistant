from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

embeddings = HuggingFaceEmbeddings()

def process_document(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500)
    chunks = splitter.split_text(text)
    db = Chroma.from_texts(chunks, embeddings)
    return db

def query(db, question):
    docs = db.similarity_search(question, k=3)
    return docs
