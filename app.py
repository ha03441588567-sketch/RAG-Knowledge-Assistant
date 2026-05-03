import streamlit as st
from openai import OpenAI
import PyPDF2
import io

st.title("RAG Knowledge Assistant")
st.write("AI-powered document Q&A system")

client = OpenAI(api_key="sOPENAI_API_KEY ="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")

uploaded_file=st.file_uploader("Upload PDF",type="pdf")
if uploaded_file:
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    st.success("PDF upload ho gaya!")
    question = st.text_input("Apna question likho:")
    if question:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Answer based on this document: {text[:3000]}"},
                {"role": "user", "content": question}
            ]
        )
        st.write("**Answer:**", response.choices[0].message.content)
