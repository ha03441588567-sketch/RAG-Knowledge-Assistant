import streamlit as st
from openai import OpenAI
import PyPDF2
import io

st.title("RAG Knowledge Assistant")
st.write("AI-powered document Q&A system")

client = OpenAI(api_key="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")

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
