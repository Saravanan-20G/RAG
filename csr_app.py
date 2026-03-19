import streamlit as st
from PyPDF2 import PdfReader

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- UI ----------------
st.title("📄 PDF Chatbot (Cloud Safe)")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")


# ---------------- CLEAN FUNCTION ----------------
def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


# ---------------- PROCESS PDF ----------------
@st.cache_resource
def process_pdf(file):

    reader = PdfReader(file)
    documents = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()

        if text:
            documents.append(
                Document(
                    page_content=clean_text(text),
                    metadata={"page": i + 1}
                )
            )

    # Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(documents)

    # Embedding
    embeddings = HuggingFaceEmbeddings()

    db = FAISS.from_documents(chunks, embeddings)

    return db


# ---------------- MAIN ----------------
if uploaded_file is not None:

    db = process_pdf(uploaded_file)

    query = st.text_input("Ask a question:")

    if query:
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        st.subheader("📌 Retrieved Answer:")
        for doc in docs:
            st.write(doc.page_content)
            st.caption(f"Page: {doc.metadata.get('page')}")