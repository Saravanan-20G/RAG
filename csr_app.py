import streamlit as st
import fitz
from PIL import Image
import pytesseract

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ---------------- UI ----------------
st.title("📄 PDF Chatbot (RAG + OCR)")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# ---------------- CLEAN FUNCTION ----------------
def clean_text(text):
    text = text.replace("\n", " ")
    text = " ".join(text.split())
    return text


# ---------------- PROCESS PDF ----------------
@st.cache_resource
def process_pdf(file_bytes):

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    documents = []

    for i, page in enumerate(doc):

        # Extract normal text
        text = page.get_text()

        # Convert page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        # OCR
        try:
            ocr_text = pytesseract.image_to_string(img)
        except:
            ocr_text = ""

        # Avoid duplication
        full_text = text if len(text.strip()) > 50 else ocr_text

        documents.append(
            Document(
                page_content=clean_text(full_text),
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


# ---------------- MAIN FLOW ----------------
if uploaded_file is not None:

    db = process_pdf(uploaded_file.read())

    query = st.text_input("Ask a question from the PDF:")

    if query:
        retriever = db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.invoke(query)

        st.subheader("📌 Answer (retrieved content):")
        for doc in docs:
            st.write(doc.page_content)
            st.caption(f"Page: {doc.metadata.get('page')}")