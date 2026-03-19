import streamlit as st

from langchain.document_loaders import PyPDFLoader
from langchain_core.documents import Document

import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
from PIL import Image
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

file_path=r'D:\Saran\RAG\CSR MODULES (1).pdf'

loader=PyPDFLoader(file_path)

documents=loader.load()



doc = fitz.open("CSR MODULES (1).pdf")

documents = []

for i, page in enumerate(doc):

    # 1️⃣ Extract normal text
    text = page.get_text()

    # 2️⃣ Convert full page to image (for OCR)
    pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    # 3️⃣ OCR on image
    ocr_text = pytesseract.image_to_string(img)

    # 4️⃣ Combine both
    full_text = text + "\n" + ocr_text

    documents.append({
        "page": i + 1,
        "text": full_text
    })


def clean_text(text):
    text=text.replace("/n"," ")
    text=" ".join(text.split())
    return text

cleaned_docs = [
    Document(
        page_content=clean_text(doc["text"]),
        metadata=doc.get("metadata", {})
    )
    for doc in documents
]

text_splitter=RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ".", " "],
    chunk_size=2000,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(cleaned_docs)

model=SentenceTransformer("all-MiniLM-L6-v2")

texts=[doc.page_content for doc in chunks]

embeddings=model.encode(texts)

embeddings=HuggingFaceEmbeddings()
texts=[doc.page_content for doc in chunks]

db=FAISS.from_texts(texts,embeddings)
query="what is Red Flag Rule"

docs=db.similarity_search(query)

for doc in docs:
    print(doc)

retriever=db.as_retriever()

query='CREATING A PROFILE IN CCS' 

docs=retriever.invoke(query)

for doc in docs:
    print(doc.page_content)


      