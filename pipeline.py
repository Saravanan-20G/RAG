# ingest/pipeline.py

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


def run_ingestion(file_path: str):
    # 1. Load
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    # 2. Split
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # 3. Embed
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 4. Store
    db = Chroma.from_documents(
        chunks,
        embedding,
        persist_directory="db"
    )

    

    print(f"Docs: {len(docs)}")
    print(f"Chunks: {len(chunks)}")