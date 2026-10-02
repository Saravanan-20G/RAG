# main.py

from ingest.pipeline import run_ingestion
from core.retriever import get_retriever
from core.generator import get_llm
from core.rag import run_rag


if __name__ == "__main__":
    # Run once
    run_ingestion(r"D:\Saran\Project_RAG\CSR MODULES (1).pdf")

    retriever = get_retriever()
    llm = get_llm()

    while True:
        q = input("Ask: ")
        print(run_rag(q, retriever, llm))