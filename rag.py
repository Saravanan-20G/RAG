# core/rag.py

def run_rag(query, retriever, llm):
    docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You must answer ONLY using the context below.
If answer not found, say "I don't know".

Context:
{context}

Question:
{query}
"""

    response = llm.invoke(prompt)

    return response.content