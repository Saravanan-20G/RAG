import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()  # ✅ MUST be at top-level (not inside function)

def get_llm():
    key = os.getenv("GROQ_API_KEY")
    if key:
        key = key.strip() 
    elif not key:
        raise ValueError("GROQ_API_KEY missing")

    print("DEBUG KEY:", key)

    return ChatGroq(
        groq_api_key=key,
        model_name="llama3-8b-8192"
    )




# import os
# from dotenv import load_dotenv
# from langchain_openai import ChatOpenAI

# def get_llm():
#     load_dotenv()  # ✅ LOAD ENV

#     key = os.getenv("GROQ_API_KEY")

#     if not key:
#         raise ValueError("GROQ_API_KEY is missing")

#     print("DEBUG KEY:", key)  # should NOT have quotes

#     return ChatOpenAI(
#         model="llama3-8b-8192",
#         api_key=key,
#         base_url="https://api.groq.com/openai/v1"
#     )