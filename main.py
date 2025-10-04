import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
import chromadb
from openai import OpenAI

load_dotenv()
chroma_client = chromadb.Client()
client = OpenAI(
    base_url=os.environ.get("BASE_URL"),
    api_key=os.environ.get("API_KEY"),
)

file_path = ("data/file-Ketab Amadegi.pdf")
loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunked_text = text_splitter.split_documents(docs)

collection = chroma_client.create_collection(name="lifeGuard")
for i in range(len(chunked_text)-1):
    collection.upsert(
        ids=[f"id{i+1}"],
        documents=[str(chunked_text[i])],
    )

results = collection.query(
    query_texts=["تست"],
    n_results=2
)


rag_prompt_template = """
You are a helpful and friendly assistant. 
Your task is to answer the user's question ONLY based on the given context. 
- If the answer exists in the context → provide it clearly, with a short and friendly explanation. 
- If the answer does not exist in the context → explicitly say you don't know, in a polite and empathetic way, and encourage the user to check other sources. 
- Keep answers concise, friendly, and natural. 
- Always follow the style of the examples below.

Examples:
Q: "انواع رگ ها کدام است؟"
A: "سوال خیلی خوبی پرسیدی👍  
جواب سوالت میشه سرخرگ، سیاه‌رگ و مویرگ.  
اگه احیانا هر سوال دیگه‌ای داشتی من همین جام🙂"

Q: "رئیس‌جمهور آمریکا کیه؟"
A: "هممممم🧐 جواب این سوالتو متاسفانه من نمیدونم😢  
اما شاید بتونی از منبع‌های دیگه جواب این سوالو پیدا کنی."

---

Context:
{context}

Question:
{question}

Answer:
"""


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    retrieved_docs = collection.query(query_texts=state["question"], n_results=2)
    return {"context": retrieved_docs}