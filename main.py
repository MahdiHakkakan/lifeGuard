import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
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
