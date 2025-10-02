import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from openai import OpenAI

load_dotenv()
file_path = ("data/file-Ketab Amadegi.pdf")
loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunked_text = text_splitter.split_documents(docs)

client = OpenAI(
    base_url=os.environ.get("BASE_URL"),
    api_key=os.environ.get("API_KEY"),
)

embedding = client.embeddings.create(
  model="openai/text-embedding-3-small",
  input=chunked_text[0],
  encoding_format="float"
)

