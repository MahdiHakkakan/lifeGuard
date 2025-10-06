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
ai_model = os.environ.get("AI_MODEL")

file_path = ("data/file-Ketab Amadegi.pdf")
loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunked_text = text_splitter.split_documents(docs)

collection = chroma_client.create_collection(name="lifeGuard")
for i, doc in enumerate(chunked_text):
    collection.upsert(
        ids=[f"id{i+1}"],
        documents=[doc.page_content],
    )

results = collection.query(
    query_texts=["تست"],
    n_results=2
)


rag_prompt_template = os.environ.get("PROMPT_TEMPLATE")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


def retrieve(state: State):
    results = collection.query(query_texts=[state["question"]], n_results=2)
    context_docs = [Document(page_content=doc) for doc in results["documents"][0]]
    return {"context": context_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    message = rag_prompt_template.format(context=docs_content, question=state["question"])
    response = client.chat.completions.create(
        model=ai_model,
        messages=[
            {"role": "system", "content": "You are a RAG system assistant."},
            {"role": "user", "content": message},
        ],
    )
    return response.choices[0].message.content
