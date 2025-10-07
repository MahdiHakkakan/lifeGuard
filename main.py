import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
import chromadb
from openai import OpenAI
import sys

load_dotenv()
collections_directory = os.environ.get("DB_PATH")
chroma_client = chromadb.PersistentClient(path=collections_directory)

client = OpenAI(
    base_url=os.environ.get("BASE_URL"),
    api_key=os.environ.get("API_KEY"),
)
ai_model = os.environ.get("AI_MODEL")


def embed_file(file_name):
    file_path = (f"data/{file_name}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunked_text = text_splitter.split_documents(docs)
    collection = chroma_client.get_or_create_collection(name="lifeGuard")
    for i, doc in enumerate(chunked_text):
        collection.upsert(
            ids=[f"id{i+1}"],
            documents=[doc.page_content],
        )



rag_prompt_template = os.environ.get("PROMPT_TEMPLATE")
collection = chroma_client.get_collection("lifeGuard")

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
    return {"answer": response.choices[0].message.content}

if __name__ == "__main__":
    if sys.argv[1] == "ask":
        question = sys.argv[2].strip()

        # مرحله ۱: بازیابی
        retrieved = retrieve({"question": question})

        # مرحله ۲: ساخت پاسخ
        state = {"question": question, "context": retrieved["context"]}
        answer = generate(state)

        print("\n🤖 پاسخ مدل:")
        print(answer)
    elif sys.argv[1] == "load":
        file_name = sys.argv[2]
        embed_file(file_name)
