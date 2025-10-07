from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import config

def embed_file(file_name):
    file_path = (f"data/{file_name}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    chunked_text = text_splitter.split_documents(docs)
    collection = config.chroma_client.get_or_create_collection(name="lifeGuard")
    for i, doc in enumerate(chunked_text):
        collection.upsert(
            ids=[f"id{i+1}"],
            documents=[doc.page_content],
        )