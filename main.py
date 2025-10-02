from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

file_path = ("data/file-Ketab Amadegi.pdf")
loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunked_text = text_splitter.split_documents(docs)

