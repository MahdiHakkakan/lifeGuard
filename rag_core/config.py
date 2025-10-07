from dataclasses import dataclass
import os
from dotenv import load_dotenv
import chromadb


load_dotenv()

@dataclass
class Config:
    DB_PATH: str = os.getenv("DB_PATH")
    BASE_URL: str = os.getenv("BASE_URL")
    API_KEY: str = os.getenv("API_KEY")
    AI_MODEL: str = os.getenv("AI_MODEL", "gpt-4o-mini")
    PROMPT_TEMPLATE: str = os.getenv("PROMPT_TEMPLATE")
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 800))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 200))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "lifeGuard")
    collections_directory = DB_PATH
    chroma_client = chromadb.PersistentClient(path=collections_directory)



config = Config()