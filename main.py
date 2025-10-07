from typing_extensions import List, TypedDict
from rag_core import loader
from langchain_core.documents import Document
from openai import OpenAI
import sys
from rag_core.config import config

client = OpenAI(
    base_url=config.BASE_URL,
    api_key=config.API_KEY,
)
ai_model = config.AI_MODEL


rag_prompt_template = config.PROMPT_TEMPLATE
collection = config.chroma_client.get_collection("lifeGuard")

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
        loader.embed_file(file_name)
