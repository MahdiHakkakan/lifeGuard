# Lifeguard RAG Assistant 🏊‍♂️🤖

A Retrieval-Augmented Generation (RAG) system for answering questions about lifeguard training documents.  
This project uses LangChain, OpenAI, ChromaDB, and PyPDFLoader to enable semantic search and question answering over PDFs.

## Features
- 📄 Load lifeguard training PDFs with **PyPDFLoader**
- 🔍 Store embeddings in **ChromaDB** for fast retrieval
- 🤖 Generate answers with **OpenAI LLMs** via LangChain
- 💬 Ask natural language questions and get accurate answers
- 🛠 Simple and extensible codebase for learning RAG systems

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/lifeguard-rag.git
   cd lifeguard-rag

pip install -r requirements.txt

---

### 4. استفاده (Usage)
```markdown
## Usage

1. Place your lifeguard PDF file in the `data/` directory.

2. Run the script to load the PDF, build the vector store, and start querying:
   ```bash
   python main.py
```


---

### 5. ساختار پروژه (Project Structure)
```markdown
## Project Structure
.
├── data/                # PDF documents (e.g., lifeguard manuals)
├── main.py              # Main entry point for RAG system
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
```

## Roadmap
- [ ] Add support for multiple PDFs
- [ ] Create a simple web UI (Streamlit or Gradio)
- [ ] Add memory for multi-turn conversations
- [ ] Deploy the system with Docker

## License
MIT License

## Acknowledgments
- [LangChain](https://www.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [OpenAI](https://platform.openai.com/)
