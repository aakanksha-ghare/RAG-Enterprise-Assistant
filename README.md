@'
# RAG Enterprise Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to ask questions about PDF documents and receive context-grounded answers using a locally running Large Language Model.

## Features

- PDF document loading and text extraction.
- Document chunking with overlapping text segments.
- Semantic embeddings using Hugging Face Sentence Transformers.
- FAISS vector database for similarity search.
- Retrieval of the top 4 relevant document passages.
- Local answer generation using Ollama and Llama 3.2 (3B).
- Source and page-level citations.
- Streamlit chat interface with conversation history.
- Retrieved passage inspection for transparency.

## Technology Stack

- Python
- Streamlit
- LangChain
- Hugging Face Sentence Transformers
- FAISS
- Ollama (Llama 3.2:3B)
- PyPDF

## Architecture

1. PDF documents are loaded and their text is extracted.
2. Documents are divided into overlapping chunks.
3. Hugging Face generates vector embeddings.
4. FAISS stores the embeddings and enables similarity search.
5. User queries retrieve the four most relevant passages.
6. Retrieved context is passed to the local LLM through Ollama.
7. The generated answer includes source and page references.

## Project Structure

```text
RAG-Enterprise-Assistant/
├── app.py
├── src/
│   ├── document_loader.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── rag_pipeline.py
│   └── __init__.py
├── data/
├── requirements.txt
├── .gitignore
└── README.md