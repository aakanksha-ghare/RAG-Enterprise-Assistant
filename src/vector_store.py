
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from src.document_loader import load_documents, split_documents


def build_vector_store():
    """Create local embeddings and save the FAISS index."""

    documents = load_documents("Data")
    chunks = split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
    )

    index_path = Path("faiss_index")
    vector_store.save_local(str(index_path))

    print(f"Pages loaded: {len(documents)}")
    print(f"Chunks indexed: {len(chunks)}")
    print(f"FAISS index saved to: {index_path.resolve()}")

    return vector_store


if __name__ == "__main__":
    build_vector_store()
