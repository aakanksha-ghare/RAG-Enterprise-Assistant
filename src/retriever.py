
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


INDEX_PATH = Path("faiss_index")


def load_vector_store():
    """Load the saved FAISS index using the same embedding model."""

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            "FAISS index not found. Run: python -m src.vector_store"
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    return FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def retrieve_documents(question: str, k: int = 4):
    """Retrieve the most relevant document chunks for a question."""

    if not question.strip():
        raise ValueError("Please enter a question.")

    vector_store = load_vector_store()
    results = vector_store.similarity_search_with_score(question, k=k)

    return results


if __name__ == "__main__":
    question = input("Ask a question about the documents: ")

    results = retrieve_documents(question)

    print("\nRelevant passages:\n")

    for rank, (document, distance) in enumerate(results, start=1):
        print(f"--- Result {rank} ---")
        print(f"Source: {document.metadata.get('source', 'Unknown')}")
        print(f"Page: {document.metadata.get('page', 'Unknown')}")
        print(f"Distance: {distance:.4f}")
        print(document.page_content[:700])
        print()
