
from src.document_loader import load_documents, split_documents


documents = load_documents("Data")
chunks = split_documents(documents)

print(f"Pages loaded: {len(documents)}")
print(f"Text chunks created: {len(chunks)}")

print("\nSample chunks:")
for index, chunk in enumerate(chunks[:3], start=1):
    print(f"\n--- Chunk {index} ---")
    print(f"Source: {chunk.metadata.get('source')}")
    print(f"Page: {chunk.metadata.get('page', 'N/A')}")
    print(chunk.page_content[:250])
