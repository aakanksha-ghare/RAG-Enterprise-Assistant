
from io import BytesIO
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


def load_documents(folder_path="Data"):
    """Load PDFs from a folder and preserve source/page metadata."""
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder.resolve()}")

    documents = []

    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))

            for document in loader.load():
                document.metadata["source"] = file_path.name
                documents.append(document)

    if not documents:
        raise ValueError(f"No readable PDF documents found in {folder.resolve()}")

    return documents


def load_uploaded_pdf(uploaded_file):
    """Extract text from an uploaded PDF without saving it permanently."""
    file_bytes = uploaded_file.getvalue()

    if not file_bytes.startswith(b"%PDF-"):
        raise ValueError(f"{uploaded_file.name} is not a valid PDF file.")

    reader = PdfReader(BytesIO(file_bytes))

    if reader.is_encrypted:
        raise ValueError(
            f"{uploaded_file.name} is encrypted. Upload an unlocked PDF."
        )

    documents = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        if text.strip():
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": uploaded_file.name,
                        "page": page_number,
                    },
                )
            )

    if not documents:
        raise ValueError(
            f"No extractable text found in {uploaded_file.name}. "
            "It may be a scanned PDF requiring OCR."
        )

    return documents


def split_documents(documents):
    """Split documents into overlapping chunks while retaining metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        add_start_index=True,
    )

    return splitter.split_documents(documents)
