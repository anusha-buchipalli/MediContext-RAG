from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


def load_pdf_file(file_path: str):
    """
    Load a PDF file and return the extracted pages
    as LangChain Document objects.
    """
    pdf_path = Path(file_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            f"Expected a PDF file, but received: {pdf_path}"
        )

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    return documents


def text_split(documents):
    """
    Split loaded documents into smaller overlapping chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    chunks = text_splitter.split_documents(documents)

    return chunks


def download_hugging_face_embeddings():
    """
    Load the Hugging Face embedding model used
    to convert text into numerical vectors.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return embeddings