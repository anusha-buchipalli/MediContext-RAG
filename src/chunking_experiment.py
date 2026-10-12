import os

from dotenv import load_dotenv
from pinecone import Pinecone

from src.helper import (
    load_pdf_file,
    text_split,
    download_hugging_face_embeddings,
)


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "medical-chatbot"
)

# Experimental configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Separate namespace for this experiment
NAMESPACE = "chunk_500_50"

BATCH_SIZE = 100


# ============================================================
# VALIDATE ENVIRONMENT
# ============================================================

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is not set in the .env file."
    )


# ============================================================
# CONNECT TO PINECONE
# ============================================================

print("Connecting to Pinecone...")

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX_NAME)


# ============================================================
# LOAD PDF
# ============================================================

print("Loading PDF...")

documents = load_pdf_file(
    "data/Medical_book.pdf"
)

print(f"Loaded documents: {len(documents)}")


# ============================================================
# CREATE EXPERIMENTAL CHUNKS
# ============================================================

print()
print("Creating experimental chunks...")

chunks = text_split(
    documents,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

print(f"Chunk size: {CHUNK_SIZE}")
print(f"Chunk overlap: {CHUNK_OVERLAP}")
print(f"Total chunks: {len(chunks)}")


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

print()
print("Loading embedding model...")

embeddings = download_hugging_face_embeddings()


# ============================================================
# UPLOAD TO EXPERIMENTAL NAMESPACE
# ============================================================

print()
print(f"Uploading to namespace: {NAMESPACE}")
print()

for start in range(
    0,
    len(chunks),
    BATCH_SIZE
):

    batch_chunks = chunks[
        start:start + BATCH_SIZE
    ]

    texts = [
        chunk.page_content
        for chunk in batch_chunks
    ]

    vectors = embeddings.embed_documents(
        texts
    )

    records = []

    for i, (chunk, vector) in enumerate(
        zip(batch_chunks, vectors)
    ):

        chunk_id = start + i

        records.append(
            {
                "id": f"{NAMESPACE}-chunk-{chunk_id}",
                "values": vector,
                "metadata": {
                    "text": chunk.page_content,
                    "source": chunk.metadata.get(
                        "source",
                        ""
                    ),
                    "page": chunk.metadata.get(
                        "page",
                        -1
                    ),
                    "page_label": chunk.metadata.get(
                        "page_label",
                        ""
                    ),
                    "chunk_size": CHUNK_SIZE,
                    "chunk_overlap": CHUNK_OVERLAP,
                },
            }
        )

    index.upsert(
        vectors=records,
        namespace=NAMESPACE
    )

    end = min(
        start + BATCH_SIZE,
        len(chunks)
    )

    print(
        f"Uploaded chunks "
        f"{start} - {end} "
        f"of {len(chunks)}"
    )


# ============================================================
# VERIFY
# ============================================================

print()
print("=" * 80)
print("EXPERIMENTAL INDEXING COMPLETED")
print("=" * 80)

print(f"Namespace: {NAMESPACE}")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Chunk overlap: {CHUNK_OVERLAP}")
print(f"Total chunks uploaded: {len(chunks)}")

print()
print("Baseline namespace 'medical' was NOT modified.")
print("=" * 80)