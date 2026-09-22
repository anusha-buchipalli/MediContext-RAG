import os

from dotenv import load_dotenv
from pinecone import Pinecone

from src.helper import (
    load_pdf_file,
    text_split,
    download_hugging_face_embeddings,
)


load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "medical-chatbot"
)

if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is not set in the .env file."
    )


# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX_NAME)


# Load and split the medical PDF
documents = load_pdf_file("data/Medical_book.pdf")
chunks = text_split(documents)

print(f"Loaded documents: {len(documents)}")
print(f"Created chunks: {len(chunks)}")


# Load embedding model
embeddings = download_hugging_face_embeddings()


# Upload chunks in batches
batch_size = 100

for start in range(0, len(chunks), batch_size):
    batch_chunks = chunks[start:start + batch_size]

    texts = [
        chunk.page_content
        for chunk in batch_chunks
    ]

    vectors = embeddings.embed_documents(texts)

    records = []

    for i, (chunk, vector) in enumerate(
        zip(batch_chunks, vectors)
    ):
        chunk_id = start + i

        records.append(
            {
                "id": f"chunk-{chunk_id}",
                "values": vector,
                "metadata": {
                    "text": chunk.page_content,
                    "source": chunk.metadata.get(
                        "source", ""
                    ),
                    "page": chunk.metadata.get(
                        "page", -1
                    ),
                    "page_label": chunk.metadata.get(
                        "page_label", ""
                    ),
                },
            }
        )

    index.upsert(
        vectors=records,
        namespace="medical"
    )

    end = min(start + batch_size, len(chunks))

    print(
        f"Uploaded chunks {start} - {end} "
        f"of {len(chunks)}"
    )


print("All chunks uploaded successfully!")