import os

from dotenv import load_dotenv
from pinecone import Pinecone

from src.helper import download_hugging_face_embeddings


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


# Load embedding model
embeddings = download_hugging_face_embeddings()


# User question
query = "What are the symptoms of diabetes?"


# Convert question into a vector
query_vector = embeddings.embed_query(query)


# Search Pinecone
results = index.query(
    vector=query_vector,
    top_k=3,
    include_metadata=True,
    namespace="medical"
)


print(f"Query: {query}")
print(f"Number of results: {len(results['matches'])}")
print()


for i, match in enumerate(results["matches"], start=1):
    print(f"--- Result {i} ---")
    print(f"Score: {match['score']}")
    print(f"Page: {match['metadata'].get('page_label')}")
    print("Text:")
    print(match["metadata"].get("text"))
    print()