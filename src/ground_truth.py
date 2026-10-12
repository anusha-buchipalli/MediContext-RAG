import os

from dotenv import load_dotenv
from pinecone import Pinecone

from src.helper import download_hugging_face_embeddings


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "medical-chatbot"
)

NAMESPACE = "medical"
TOP_K = 20


# ============================================================
# EVALUATION QUESTIONS
# ============================================================

evaluation_questions = [
    "What are the symptoms of diabetes?",
    "What is diabetes mellitus?",
    "What are the complications of diabetes?",
    "What is hypertension?",
    "What is anemia?",
    "What is insulin?",
    "What is hyperglycemia?",
    "What is hypoglycemia?",
    "How is diabetes diagnosed?",
    "What is the capital of France?",
]


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
# LOAD EMBEDDING MODEL
# ============================================================

print("Loading embedding model...")

embeddings = download_hugging_face_embeddings()


# ============================================================
# RUN GROUND-TRUTH CANDIDATE RETRIEVAL
# ============================================================

print()
print("=" * 100)
print("GROUND TRUTH CANDIDATE RETRIEVAL")
print("=" * 100)

print(f"Number of evaluation questions: {len(evaluation_questions)}")
print(f"Top-K candidates per question: {TOP_K}")
print(f"Namespace: {NAMESPACE}")

print()


for question_number, query in enumerate(
    evaluation_questions,
    start=1
):

    print()
    print("#" * 100)
    print(f"QUESTION {question_number}")
    print("#" * 100)

    print(f"Question: {query}")

    # --------------------------------------------------------
    # Convert question into embedding
    # --------------------------------------------------------

    query_vector = embeddings.embed_query(query)

    # --------------------------------------------------------
    # Retrieve candidate chunks
    # --------------------------------------------------------

    results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
        namespace=NAMESPACE
    )

    matches = results["matches"]

    print(f"Number of candidates: {len(matches)}")
    print()

    # --------------------------------------------------------
    # Display retrieved candidates
    # --------------------------------------------------------

    for rank, match in enumerate(matches, start=1):

        metadata = match["metadata"]

        print("-" * 100)

        print(f"Rank: {rank}")
        print(f"Chunk ID: {match['id']}")
        print(f"Score: {match['score']:.4f}")
        print(f"Page: {metadata.get('page_label')}")
        print(f"Source: {metadata.get('source')}")

        print("Text:")

        print(metadata.get("text"))

        print()


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print()
print("=" * 100)
print("ALL GROUND TRUTH CANDIDATE RETRIEVALS COMPLETED")
print("=" * 100)