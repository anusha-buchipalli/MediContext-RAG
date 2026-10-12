import os

from dotenv import load_dotenv
from pinecone import Pinecone

from src.helper import download_hugging_face_embeddings


# --------------------------------------------------
# 1. Load environment variables
# --------------------------------------------------

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


# --------------------------------------------------
# 2. Connect to Pinecone
# --------------------------------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index(PINECONE_INDEX_NAME)


# --------------------------------------------------
# 3. Load embedding model
# --------------------------------------------------

embeddings = download_hugging_face_embeddings()


# --------------------------------------------------
# 4. Evaluation questions
# --------------------------------------------------

evaluation_questions = [

    {
        "id": "R01",
        "question": "What are the symptoms of diabetes?"
    },

    {
        "id": "R02",
        "question": "What is diabetes mellitus?"
    },

    {
        "id": "R03",
        "question": "What are the complications of diabetes?"
    },

    {
        "id": "R04",
        "question": "What is hypertension?"
    },

    {
        "id": "R05",
        "question": "What is anemia?"
    },

    {
        "id": "R06",
        "question": "What is insulin?"
    },

    {
        "id": "R07",
        "question": "What is hyperglycemia?"
    },

    {
        "id": "R08",
        "question": "What is hypoglycemia?"
    },

    {
        "id": "R09",
        "question": "How is diabetes diagnosed?"
    },

    {
        "id": "R10",
        "question": "What is the capital of France?"
    }

]


# --------------------------------------------------
# 5. Retrieve Top-5 results for each question
# --------------------------------------------------

TOP_K = 5


for test_case in evaluation_questions:

    test_id = test_case["id"]
    query = test_case["question"]


    # Convert question into an embedding
    query_vector = embeddings.embed_query(query)


    # Search Pinecone
    results = index.query(
        vector=query_vector,
        top_k=TOP_K,
        include_metadata=True,
        namespace="medical"
    )


    # --------------------------------------------------
    # 6. Display results
    # --------------------------------------------------

    print("=" * 80)

    print(f"Test ID: {test_id}")
    print(f"Query: {query}")
    print(f"Number of results: {len(results['matches'])}")

    print()


    for rank, match in enumerate(
        results["matches"],
        start=1
    ):

        print(f"--- Rank {rank} ---")

        print(
            f"Score: {match['score']}"
        )

        print(
            f"Page: "
            f"{match['metadata'].get('page_label')}"
        )

        print(
            "Text:"
        )

        print(
            match["metadata"].get("text")
        )

        print()


print("=" * 80)

print("Retrieval evaluation completed.")