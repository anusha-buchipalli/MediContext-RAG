import os

from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_groq import ChatGroq

from src.helper import download_hugging_face_embeddings


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "medical-chatbot"
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if not PINECONE_API_KEY:
    raise ValueError(
        "PINECONE_API_KEY is not set in the .env file."
    )

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY is not set in the .env file."
    )


# ============================================================
# CONNECT TO PINECONE
# ============================================================

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index = pc.Index(
    PINECONE_INDEX_NAME
)


# ============================================================
# LOAD HUGGING FACE EMBEDDING MODEL
# ============================================================

embeddings = download_hugging_face_embeddings()


# ============================================================
# CONNECT TO GROQ
# ============================================================

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)


# ============================================================
# GET ANSWER
# ============================================================

def get_answer(question: str):

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question or not question.strip():
        return "Please enter a question."


    question = question.strip()


    # --------------------------------------------------------
    # Convert question into embedding
    # --------------------------------------------------------

    query_vector = embeddings.embed_query(
        question
    )


    # --------------------------------------------------------
    # Search Pinecone
    # --------------------------------------------------------

    search_results = index.query(
        vector=query_vector,
        top_k=3,
        include_metadata=True,
        namespace="medical"
    )


    # --------------------------------------------------------
    # Extract retrieved context
    # --------------------------------------------------------

    context_parts = []

    for match in search_results["matches"]:

        metadata = match.get("metadata", {})

        text = metadata.get("text", "")

        if text:
            context_parts.append(text)


    context = "\n\n".join(context_parts)


    # --------------------------------------------------------
    # Check whether context was retrieved
    # --------------------------------------------------------

    if not context.strip():

        return (
            "I could not find relevant information in the "
            "medical knowledge base to answer this question."
        )


    # --------------------------------------------------------
    # RAG PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a medical information assistant.

Your knowledge base is the medical information provided in
the CONTEXT section below.

Answer the user's question using the provided context.

CONTEXT:
{context}

USER QUESTION:
{question}

INSTRUCTIONS:

1. Answer the user's question clearly and naturally.

2. Use the provided medical context as the primary source.

3. Do not invent facts that are not supported by the context.

4. If the context does not contain enough information to
   answer the question, say:

   "I don't have enough information in the medical
   knowledge base to answer that question."

5. Do not diagnose the user.

6. Do not prescribe medication.

7. Do not give personalized treatment instructions.

8. If the question is unrelated to the medical knowledge
   base, explain that the chatbot is designed to answer
   questions based on its medical knowledge base.

9. Keep the answer easy to understand.

10. Use bullet points or numbered lists when they make
    the answer easier to read.

11. Do not mention Pinecone, Groq, embeddings, retrieval,
    prompts, or the internal system to the user.

ANSWER:
"""


    # --------------------------------------------------------
    # SEND PROMPT TO GROQ
    # --------------------------------------------------------

    response = llm.invoke(prompt)


    # --------------------------------------------------------
    # EXTRACT RESPONSE
    # --------------------------------------------------------

    content = response.content


    if isinstance(content, list):

        text_parts = []

        for item in content:

            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            ):
                text_parts.append(
                    item.get("text", "")
                )

        return "\n".join(text_parts)


    return content


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    while True:

        question = input(
            "\nAsk a medical question "
            "(type 'exit' to quit): "
        )

        if question.lower().strip() == "exit":
            print("\nExiting...")
            break

        answer = get_answer(question)

        print("\nAnswer:")
        print(answer)