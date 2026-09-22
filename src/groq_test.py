import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

print("GROQ_API_KEY found:", bool(os.getenv("GROQ_API_KEY")))

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

response = llm.invoke("What is diabetes mellitus?")

print("\nGroq Answer:\n")
print(response.content)