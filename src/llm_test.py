import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not set in the .env file."
    )


llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    #temperature=0,
    google_api_key=GEMINI_API_KEY
)


question = "What is diabetes mellitus?"

response = llm.invoke(question)


print("Question:")
print(question)

print("\nLLM Response:")
print(response.content)