# 🩺 Medical Chatbot — RAG-Based Medical Question Answering System

A Retrieval-Augmented Generation (RAG) based medical chatbot that allows users to ask questions about medical conditions, symptoms, diseases, and other information contained in the medical knowledge base.

The system retrieves relevant medical information from a Pinecone vector database and uses a Groq-powered Large Language Model to generate a concise, context-grounded response.

---

## 📌 Project Overview

The Medical Chatbot combines:

- Natural Language Processing
- Semantic Search
- Vector Embeddings
- Vector Database
- Retrieval-Augmented Generation (RAG)
- Large Language Models
- Flask REST API
- HTML/CSS/JavaScript frontend

Instead of directly asking an LLM to answer every question from its general knowledge, the chatbot first searches the medical knowledge base for relevant information.

The retrieved information is then provided to the LLM as context so that the generated response is grounded in the available medical documents.

---

# ✨ Features

### 🔎 Semantic Medical Search

User questions are converted into vector embeddings and searched against the medical knowledge stored in Pinecone.

### 🧠 Retrieval-Augmented Generation

The chatbot follows a RAG pipeline:

```text
User Question
      ↓
Question Embedding
      ↓
Pinecone Vector Search
      ↓
Relevant Medical Documents
      ↓
Context Construction
      ↓
Prompt
      ↓
Groq LLM
      ↓
Generated Answer
      ↓
User
```

### 💬 Interactive Web Interface

Users can enter medical questions directly through a browser-based interface.

### 🌐 Flask Backend

Flask provides the backend API responsible for:

- Receiving user questions
- Running the RAG pipeline
- Generating responses
- Returning the answer to the frontend

### 🛡️ Context-Grounded Responses

The chatbot is instructed to answer using the retrieved medical context.

If the available medical knowledge does not contain enough information, the chatbot can respond that the available context is insufficient instead of inventing an answer.

For example, an unrelated question such as:

```text
What is Python?
```

can result in:

```text
I don't have enough information in the medical knowledge base to answer that question.
```

### 🎨 Responsive Frontend

The frontend contains:

- Medical chatbot interface
- Question input area
- Ask Question button
- Answer section
- Loading/response handling
- Error handling

---

# 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │        User          │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ HTML/CSS/JavaScript  │
                         │    Web Interface     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Flask Backend     │
                         │       /ask API       │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    RAG Pipeline      │
                         └──────────┬───────────┘
                                    │
                       ┌────────────┴────────────┐
                       │                         │
                       ▼                         ▼
              ┌─────────────────┐       ┌─────────────────┐
              │ Hugging Face    │       │    Pinecone     │
              │   Embeddings    │──────▶│ Vector Database │
              └─────────────────┘       └────────┬────────┘
                                                  │
                                                  ▼
                                        ┌─────────────────┐
                                        │ Relevant Medical│
                                        │    Context      │
                                        └────────┬────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │    Groq LLM     │
                                        │                 │
                                        │ openai/gpt-     │
                                        │ oss-20b         │
                                        └────────┬────────┘
                                                 │
                                                 ▼
                                        ┌─────────────────┐
                                        │ Generated Answer│
                                        └────────┬────────┘
                                                 │
                                                 ▼
                                             User
```

---

# 🔄 RAG Pipeline

The application follows the following Retrieval-Augmented Generation workflow.

## Step 1 — Medical Documents

The project uses a medical knowledge source containing information about diseases, symptoms, conditions, diagnosis, and related medical topics.

---

## Step 2 — Document Processing

The medical documents are processed into smaller text chunks so that relevant information can be efficiently retrieved.

---

## Step 3 — Embeddings

Each document chunk is converted into a numerical vector representation using a Hugging Face embedding model.

```text
Medical Text
     ↓
Embedding Model
     ↓
Vector Representation
```

Embeddings allow the system to perform semantic similarity search.

---

## Step 4 — Pinecone Vector Database

The generated embeddings are stored in Pinecone.

Pinecone is used to efficiently search for documents that are semantically similar to the user's question.

---

## Step 5 — User Question

When a user asks a question:

```text
What are the symptoms of diabetes?
```

the question is converted into an embedding using the same embedding model.

---

## Step 6 — Similarity Search

The question embedding is sent to Pinecone.

The system retrieves the most relevant medical text chunks.

```text
User Question
      ↓
Question Embedding
      ↓
Pinecone Similarity Search
      ↓
Top Relevant Documents
```

---

## Step 7 — Context Construction

The retrieved documents are combined into a medical context.

The prompt sent to the LLM contains:

```text
Medical Context
+
User Question
+
Instructions
```

---

## Step 8 — Groq LLM

The constructed prompt is sent to the Groq-powered LLM.

Current model used:

```text
openai/gpt-oss-20b
```

The LLM generates an answer based on the retrieved context.

---

## Step 9 — Response

The generated answer is returned through the Flask API and displayed in the web interface.

---

# 🧰 Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | LLM/RAG integration |
| Groq | Large Language Model inference |
| Pinecone | Vector database |
| Hugging Face | Text embedding model |
| Flask | Backend REST API |
| HTML | Web page structure |
| CSS | User interface styling |
| JavaScript | Frontend interaction and API communication |
| Conda | Python environment management |
| Git | Version control |
| GitHub | Source code hosting |

---

# 📁 Project Structure

```text
medical-chatbot/
│
├── data/
│   └── Medical knowledge/data files
│
├── research/
│   └── Research and experimentation files
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── helper.py
│   ├── groq_test.py
│   ├── llm_test.py
│   ├── rag_test.py
│   ├── retrieval_test.py
│   └── store_index.py
│
├── static/
│   ├── script.js
│   └── style.css
│
├── templates/
│   └── index.html
│
├── .env
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

---

# 📄 Important Files

### `src/app.py`

Main Flask application.

Responsibilities include:

- Starting the Flask server
- Serving the frontend
- Receiving questions
- Calling the RAG pipeline
- Returning answers to the frontend

---

### `src/rag_test.py`

Contains the main RAG workflow.

It:

1. Converts the question into an embedding
2. Queries Pinecone
3. Retrieves relevant medical context
4. Builds the prompt
5. Sends the prompt to Groq
6. Extracts the generated answer

---

### `src/store_index.py`

Responsible for processing the medical documents and storing their vector representations in Pinecone.

---

### `src/retrieval_test.py`

Used for testing the retrieval stage independently.

---

### `src/groq_test.py`

Used to verify the Groq API integration independently.

---

### `src/helper.py`

Contains helper functions used by the application, including embedding-related functionality.

---

### `templates/index.html`

Main frontend HTML page.

---

### `static/style.css`

Contains the styling for the chatbot interface.

---

### `static/script.js`

Handles frontend interaction with the Flask API.

The JavaScript sends requests to:

```text
POST /ask
```

and displays the returned answer.

---

# 🔐 Environment Variables

The application uses environment variables to keep API credentials outside the source code.

Create a `.env` file:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=medical-chatbot
GROQ_API_KEY=your_groq_api_key
```

### ⚠️ Important

Never commit `.env` to GitHub.

Your `.gitignore` should contain:

```gitignore
.env
__pycache__/
*.pyc
.venv/
venv/
.vscode/
```

---

# 📝 `.env.example`

Create a `.env.example` file containing:

```env
PINECONE_API_KEY=
PINECONE_INDEX_NAME=medical-chatbot
GROQ_API_KEY=
```

This file can safely be committed because it does not contain actual credentials.

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/medical-chatbot.git
```

Move into the project:

```bash
cd medical-chatbot
```

---

## 2. Create the Conda Environment

```bash
conda create -n medical-chatbot python=3.10
```

Activate it:

```bash
conda activate medical-chatbot
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create:

```text
.env
```

and add:

```env
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=medical-chatbot
GROQ_API_KEY=your_groq_api_key
```

---

# 🗄️ Pinecone Setup

The application uses Pinecone to store and retrieve medical document embeddings.

The vector database contains the embedded medical knowledge used during question answering.

The retrieval process searches the Pinecone index for the most relevant chunks based on semantic similarity.

---

# 🧪 Testing

The project was tested at multiple levels.

## 1. LLM Test

```bash
python -m src.groq_test
```

This verifies that the Groq API is working correctly.

---

## 2. Retrieval Test

```bash
python -m src.retrieval_test
```

This verifies that relevant medical documents can be retrieved from Pinecone.

---

## 3. RAG Test

```bash
python -m src.rag_test
```

This tests the complete:

```text
Question
↓
Embedding
↓
Pinecone Retrieval
↓
Context
↓
Groq
↓
Answer
```

pipeline.

---

## 4. Flask Application Test

Start the application:

```bash
python -m src.app
```

The application runs locally at:

```text
http://127.0.0.1:5000
```

Open the address in a browser and interact with the chatbot.

---

# 🧪 Example Test Cases

### Medical Question

```text
What are the symptoms of diabetes?
```

The chatbot retrieves relevant medical information and generates a context-based response.

---

### Another Medical Question

```text
What is hypertension?
```

The system searches the medical knowledge base for relevant information.

---

### Detailed Question

```text
What are the complications of diabetes?
```

The RAG system retrieves relevant chunks and generates an answer from the available context.

---

### Unrelated Question

```text
What is Python?
```

The chatbot should not rely on the LLM's general knowledge.

Instead, when relevant information is unavailable in the medical knowledge base, it can respond that the available context is insufficient.

This helps reduce unsupported answers.

---

### Empty Input

Submitting an empty question is handled by the frontend rather than sending an unnecessary request to the backend.

---

# 🛡️ Medical Disclaimer

This project is an educational demonstration of Retrieval-Augmented Generation and should not be considered a medical diagnostic or treatment system.

The chatbot does not replace a qualified doctor or healthcare professional.

Users should consult qualified healthcare professionals for diagnosis, treatment, or urgent medical concerns.

---

# 🔒 Security Considerations

The following information should never be committed to GitHub:

- API keys
- Passwords
- `.env`
- Cloud credentials
- Private tokens
- Database credentials

Use environment variables instead.

---

# 📈 Future Improvements

Possible future improvements include:

- Docker containerization
- Production WSGI server
- Cloud deployment
- AWS deployment
- CI/CD using GitHub Actions
- Conversation history
- Streaming responses
- Improved retrieval ranking
- Metadata filtering
- Source citations in responses
- Authentication
- Rate limiting
- Automated evaluation of RAG responses
- Improved medical safety guardrails

---

# 🎯 Learning Outcomes

This project demonstrates practical understanding of:

- Retrieval-Augmented Generation
- Semantic search
- Text embeddings
- Vector databases
- Pinecone
- LangChain
- Large Language Models
- Groq API integration
- Prompt engineering
- Flask REST APIs
- Frontend/backend integration
- Environment variable management
- Git and GitHub

---

# 👩‍💻 Author

**Anusha Buchipalli**

Integrated M.Tech — Computer Science  
Vellore Institute of Technology, Vellore

---

# ⭐ Project Summary

This project demonstrates how a domain-specific medical chatbot can be built using a Retrieval-Augmented Generation architecture.

Instead of relying entirely on an LLM's general knowledge, the system retrieves relevant information from a medical vector database and provides that information as context to the LLM.

```text
Medical Documents
       ↓
Text Processing
       ↓
Embeddings
       ↓
Pinecone
       ↓
Semantic Retrieval
       ↓
Relevant Context
       ↓
Groq LLM
       ↓
Flask API
       ↓
Web Interface
       ↓
User
```

The project was developed and tested locally as an end-to-end RAG application.