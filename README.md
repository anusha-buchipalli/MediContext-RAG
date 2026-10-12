# 🩺 AURA Clinical AI — Enterprise Decision Support & Verified Medical RAG

An enterprise-grade Clinical Decision Support and Medical Retrieval-Augmented Generation (RAG) platform. AURA integrates localized vector search across patient clinical records, verified external medical search across trusted global health registries (NIH, WHO, CDC), real-time Groq Whisper speech transcription, role-based prompt calibration (Patient vs. Clinician), and multi-tiered compliance guardrails.

---

## 🏛️ System Architecture

```text
                                   ┌──────────────────────────────────────┐
                                   │      USER INQUIRY / AUDIO STREAM     │
                                   └──────────────────┬───────────────────┘
                                                      │
                                 ┌────────────────────▼────────────────────┐
                                 │   1. UPSTREAM EMERGENCY TRIAGE ENGINE   │
                                 │  (Chest pain, stroke, dyspnea, trauma)  │
                                 └──────────────┬───────────────────┬──────┘
                                                │                   │
                                         [RED FLAG MATCH]      [NO RED FLAG]
                                                │                   │
                              ┌─────────────────▼─────────┐         │
                              │  IMMEDIATE INFERENCE HALT │         │
                              │  High-Contrast Alert Card │         │
                              │  911 / 112 / 988 Hotlines │         │
                              └───────────────────────────┘         │
                                                                    │
                                 ┌──────────────────────────────────▼──────┐
                                 │   2. INTENT-BASED EXTENT ROUTER (LANG)  │
                                 │   Classifies: INTERNAL vs. EXTERNAL     │
                                 └──────────────┬───────────────────┬──────┘
                                                │                   │
                                         [INTERNAL RAG]      [EXTERNAL WEB]
                                                │                   │
                                 ┌──────────────▼─────┐   ┌─────────▼──────────────┐
                                 │ Pinecone Index     │   │ LangChain DuckDuckGo   │
                                 │ Namespace: medical │   │ Site filters: nih.gov, │
                                 │ 384-dim Embeddings │   │ who.int, cdc.gov       │
                                 └──────────────┬─────┘   └─────────┬──────────────┘
                                                │                   │
                                                └─────────┬─────────┘
                                                          │
                                 ┌────────────────────────▼────────────────┐
                                 │  3. PATIENT VS. CLINICIAN DUAL PROMPTS  │
                                 │  Patient: Empathetic, 8th-grade, plain  │
                                 │  Clinician: ICD-10, metrics, shorthand  │
                                 └────────────────────────┬────────────────┘
                                                          │
                                 ┌────────────────────────▼────────────────┐
                                 │  4. POSTERIOR HALLUCINATION AUDITOR     │
                                 │  Strict grounding check vs. context     │
                                 └──────────────┬───────────────────┬──────┘
                                                │                   │
                                          [AUDIT PASS]        [AUDIT FAIL]
                                                │                   │
                                 ┌──────────────▼─────┐   ┌─────────▼──────────────┐
                                 │ Authenticated      │   │ Controlled Compliance  │
                                 │ Clinical Response  │   │ Refusal Notice         │
                                 └────────────────────┘   └────────────────────────┘
```

---

## 🌟 Core Features

### 1. Intent-Based Extent Router (LangChain)
- Automatically distinguishes whether a query references specific patient documentation (`INTERNAL`) or general health, disease, and pharmacology questions (`EXTERNAL`).
- **Internal Pipeline:** Uses Hugging Face embeddings (`sentence-transformers/all-MiniLM-L6-v2`) to query your Pinecone vector database (`medical-chatbot`, namespace `medical`).
- **External Pipeline:** Uses LangChain's `DuckDuckGoSearchRun` restricted with strict trusted domain constraints: `site:nih.gov OR site:who.int OR site:cdc.gov`. This guarantees the Groq LLM is grounded in authentic clinical literature without hallucinating.

### 2. Groq Whisper Medical Speech Transcription
- Dedicated `/api/voice-transcribe` endpoint directly accepts WebAudio binary fragments from the browser.
- Submits audio to Groq's high-speed `whisper-large-v3` model.
- Uses an extensive medical lexicon prompt containing complex pharmacological names, diagnoses, and anatomical terms (*cardiomyopathy, atorvastatin, lisinopril, troponin-I, anaphylaxis, encephalopathy, thrombocytopenia, HbA1c, GFR, normocytic anemia, ICD-10*) to prevent transcription errors.

### 3. Patient vs. Clinician Dual-Prompt Calibration
- Real-time segmented toggle on the UI:
  - **Patient Mode:** Formulates responses using empathetic, jargon-free, 8th-grade level terminology with relatable analogies and practical questions for their doctor.
  - **Clinician Mode:** Delivers technical precision with diagnostic shorthand, ICD-10/ICD-11 indicators, physiological mechanisms, and laboratory reference ranges.

### 4. Compliance Guardrails & Triage Engine
- **Upstream Emergency Halt:** Curated regex screening for life-threatening emergencies (*crushing chest pain, stroke, acute dyspnea, anaphylaxis, severe hemorrhage, unresponsiveness, suicidal ideation*). If flagged, halts LLM generation immediately and renders a high-contrast emergency warning card with direct-dial hotlines (911 / 112 / 999, 988 Lifeline, Poison Control).
- **Posterior Hallucination Guardrail:** Evaluates the candidate answer against the retrieved reference context. If ungrounded clinical claims, fake dosages, or unsupported metrics are detected, the system intercepts the response and displays a Controlled Compliance Refusal Notice.

### 5. Obsidian-and-Slate Enterprise UI & Session Export
- High-end dark slate palette (`#070A0F`, `#0C121D`, `#152238`) with hairline glassmorphism borders and pulsing telemetry status rings.
- Audio recording dock with real-time waveform animation and recording duration timer.
- Dynamic sidebar showing the live routing pipeline, active Pinecone vector counts, and expandable source citations.
- **Export Clinical Consultation Log:** Compiles the consultation state array into a formatted clinical report (`.txt` and `.json`) complete with timestamps, triage audits, and source citations for physician handoff.

---

## 📁 Repository Structure

```
medical-chatbot/
├── app.py                       # Core Flask application, LangChain router & Groq endpoints
├── run_production.py            # Multi-threaded Waitress WSGI production runner
├── requirements.txt             # Modular categorized dependencies
├── .env.example                 # Configuration credentials template
├── .env                         # Local environment keys (ignored by git)
├── data/                        # Uploaded patient records & medical PDFs
├── src/
│   ├── app.py                   # Modular entrypoint redirecting to root app
│   └── helper.py                # Embedding & document helper routines
├── static/
│   ├── index.html               # Enterprise clinical workstation UI
│   ├── style.css                # Obsidian & slate dark theme stylesheet
│   └── script.js                # State management, WebAudio recording, and export handlers
└── templates/
    └── index.html               # Synchronized Flask template
```

---

## ⚙️ Installation & Setup

### 1. Environment Activation
Activate your conda or virtual environment:
```bash
conda activate medical-chatbot
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Copy `.env.example` to `.env` and provide your API keys:
```ini
GROQ_API_KEY=gsk_your_groq_api_key_here
PINECONE_API_KEY=pcsk_your_pinecone_api_key_here
PINECONE_INDEX_NAME=medical-chatbot
PINECONE_NAMESPACE=medical
GROQ_CHAT_MODEL=openai/gpt-oss-20b
GROQ_WHISPER_MODEL=whisper-large-v3
```

---

## 🚀 Running the Application

### Development Mode:
```bash
python app.py
```

### Production WSGI Mode (Waitress):
```bash
python run_production.py
```

Open your browser at **`http://127.0.0.1:5000`**.

---

## 🔌 API Endpoints Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/ask` | `POST` | Core clinical inquiry endpoint (Triage -> Router -> RAG -> Auditor) |
| `/api/voice-transcribe` | `POST` | Groq Whisper speech-to-text pipeline with medical lexicon |
| `/api/upload-pdf` | `POST` | Ingests a new clinical PDF document into Pinecone |
| `/api/status` | `GET` | Returns vector counts, connection status, and model info |

---

## ⚖️ Clinical Safety Disclaimer
*AURA Clinical AI is designed strictly for informational and clinical decision-support reference. It does not establish a physician-patient relationship, prescribe pharmaceuticals, or substitute for emergency medical services. In an acute medical emergency, call 911 (US/Canada), 112 (Europe), 999 (UK), or proceed to the nearest emergency department.*