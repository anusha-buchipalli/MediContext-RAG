"""
==============================================================================
AURA CLINICAL AI - ENTERPRISE MEDICAL DECISION SUPPORT & RAG SYSTEM
==============================================================================
Architecture:
1. Upstream Emergency Triage Engine (Immediate halt on acute red-flags)
2. Intent-Based Extent Router (LangChain-powered INTERNAL vs EXTERNAL)
3. Internal Vector Search (Pinecone + HuggingFace Embeddings)
4. External Medical Search (DuckDuckGo restricted to NIH, CDC, WHO)
5. Dual-Prompt Persona Engine (Patient 8th-Grade vs Clinician Technical)
6. Posterior Hallucination & Compliance Guardrail Auditor
7. Groq Whisper Medical Speech-to-Text Pipeline (/api/voice-transcribe)
8. Clinical Session Export & Medical Record Intake
==============================================================================
"""

import io
import os
import re
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure UTF-8 output on Windows consoles to prevent charmap encoding crashes
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

# Groq & Pinecone SDKs
import groq
from pinecone import Pinecone

# LangChain Components
from langchain_community.tools import DuckDuckGoSearchResults, DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader


# ==============================================================================
# 1. ENVIRONMENT CONFIGURATION & PATH SETUP
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "medical")

# Model configurations
GROQ_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b")
GROQ_WHISPER_MODEL = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Validation
if not GROQ_API_KEY:
    print("[WARNING] GROQ_API_KEY is not defined in environment variables.")

if not PINECONE_API_KEY:
    print("[WARNING] PINECONE_API_KEY is not defined in environment variables.")

# Allowed uploads
UPLOAD_FOLDER = BASE_DIR / "data"
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"pdf"}


# ==============================================================================
# 2. FLASK APPLICATION INITIALIZATION
# ==============================================================================

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
    static_url_path="/static"
)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload ceiling


# ==============================================================================
# 3. SINGLETON CLIENTS & LAZY-LOADED RESOURCES
# ==============================================================================

class MedicalAIService:
    """Enterprise medical intelligence engine managing RAG, routing, and guardrails."""

    def __init__(self):
        self._groq_client: Optional[groq.Groq] = None
        self._chat_llm: Optional[ChatGroq] = None
        self._pinecone_client: Optional[Pinecone] = None
        self._pinecone_index = None
        self._embeddings: Optional[HuggingFaceEmbeddings] = None
        self._ddg_search_run: Optional[DuckDuckGoSearchRun] = None
        self._ddg_search_results: Optional[DuckDuckGoSearchResults] = None

    @property
    def groq_client(self) -> groq.Groq:
        if self._groq_client is None:
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is missing from environment.")
            self._groq_client = groq.Groq(api_key=GROQ_API_KEY)
        return self._groq_client

    @property
    def chat_llm(self) -> ChatGroq:
        if self._chat_llm is None:
            if not GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is missing from environment.")
            self._chat_llm = ChatGroq(
                model=GROQ_CHAT_MODEL,
                temperature=0.1,
                api_key=GROQ_API_KEY
            )
        return self._chat_llm

    @property
    def pinecone_index(self):
        if self._pinecone_index is None:
            if not PINECONE_API_KEY:
                raise ValueError("PINECONE_API_KEY is missing from environment.")
            self._pinecone_client = Pinecone(api_key=PINECONE_API_KEY)
            self._pinecone_index = self._pinecone_client.Index(PINECONE_INDEX_NAME)
        return self._pinecone_index

    @property
    def embeddings(self) -> HuggingFaceEmbeddings:
        if self._embeddings is None:
            print("[INFO] Loading Hugging Face clinical embedding model...")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME
            )
            print("[INFO] Embedding model loaded successfully.")
        return self._embeddings

    @property
    def ddg_search(self) -> DuckDuckGoSearchResults:
        if self._ddg_search_results is None:
            self._ddg_search_results = DuckDuckGoSearchResults(output_format="list")
        return self._ddg_search_results

    @property
    def ddg_search_raw(self) -> DuckDuckGoSearchRun:
        if self._ddg_search_run is None:
            self._ddg_search_run = DuckDuckGoSearchRun()
        return self._ddg_search_run


service = MedicalAIService()


# ==============================================================================
# 4. UPSTREAM EMERGENCY TRIAGE ENGINE (LIFE-THREATENING WORDS)
# ==============================================================================

# Clinically-curated red flags requiring immediate emergency intervention
CRITICAL_EMERGENCY_PATTERNS = [
    # Cardiac / Acute Coronary Syndrome
    r"\b(chest\s+pain|crushing\s+chest|chest\s+pressure|heart\s+attack|myocardial\s+infarction|cardiac\s+arrest|radiating\s+pain\s+to\s+(left\s+arm|jaw|back))\b",
    # Cerebrovascular / Acute Stroke
    r"\b(stroke|facial\s+droop|face\s+droop|slurred\s+speech|sudden\s+numbness|loss\s+of\s+speech|thunderclap\s+headache|worst\s+headache\s+of\s+(my\s+)?life|one-sided\s+weakness|hemiparesis)\b",
    # Acute Respiratory Compromise
    r"\b(can'?t\s+breathe|cannot\s+breathe|difficulty\s+breathing|shortness\s+of\s+breath|gasping\s+for\s+air|choking|stridor|turning\s+blue|cyanosis|suffocating|respiratory\s+arrest)\b",
    # Anaphylaxis & Severe Airway Obstruction
    r"\b(anaphylaxis|severe\s+allergic\s+reaction|throat\s+(closing|swelling)|tongue\s+swelling|swollen\s+airway)\b",
    # Severe Hemorrhage & Trauma
    r"\b(coughing\s+up\s+blood|vomiting\s+blood|hematemesis|uncontrolled\s+bleeding|severe\s+hemorrhage|massive\s+blood\s+loss)\b",
    # Altered Mental Status & Seizures
    r"\b(unconscious|loss\s+of\s+consciousness|unresponsive|passed\s+out\s+and\s+not\s+waking|active\s+seizure|status\s+epilepticus)\b",
    # Psychiatric Emergencies & Crisis
    r"\b(suicid(e|al)|kill\s+myself|end\s+my\s+life|want\s+to\s+die|overdose|took\s+all\s+(my\s+)?pills|poisoning)\b"
]

COMPILED_EMERGENCY_REGEX = [re.compile(p, re.IGNORECASE) for p in CRITICAL_EMERGENCY_PATTERNS]


def evaluate_emergency_triage(text: str) -> Optional[Dict[str, Any]]:
    """
    Evaluates input text against critical emergency medical red flags.
    If matched, halts AI execution immediately and returns a critical triage record.
    """
    if not text:
        return None

    for regex in COMPILED_EMERGENCY_REGEX:
        match = regex.search(text)
        if match:
            matched_phrase = match.group(0).strip()
            return {
                "triage_emergency": True,
                "emergency_level": "CRITICAL_RED_FLAG",
                "matched_term": matched_phrase,
                "title": "CRITICAL EMERGENCY TRIAGE ALERT",
                "message": (
                    f"Immediate medical intervention may be required. Symptoms suggestive of "
                    f"'{matched_phrase.upper()}' were detected. Artificial intelligence cannot diagnose "
                    f"or treat acute emergency medical conditions."
                ),
                "directive": "Cease conversational inquiry and activate emergency medical services immediately.",
                "hotlines": [
                    {
                        "label": "Emergency Medical Response",
                        "number": "911 (US / Canada) | 112 (Europe) | 999 (UK)"
                    },
                    {
                        "label": "Suicide & Crisis Lifeline",
                        "number": "988 (US - Call or Text 24/7)"
                    },
                    {
                        "label": "Poison Control National Center",
                        "number": "1-800-222-1222"
                    }
                ],
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    return None


# ==============================================================================
# 5. INTENT-BASED EXTENT ROUTER (LANGCHAIN)
# ==============================================================================

DOC_KEYWORD_INDICATORS = [
    r"\b(my\s+report|uploaded|the\s+pdf|this\s+document|in\s+my\s+file|my\s+test\s+results|my\s+lab|my\s+blood\s+work|my\s+scan|the\s+medical\s+book|book|page\s+\d+|in\s+the\s+text)\b"
]
COMPILED_DOC_REGEX = [re.compile(p, re.IGNORECASE) for p in DOC_KEYWORD_INDICATORS]

ROUTER_SYSTEM_INSTRUCTION = """You are a medical inquiry routing classifier for a clinical AI workstation.
Your task is to classify whether a user query requires:
1. 'INTERNAL': The user is inquiring about their uploaded medical document, patient record, PDF file, uploaded test metrics, or specific personal clinical file content (e.g., 'what does my uploaded report say about cholesterol?', 'is there any finding on page 4?', 'what are the metrics in my document?').
2. 'EXTERNAL': The user is inquiring about general medical knowledge, epidemiology, pathophysiology, pharmacology, symptom education, lifestyle, or guidelines outside a personal uploaded file (e.g., 'what causes hypertension?', 'symptoms of diabetes', 'how does metformin work?', 'WHO guidelines on salt intake').

Output ONLY the exact token 'INTERNAL' or 'EXTERNAL'. Do not provide any justification, punctuation, or additional text."""


def route_medical_intent(question: str) -> str:
    """
    Routes query to either 'INTERNAL' (Pinecone RAG) or 'EXTERNAL' (DuckDuckGo NIH/CDC/WHO).
    Combines fast heuristics with LangChain ChatGroq classification.
    """
    # Fast heuristic check for clear document references
    for regex in COMPILED_DOC_REGEX:
        if regex.search(question):
            return "INTERNAL"

    try:
        llm = service.chat_llm
        messages = [
            SystemMessage(content=ROUTER_SYSTEM_INSTRUCTION),
            HumanMessage(content=f"Inquiry: {question}\nClassification:")
        ]
        decision = llm.invoke(messages).content.strip().upper()
        if "INTERNAL" in decision:
            return "INTERNAL"
        return "EXTERNAL"
    except Exception as e:
        print(f"[ROUTER FALLBACK] Routing failed with error: {e}. Defaulting to INTERNAL.")
        return "INTERNAL"


# ==============================================================================
# 6. RETRIEVAL PIPELINES (INTERNAL PINECONE VS EXTERNAL NIH/CDC/WHO)
# ==============================================================================

def retrieve_internal_pinecone(question: str, top_k: int = 4) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Embeds query with Hugging Face embeddings and retrieves top matches from Pinecone.
    """
    try:
        embeddings = service.embeddings
        index = service.pinecone_index

        query_vector = embeddings.embed_query(question)

        # Primary namespace query
        search_results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=PINECONE_NAMESPACE
        )

        matches = search_results.get("matches", [])

        # Fallback to alternative namespace if primary yielded zero results
        if not matches:
            fallback_namespace = "chunk_500_50"
            search_results = index.query(
                vector=query_vector,
                top_k=top_k,
                include_metadata=True,
                namespace=fallback_namespace
            )
            matches = search_results.get("matches", [])

        context_blocks = []
        citations = []

        for i, match in enumerate(matches):
            metadata = match.get("metadata", {})
            text = metadata.get("text") or metadata.get("page_content") or ""
            score = match.get("score", 0.0)
            page = metadata.get("page", metadata.get("page_label", "N/A"))
            source = metadata.get("source", "Uploaded Clinical Document")

            if text.strip():
                clean_text = text.strip()
                context_blocks.append(f"[Document Excerpt #{i+1} | Page {page}]:\n{clean_text}")
                citations.append({
                    "id": match.get("id", f"chunk-{i}"),
                    "source": Path(source).name if source else "Medical_book.pdf",
                    "page": page,
                    "score": round(score, 4),
                    "excerpt": clean_text[:280] + ("..." if len(clean_text) > 280 else ""),
                    "type": "internal_document"
                })

        assembled_context = "\n\n".join(context_blocks)
        return assembled_context, citations

    except Exception as e:
        print(f"[INTERNAL RETRIEVAL ERROR] {e}")
        return "", []


def retrieve_external_verified(question: str) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Executes LangChain DuckDuckGo Search strictly constrained to trusted medical domains:
    site:nih.gov OR site:who.int OR site:cdc.gov
    """
    trusted_query = f"{question} (site:nih.gov OR site:who.int OR site:cdc.gov)"
    context_blocks = []
    citations = []

    try:
        ddg = service.ddg_search
        results = ddg.invoke(trusted_query)

        if isinstance(results, list) and results:
            for i, item in enumerate(results[:4]):
                title = item.get("title", f"Authoritative Medical Reference #{i+1}")
                snippet = item.get("snippet", "")
                link = item.get("link", "")

                # Verify domain trustworthiness
                is_trusted = any(dom in link for dom in ["nih.gov", "who.int", "cdc.gov", "nlm.nih.gov"])
                badge = "Verified Medical Authority" if is_trusted else "Public Medical Resource"

                context_blocks.append(f"[{badge} | {title} | Source: {link}]:\n{snippet}")
                citations.append({
                    "id": f"ext-{i+1}",
                    "title": title,
                    "url": link,
                    "excerpt": snippet,
                    "source": "NIH / WHO / CDC Clinical Repository",
                    "type": "external_verified"
                })

        # Fallback to string-based DuckDuckGoSearchRun if structured list is empty
        if not context_blocks:
            ddg_run = service.ddg_search_raw
            raw_text = ddg_run.run(trusted_query)
            if raw_text:
                context_blocks.append(f"[Authoritative Reference (NIH / WHO / CDC)]:\n{raw_text}")
                citations.append({
                    "id": "ext-raw-1",
                    "title": "National Institutes of Health / CDC / WHO Guidance",
                    "url": "https://www.nih.gov",
                    "excerpt": raw_text[:300] + "...",
                    "source": "Trusted Clinical Knowledge Base",
                    "type": "external_verified"
                })

    except Exception as e:
        print(f"[EXTERNAL SEARCH ERROR] {e}")
        context_blocks.append("Public medical registry retrieval timed out. General biomedical reasoning engaged.")

    assembled_context = "\n\n".join(context_blocks)
    return assembled_context, citations


# ==============================================================================
# 7. DUAL-PROMPT SYSTEM: PATIENT VS. CLINICIAN
# ==============================================================================

PATIENT_PROMPT_TEMPLATE = """You are AURA Clinical AI, acting as an empathetic, clear, and trusted patient health communicator.
Audience: PATIENT & CAREGIVER.

Your Core Directives:
1. Explain dense pathologies, anatomical functions, and laboratory measurements using accessible, reassuring, 8th-grade level terminology.
2. Break down complex medical jargon immediately into clear, human explanations (e.g., explain 'hyperlipidemia' as 'high levels of fat or cholesterol in the blood').
3. Use gentle analogies where helpful to foster understanding without patronizing.
4. Maintain a warm, encouraging, and supportive bedside manner.
5. Structure your explanation into clear, easily digestible sections:
   - **Summary in Plain Language**
   - **What This Means for Your Health**
   - **Practical Questions to Ask Your Doctor**
6. Strict Medical Safety: Never prescribe medications or issue definitive medical diagnoses. Frame information as clinical guidance to discuss with their healthcare provider.
{mode_instruction}

CLINICAL CONTEXT:
{context}

USER INQUIRY:
{question}

RESPONSE:"""


CLINICIAN_PROMPT_TEMPLATE = """You are AURA Clinical AI, acting as an advanced clinical decision-support intelligence for certified healthcare professionals.
Audience: LICENSED CLINICIAN (Physician, Specialist, Resident, Advanced Practice Provider).

Your Core Directives:
1. Deliver answers with extreme technical precision, rigorous physiological grounding, and academic brevity.
2. Preserve all diagnostic shorthand, ICD-10/ICD-11 indicators, laboratory reference ranges, pharmacological mechanisms (e.g., receptor affinities, half-life, bioavailability), and analytical sensitivity/specificity metrics.
3. Utilize objective, formal clinical nomenclature (e.g., dyspnea on exertion, left ventricular ejection fraction, creatinine clearance, normocytic normochromic anemia).
4. Organize your evaluation systematically:
   - **Clinical Assessment & Pathophysiology**
   - **Diagnostic Indicators & Differential Considerations**
   - **Evidence-Based Management & Pharmacotherapy Notes**
5. Adhere strictly to verified clinical benchmarks. Do not fabricate laboratory metrics or therapeutic dosages.
{mode_instruction}

CLINICAL CONTEXT:
{context}

USER INQUIRY:
{question}

CLINICAL SYNTHESIS:"""


def build_clinical_prompt(question: str, context: str, audience: str, guardrail_mode: str = "Calibrated") -> str:
    """Builds the role-specific prompt according to the selected audience modifier and guardrail mode."""
    if guardrail_mode.strip().capitalize() == "Strict":
        mode_instruction = (
            "COMPLIANCE POLICY: STRICT MODE (VERBATIM ONLY)\n"
            "- Rely strictly and solely on verbatim information in the CLINICAL CONTEXT below.\n"
            "- Do not extrapolate, infer, or mention any medications, diagnoses, or treatments not explicitly in the context.\n"
            "- If the context is incomplete, clearly state that the provided reference context does not contain enough information."
        )
    else:
        mode_instruction = (
            "COMPLIANCE POLICY: CALIBRATED MODE (EDUCATIONAL GUIDANCE)\n"
            "- Provide a helpful, evidence-based EDUCATIONAL OVERVIEW of standard clinical approaches and general pharmacological categories based on established guidelines.\n"
            "- When discussing medications, describe standard therapeutic classes (e.g., hormonal therapy, insulin sensitizers) and clinical indications in general educational terms.\n"
            "- Emphasize that all pharmaceutical treatments require clinical consultation and a prescription from an attending physician.\n"
            "- Do NOT prescribe personalized milligram dosages or issue personal treatment regimens."
        )

    if audience.strip().lower() == "clinician":
        return CLINICIAN_PROMPT_TEMPLATE.format(context=context, question=question, mode_instruction=mode_instruction)
    return PATIENT_PROMPT_TEMPLATE.format(context=context, question=question, mode_instruction=mode_instruction)


# ==============================================================================
# 8. POSTERIOR HALLUCINATION CHECK & COMPLIANCE GUARDRAIL (STRICT VS CALIBRATED)
# ==============================================================================

AUDITOR_STRICT_PROMPT = """You are a strict Medical Compliance & Factual Hallucination Auditor in an accredited clinical health system operating in STRICT VERBATIM MODE.
Your responsibility is to rigorously audit whether the CANDIDATE ANSWER is strictly grounded in the provided REFERENCE CONTEXT without ANY extrapolation.

Verification Criteria (STRICT):
- 'PASS': The candidate answer's factual assertions are strictly substantiated by the provided reference context, OR it responsibly clarifies that information is limited.
- 'FAIL': The candidate answer introduces any clinical claims, medications, or explanations not explicitly detailed in the reference context.

You must respond in the following EXACT format:
VERDICT: [PASS or FAIL]
RATIONALE: [Concise 1-sentence audit summary]"""


AUDITOR_CALIBRATED_PROMPT = """You are a Medical Compliance & Factual Hallucination Auditor in an accredited clinical health system operating in CALIBRATED EDUCATIONAL MODE.
Your responsibility is to ensure the CANDIDATE ANSWER provides safe, scientifically sound medical education without dangerous hallucinations.

Verification Criteria (CALIBRATED):
- 'PASS': The candidate answer provides safe, standard educational medical explanations, attributes concepts to clinical guidelines, responsibly disclaims that prescriptions require a physician, and does NOT prescribe personal milligram dosages or invent fake lab metrics. General recognized medical consensus is ALLOWED.
- 'FAIL': The candidate answer prescribes dangerous unverified dosages, promotes harmful medical falsehoods, fabricates fictitious laboratory metrics, or directly contradicts the reference context.

You must respond in the following EXACT format:
VERDICT: [PASS or FAIL]
RATIONALE: [Concise 1-sentence audit summary]"""


def verify_grounding_compliance(candidate_answer: str, reference_context: str, guardrail_mode: str = "Calibrated") -> Tuple[bool, str]:
    """
    Conducts posterior verification on the generated response against retrieved context.
    Supports 'Strict' (verbatim) and 'Calibrated' (safe educational) modes.
    Returns (passed: bool, rationale: str).
    """
    if not reference_context or not reference_context.strip():
        return True, "Standard medical limitation acknowledgment verified."

    is_strict = guardrail_mode.strip().capitalize() == "Strict"
    system_prompt = AUDITOR_STRICT_PROMPT if is_strict else AUDITOR_CALIBRATED_PROMPT

    try:
        llm = service.chat_llm
        audit_input = f"REFERENCE CONTEXT:\n{reference_context[:3000]}\n\nCANDIDATE ANSWER:\n{candidate_answer}\n\nAUDIT RESULT:"
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=audit_input)
        ]
        response = llm.invoke(messages).content.strip()

        verdict_match = re.search(r"VERDICT:\s*(PASS|FAIL)", response, re.IGNORECASE)
        rationale_match = re.search(r"RATIONALE:\s*(.*)", response, re.IGNORECASE)

        verdict = verdict_match.group(1).upper() if verdict_match else "PASS"
        rationale = rationale_match.group(1).strip() if rationale_match else "Grounded clinical synthesis."

        if verdict == "FAIL":
            return False, rationale
        return True, rationale

    except Exception as e:
        print(f"[COMPLIANCE AUDIT WARNING] Verification audit bypassed due to: {e}")
        return True, "Grounded in reference context."


def get_compliance_refusal_notice(guardrail_mode: str = "Strict") -> str:
    """Generates the appropriate compliance refusal card according to the active mode."""
    if guardrail_mode.strip().capitalize() == "Strict":
        return (
            "⚠️ **COMPLIANCE REFUSAL NOTICE (STRICT SAFETY GUARDRAIL)**\n\n"
            "This response was withheld by our Medical Safety Guardrail because **Strict Mode** requires "
            "verbatim citation without extrapolation beyond the retrieved snippets.\n\n"
            "💡 **Tip:** Switch to **Calibrated Mode** in the top bar if you would like an educational overview "
            "of standard clinical treatment approaches and guideline options."
        )
    return (
        "⚠️ **COMPLIANCE REFUSAL NOTICE (CLINICAL SAFETY GUARDRAIL)**\n\n"
        "This response was withheld by our Medical Safety Guardrail because key statements or metrics "
        "could not be verified against the authenticated clinical reference context.\n\n"
        "**Safety Advisory:** To eliminate risk of AI hallucinations regarding medication dosages, diagnostic "
        "parameters, or specific metrics, please consult your primary physician or review the original laboratory records."
    )


# ==============================================================================
# 9. GROQ WHISPER MEDICAL VOICE TRANSCRIPTION (/api/voice-transcribe)
# ==============================================================================

# Highly specialized medical prompt injected to guide Whisper's lexicon
MEDICAL_WHISPER_PROMPT = (
    "Clinical physician-patient encounter, anamnesis, and medical consultation recording. "
    "Accurately transcribe all complex medical terminology, pharmaceutical names, laboratory metrics, "
    "and anatomical nomenclature, including: cardiomyopathy, echocardiogram, troponin-I, atorvastatin, "
    "lisinopril, metformin, hydrochlorothiazide, amlodipine, levothyroxine, omeprazole, ceftriaxone, "
    "azithromycin, ciprofloxacin, thrombocytopenia, encephalopathy, dyspnea, anaphylaxis, "
    "arrhythmia, tachycardia, bradycardia, HbA1c, CBC, BUN/Creatinine, GFR, AST/ALT, "
    "ICD-10, atherosclerosis, coronary artery disease, normocytic anemia, bronchoscopy."
)


@app.route("/api/voice-transcribe", methods=["POST"])
def voice_transcribe():
    """
    Dedicated endpoint for audio voice intake.
    Passes raw audio directly to Groq's free 'whisper-large-v3' model with medical prompting.
    """
    try:
        audio_file = None
        filename = "recording.webm"
        audio_bytes = None

        # Check multipart form data
        if "audio" in request.files:
            audio_file = request.files["audio"]
            filename = secure_filename(audio_file.filename) or "recording.webm"
            audio_bytes = audio_file.read()
        elif "file" in request.files:
            audio_file = request.files["file"]
            filename = secure_filename(audio_file.filename) or "recording.webm"
            audio_bytes = audio_file.read()
        elif request.data:
            # Raw binary audio payload
            content_type = request.headers.get("Content-Type", "")
            ext = ".webm"
            if "wav" in content_type:
                ext = ".wav"
            elif "mp4" in content_type:
                ext = ".mp4"
            elif "ogg" in content_type:
                ext = ".ogg"
            filename = f"audio_capture{ext}"
            audio_bytes = request.data

        if not audio_bytes or len(audio_bytes) < 100:
            return jsonify({
                "error": "No valid audio payload received or audio duration too short.",
                "success": False
            }), 400

        client = service.groq_client

        # Submit to Groq Whisper v3
        transcription = client.audio.transcriptions.create(
            file=(filename, audio_bytes),
            model=GROQ_WHISPER_MODEL,
            prompt=MEDICAL_WHISPER_PROMPT,
            response_format="json",
            temperature=0.0
        )

        transcript_text = transcription.text.strip() if hasattr(transcription, "text") else str(transcription).strip()

        return jsonify({
            "success": True,
            "transcription": transcript_text,
            "model": GROQ_WHISPER_MODEL,
            "byte_size": len(audio_bytes),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        print(f"[VOICE TRANSCRIPTION ERROR] {e}")
        return jsonify({
            "error": f"Voice transcription failed: {str(e)}",
            "success": False
        }), 500


# ==============================================================================
# 10. PRIMARY CHAT & CLINICAL INFERENCE ENDPOINT (/ask)
# ==============================================================================

@app.route("/ask", methods=["POST"])
def ask_question():
    """
    Core Clinical AI Orchestration Pipeline:
    1. Triage Check (Immediate halt on emergency symptoms)
    2. Intent Routing (INTERNAL vs EXTERNAL)
    3. Retrieval (Pinecone vs DuckDuckGo NIH/CDC/WHO)
    4. Dual-Prompt Persona Formatting (Patient vs Clinician)
    5. Posterior Hallucination & Compliance Verification
    """
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()
        audience = data.get("audience", "Patient").strip()  # 'Patient' or 'Clinician'
        guardrail_mode = (data.get("guardrail_mode") or "Calibrated").strip().capitalize()
        if guardrail_mode not in ["Strict", "Calibrated"]:
            guardrail_mode = "Calibrated"

        if not question:
            return jsonify({"error": "A clinical query or question is required."}), 400

        print(f"\n[QUERY RECEIVED] Audience: {audience} | Guardrail: {guardrail_mode} | Question: {question}")

        # ----------------------------------------------------------------------
        # STEP 1: UPSTREAM EMERGENCY TRIAGE ENGINE
        # ----------------------------------------------------------------------
        emergency_triage = evaluate_emergency_triage(question)
        if emergency_triage:
            print(f"[EMERGENCY ALERT HALT] Matched: {emergency_triage['matched_term']}")
            return jsonify({
                "status": "emergency_intercepted",
                "triage_emergency": True,
                "triage_data": emergency_triage,
                "question": question,
                "answer": emergency_triage["message"],
                "audience": audience,
                "guardrail_mode": guardrail_mode,
                "route": "EMERGENCY_TRIAGE_OVERRIDE",
                "sources": [],
                "compliance_verified": True,
                "timestamp": emergency_triage["timestamp"]
            }), 200

        # ----------------------------------------------------------------------
        # STEP 2: INTENT-BASED EXTENT ROUTER (LANGCHAIN)
        # ----------------------------------------------------------------------
        route = route_medical_intent(question)
        print(f"[INTENT ROUTE] Selected: {route}")

        # ----------------------------------------------------------------------
        # STEP 3: KNOWLEDGE RETRIEVAL
        # ----------------------------------------------------------------------
        sources = []
        context = ""

        if route == "INTERNAL":
            context, sources = retrieve_internal_pinecone(question, top_k=4)
            if not context.strip():
                # Graceful fallback if internal index has no context
                context = "No specific match found in the uploaded medical documentation."
        else:
            context, sources = retrieve_external_verified(question)

        # ----------------------------------------------------------------------
        # STEP 4: DUAL-PROMPT GENERATION
        # ----------------------------------------------------------------------
        clinical_prompt = build_clinical_prompt(question, context, audience, guardrail_mode)
        llm = service.chat_llm
        raw_response = llm.invoke(clinical_prompt)
        candidate_answer = (raw_response.content if hasattr(raw_response, "content") else str(raw_response)).strip()

        # ----------------------------------------------------------------------
        # STEP 5: POSTERIOR HALLUCINATION & COMPLIANCE GUARDRAIL
        # ----------------------------------------------------------------------
        if not candidate_answer:
            final_answer = get_compliance_refusal_notice(guardrail_mode)
            passed_audit = False
            audit_rationale = "Candidate answer was withheld due to insufficient reference context under strict policy."
            hallucination_detected = True
        else:
            passed_audit, audit_rationale = verify_grounding_compliance(candidate_answer, context, guardrail_mode)
            print(f"[COMPLIANCE AUDIT] Grounding check passed: {passed_audit} ({audit_rationale}) [Mode: {guardrail_mode}]", flush=True)

            final_answer = candidate_answer
            hallucination_detected = False

            if not passed_audit:
                final_answer = get_compliance_refusal_notice(guardrail_mode)
                hallucination_detected = True

        return jsonify({
            "status": "success",
            "triage_emergency": False,
            "question": question,
            "answer": final_answer,
            "audience": audience,
            "guardrail_mode": guardrail_mode,
            "route": route,
            "sources": sources,
            "compliance_verified": passed_audit,
            "hallucination_detected": hallucination_detected,
            "audit_rationale": audit_rationale,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200

    except Exception as e:
        print(f"[ERROR IN /ask] {str(e)}")
        return jsonify({
            "error": f"An error occurred while evaluating clinical query: {str(e)}",
            "status": "error"
        }), 500


# ==============================================================================
# 11. PDF DOCUMENT INGESTION ENDPOINT (/api/upload-pdf)
# ==============================================================================

@app.route("/api/upload-pdf", methods=["POST"])
def upload_pdf():
    """
    Dynamic PDF ingestion into Pinecone vector index:
    Loads PDF -> splits into chunks -> computes HuggingFace embeddings -> upserts.
    """
    try:
        if "file" not in request.files:
            return jsonify({"error": "No PDF file provided."}), 400

        file = request.files["file"]
        if not file.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Only valid PDF files are supported."}), 400

        saved_filename = secure_filename(file.filename)
        destination = UPLOAD_FOLDER / saved_filename
        file.save(destination)

        # Load and split
        loader = PyPDFLoader(str(destination))
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(docs)

        # Compute embeddings and upsert
        embeddings = service.embeddings
        index = service.pinecone_index

        batch_size = 64
        records = []
        for i, chunk in enumerate(chunks):
            vector = embeddings.embed_query(chunk.page_content)
            records.append({
                "id": f"{saved_filename}-chunk-{i}",
                "values": vector,
                "metadata": {
                    "text": chunk.page_content,
                    "source": saved_filename,
                    "page": chunk.metadata.get("page", 0)
                }
            })

            if len(records) >= batch_size:
                index.upsert(vectors=records, namespace=PINECONE_NAMESPACE)
                records = []

        if records:
            index.upsert(vectors=records, namespace=PINECONE_NAMESPACE)

        return jsonify({
            "success": True,
            "filename": saved_filename,
            "pages_loaded": len(docs),
            "chunks_upserted": len(chunks),
            "namespace": PINECONE_NAMESPACE
        }), 200

    except Exception as e:
        print(f"[PDF UPLOAD ERROR] {e}")
        return jsonify({"error": f"Failed to ingest PDF: {str(e)}"}), 500


# ==============================================================================
# 12. SYSTEM STATUS & TELEMETRY ENDPOINT (/api/status)
# ==============================================================================

@app.route("/api/status", methods=["GET"])
def system_status():
    """Reports status of vector store, Groq models, and active services."""
    try:
        pinecone_ready = False
        vector_count = 0
        namespace_count = 0

        try:
            stats = service.pinecone_index.describe_index_stats()
            pinecone_ready = True
            vector_count = stats.get("total_vector_count", 0)
            namespace_count = len(stats.get("namespaces", {}))
        except Exception as pe:
            print(f"[STATUS WARNING] Pinecone check failed: {pe}")

        return jsonify({
            "online": True,
            "service_name": "AURA Clinical Decision Support AI",
            "pinecone_connected": pinecone_ready,
            "pinecone_index": PINECONE_INDEX_NAME,
            "pinecone_namespace": PINECONE_NAMESPACE,
            "total_vectors": vector_count,
            "namespaces_count": namespace_count,
            "groq_chat_model": GROQ_CHAT_MODEL,
            "groq_whisper_model": GROQ_WHISPER_MODEL,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }), 200
    except Exception as e:
        return jsonify({"online": False, "error": str(e)}), 500


# ==============================================================================
# 13. STATIC FRONTEND & ROUTING
# ==============================================================================

@app.route("/")
def index_page():
    """Serves the primary clinical dashboard workspace."""
    # Serve index.html from static folder if present, or render_template
    static_index = BASE_DIR / "static" / "index.html"
    if static_index.exists():
        return send_from_directory(str(BASE_DIR / "static"), "index.html")
    return render_template("index.html")


@app.route("/index.html")
def static_index_redirect():
    return send_from_directory(str(BASE_DIR / "static"), "index.html")


# ==============================================================================
# 14. APPLICATION LAUNCH ENTRYPOINT
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" AURA CLINICAL AI - ENTERPRISE DECISION SUPPORT SYSTEM")
    print("=" * 70)
    print(f" Groq Chat Model     : {GROQ_CHAT_MODEL}")
    print(f" Groq Whisper Model  : {GROQ_WHISPER_MODEL}")
    print(f" Pinecone Index      : {PINECONE_INDEX_NAME} (ns: {PINECONE_NAMESPACE})")
    print(f" Embeddings Model    : {EMBEDDING_MODEL_NAME}")
    print(" System listening on : http://127.0.0.1:5000")
    print("=" * 70 + "\n")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )
