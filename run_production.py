"""
==============================================================================
AURA CLINICAL AI - PRODUCTION MULTI-THREADED WSGI RUNNER
==============================================================================
Production deployment server powered by Waitress.
Supports concurrent client queries, high-throughput voice transcription,
and scalable medical RAG retrieval without development server warnings.
==============================================================================
"""

import os
from dotenv import load_dotenv
from waitress import serve
from app import app

load_dotenv()

HOST = os.getenv("FLASK_HOST", "127.0.0.1")
PORT = int(os.getenv("FLASK_PORT", 5000))
THREADS = int(os.getenv("WSGI_THREADS", 6))

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print(" AURA CLINICAL AI - PRODUCTION WSGI SERVER (WAITRESS)")
    print("=" * 70)
    print(f" Host: http://{HOST}:{PORT}")
    print(f" Multi-threading: {THREADS} concurrent worker threads active")
    print(" Press Ctrl+C to terminate.")
    print("=" * 70 + "\n")

    serve(
        app,
        host=HOST,
        port=PORT,
        threads=THREADS,
        channel_timeout=120
    )
