import os

from flask import Flask, render_template, request, jsonify

from src.rag_test import get_answer


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TEMPLATE_DIR = os.path.join(
    BASE_DIR,
    "templates"
)

STATIC_DIR = os.path.join(
    BASE_DIR,
    "static"
)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(
    __name__,
    template_folder=TEMPLATE_DIR,
    static_folder=STATIC_DIR
)


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template("index.html")


# ============================================================
# ASK QUESTION
# ============================================================

@app.route("/ask", methods=["POST"])
def ask():

    try:

        data = request.get_json()

        if not data:

            return jsonify({
                "error": "No data received."
            }), 400


        question = data.get(
            "question",
            ""
        ).strip()


        if not question:

            return jsonify({
                "error": "Please enter a question."
            }), 400


        print(
            f"\nUser Question: {question}"
        )


        # Send question to RAG system
        answer = get_answer(question)


        print(
            f"Answer: {answer}"
        )


        return jsonify({

            "question": question,

            "answer": answer

        })


    except Exception as e:

        print(
            "Error:",
            str(e)
        )


        return jsonify({

            "error":
                "Something went wrong while processing your question."

        }), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    app.run(

        debug=True,

        host="127.0.0.1",

        port=5000

    )