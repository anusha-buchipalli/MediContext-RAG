# ============================================================
# RAG RETRIEVAL EVALUATION
# Top-K Comparison
# Precision@K and MRR
# ============================================================


# ------------------------------------------------------------
# Evaluation questions
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Human-reviewed relevance labels
#
# These are the relevance labels for the original
# Top-5 retrieved results.
#
# 1 = relevant
# 0 = not relevant
# ------------------------------------------------------------

relevance_labels = {

    "R01": [1, 0, 0, 1, 0],

    "R02": [1, 1, 1, 0, 0],

    "R03": [1, 0, 1, 0, 0],

    "R04": [1, 1, 1, 0, 0],

    "R05": [1, 0, 0, 1, 0],

    "R06": [1, 1, 1, 0, 0],

    "R07": [1, 0, 0, 0, 0],

    "R08": [1, 0, 0, 0, 1],

    "R09": [0, 0, 0, 1, 0],

    "R10": [0, 0, 0, 0, 0],
}


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def precision_at_k(relevance, k):
    """
    Calculate Precision@K.

    Precision@K =
        relevant results in top K
        -------------------------
                  K
    """

    top_k = relevance[:k]

    return sum(top_k) / k


def reciprocal_rank(relevance):
    """
    Calculate Reciprocal Rank.

    If the first relevant result is at rank r:

        RR = 1 / r

    If no relevant result exists:

        RR = 0
    """

    for rank, relevant in enumerate(relevance, start=1):

        if relevant == 1:
            return 1 / rank

    return 0


# ------------------------------------------------------------
# Run evaluation for one K
# ------------------------------------------------------------

def evaluate_at_k(k):

    precision_scores = []
    reciprocal_rank_scores = []

    print()
    print("=" * 80)
    print(f"RESULTS FOR TOP-K = {k}")
    print("=" * 80)

    for index, question in enumerate(
        evaluation_questions,
        start=1
    ):

        question_id = f"R{index:02d}"

        relevance = relevance_labels[question_id]

        precision = precision_at_k(
            relevance,
            k
        )

        rr = reciprocal_rank(relevance)

        precision_scores.append(precision)

        reciprocal_rank_scores.append(rr)

        print(
            f"{question_id}: "
            f"Precision@{k} = {precision * 100:.2f}%"
        )

        print(
            f"    Reciprocal Rank = {rr:.4f}"
        )

    mean_precision = (
        sum(precision_scores)
        / len(precision_scores)
    )

    mrr = (
        sum(reciprocal_rank_scores)
        / len(reciprocal_rank_scores)
    )

    print()
    print(
        f"Mean Precision@{k}: "
        f"{mean_precision * 100:.2f}%"
    )

    print(
        f"MRR: "
        f"{mrr:.4f}"
    )

    return mean_precision, mrr


# ============================================================
# MAIN
# ============================================================

print("=" * 80)
print("TOP-K RETRIEVAL COMPARISON")
print("=" * 80)

print()
print("Testing Top-K values: 1, 3, 5")


results = {}


for k in [1, 3, 5]:

    precision, mrr = evaluate_at_k(k)

    results[k] = {
        "precision": precision,
        "mrr": mrr
    }


# ============================================================
# FINAL COMPARISON
# ============================================================

print()
print("=" * 80)
print("FINAL TOP-K COMPARISON")
print("=" * 80)

print()

print(
    f"{'Top-K':<10}"
    f"{'Mean Precision':<20}"
    f"{'MRR':<10}"
)

print("-" * 40)


for k in [1, 3, 5]:

    precision = results[k]["precision"]

    mrr = results[k]["mrr"]

    print(
        f"{k:<10}"
        f"{precision * 100:<20.2f}"
        f"{mrr:<10.4f}"
    )

print()
print("=" * 80)