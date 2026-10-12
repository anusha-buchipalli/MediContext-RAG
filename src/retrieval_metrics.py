# Phase 2.3
# Retrieval evaluation metrics:
# - Precision@5
# - Reciprocal Rank
# - Mean Reciprocal Rank (MRR)

evaluation_results = {
    "R01": {
        "question": "What are the symptoms of diabetes?",
        "relevance": [1, 1, 0, 1, 0]
    },
    "R02": {
        "question": "What is diabetes mellitus?",
        "relevance": [1, 1, 1, 0, 0]
    },
    "R03": {
        "question": "What are the complications of diabetes?",
        "relevance": [1, 0, 1, 0, 0]
    },
    "R04": {
        "question": "What is hypertension?",
        "relevance": [1, 0, 1, 0, 0]
    },
    "R05": {
        "question": "What is anemia?",
        "relevance": [1, 1, 1, 1, 1]
    },
    "R06": {
        "question": "What is insulin?",
        "relevance": [1, 1, 1, 1, 1]
    },
    "R07": {
        "question": "What is hyperglycemia?",
        "relevance": [1, 1, 0, 0, 0]
    },
    "R08": {
        "question": "What is hypoglycemia?",
        "relevance": [1, 1, 0, 0, 1]
    },
    "R09": {
        "question": "How is diabetes diagnosed?",
        "relevance": [1, 0, 1, 1, 0]
    },
    "R10": {
        "question": "What is the capital of France?",
        "relevance": [0, 0, 0, 0, 0]
    },
}


def precision_at_k(relevance, k):
    """
    Calculate Precision@K.

    Precision@K =
        number of relevant results in top K / K
    """
    top_k = relevance[:k]

    if not top_k:
        return 0.0

    return sum(top_k) / len(top_k)


def reciprocal_rank(relevance):
    """
    Find the rank of the first relevant result.

    Reciprocal Rank = 1 / rank

    If no relevant result exists, return 0.
    """
    for rank, is_relevant in enumerate(relevance, start=1):
        if is_relevant == 1:
            return 1 / rank

    return 0.0


def main():

    k = 5

    precision_scores = []
    reciprocal_rank_scores = []

    print("=" * 80)
    print("RETRIEVAL EVALUATION METRICS")
    print("=" * 80)

    for test_id, result in evaluation_results.items():

        relevance = result["relevance"]

        precision = precision_at_k(relevance, k)
        rr = reciprocal_rank(relevance)

        precision_scores.append(precision)
        reciprocal_rank_scores.append(rr)

        print()
        print(f"Test ID: {test_id}")
        print(f"Question: {result['question']}")
        print(f"Relevance: {relevance}")
        print(f"Precision@{k}: {precision:.2f}")
        print(f"Reciprocal Rank: {rr:.2f}")

    mean_precision = sum(precision_scores) / len(precision_scores)
    mrr = sum(reciprocal_rank_scores) / len(reciprocal_rank_scores)

    print()
    print("=" * 80)
    print("FINAL METRICS")
    print("=" * 80)

    print(f"Mean Precision@{k}: {mean_precision:.2f}")
    print(f"Mean Precision@{k}: {mean_precision * 100:.2f}%")

    print(f"MRR: {mrr:.2f}")

    print("=" * 80)


if __name__ == "__main__":
    main()