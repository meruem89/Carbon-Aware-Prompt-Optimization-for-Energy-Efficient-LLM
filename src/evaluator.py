from bert_score import score


def evaluate_semantic_fidelity(reference_response: str, candidate_response: str) -> float:
    """
    Evaluate semantic fidelity between a reference response and a candidate response.
    Returns the rounded BERTScore F1 similarity.
    """
    _, _, f1_scores = score(
        [candidate_response],
        [reference_response],
        lang="en",
        rescale_with_baseline=True,
    )
    return round(f1_scores[0].item(), 4)


if __name__ == "__main__":
    reference = "Machine learning is a method that lets computers learn patterns from data."
    candidate = "Machine learning enables computers to discover patterns in data and improve from experience."

    fidelity = evaluate_semantic_fidelity(reference, candidate)
    print(f"Semantic fidelity score: {fidelity}")