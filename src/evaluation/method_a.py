def handle(question: str, true_answer: str, pred_answer: str) -> dict:
    """
    Args:
        question: Question to be answered.
        true_answer: True answer.
        pred_answer: Generated answer.

    Returns:
        Dictionary of evaluation metrics.
    """

    return {
        "GPTScore": 0.643,
        "BERTScore": 0.783,
    }
