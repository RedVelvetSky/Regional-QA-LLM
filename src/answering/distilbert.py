from transformers import pipeline


def handle(question: str, context: str) -> str:
    """
    Args:
        question: Question to be answered.
        context: Context to be used for answering the question.

    Returns:
        Answer to the question.
    """

    qa_pipeline = pipeline(
        model_id="distilbert-base-cased-distilled-squad",
        task="question-answering",
        device_map="auto",
    )
    result = qa_pipeline(
        question=question,
        context=context,
    )
    return result["answer"]
