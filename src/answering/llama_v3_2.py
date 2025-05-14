import torch
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
        model_id="meta-llama/Llama-3.2-3B-Instruct",
        task="text-generation",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    messages = [
        {"role": "system", "content": "You are an AI assistant. Your task is to answer the provided question based on the given context. Respond with the answer only."},
        {"role": "user", "content": f"Here is the context: ```{context}``` \n\n Based on this context, please answer the following question: ```{question}```"},
    ]

    outputs = qa_pipeline(messages, max_new_tokens=256)
    result = outputs[0]["generated_text"][-1]

    return result
