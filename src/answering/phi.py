from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def handle(question: str, context: str) -> str:
    """
    Args:
        question: Question to be answered.
        context: Context to be used for answering the question.

    Returns:
        Answer to the question.
    """

    model_path = "microsoft/Phi-4-mini-instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant. Your task is to answer the provided question based on the given context. Respond with the answer only."},
        {"role": "user", "content": f"Here is the context: ```{context}```"},
        {"role": "assistant", "content": "Sure! What's the equation? I can help you answer it."},
        {"role": "user", "content": f"Based on the context, please answer the following question: ```{question}```"},
    ]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 20,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output = pipe(messages, **generation_args)
    result = output[0]["generated_text"]

    return result
