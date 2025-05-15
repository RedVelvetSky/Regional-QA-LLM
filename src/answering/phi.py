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
        {
            "role": "system",
            "content": (
                "You are an AI assistant. Your task is to answer the user's question. "
                "1. First, analyze the provided context. "
                "2. If the context is empty or does not contain the answer, make an educated guess based on your general knowledge. "
                "3. Provide the final answer prefixed with 'Final Answer: '. "
                "Example of a context-based response: \n"
                "Reasoning: The context states that the capital of France is Paris. \n"
                "Final Answer: Paris"
                "\n\n"
                "Example of a guess: \n"
                "Reasoning: The provided context is empty and does not mention the capital of Mexico. Based on my general knowledge, the capital of Mexico is Mexico City. \n"
                "Final Answer: Mexico City"
            ),
        },
        {"role": "user", "content": f"Context: ```{context}```"},
        {"role": "assistant", "content": "Understood. I will analyze the context, or make an educated guess if needed. I will explain my reasoning and then provide the final answer prefixed with 'Final Answer: ' on a new line. What is your question?"},
        {"role": "user", "content": f"My question is: ```{question}```"},
    ]

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    generation_args = {
        "max_new_tokens": 500,
        "return_full_text": False,
        "temperature": 0.0,
        "do_sample": False,
    }

    output = pipe(messages, **generation_args)
    result = output[0]["generated_text"]
    answer = result.split("Final Answer: ")[-1].strip()

    return answer
