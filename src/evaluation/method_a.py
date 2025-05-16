import re
from rouge_score import rouge_scorer
import evaluate

from bert_score import score
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def evaluate_rag_answers_bertscore(candidate_answer, reference_answer):
    model_type = "xlm-roberta-large"
    P, R, F1 = score([candidate_answer], [reference_answer], model_type=model_type, device="cuda")
    return {
        "P": P.item(),
        "R": R.item(),
        "F1": F1.item(),
    }


model_path = "microsoft/Phi-4-mini-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
generation_args = {
    "max_new_tokens": 200,
    "return_full_text": False,
    "temperature": 0.0,
    "do_sample": False,
}


def evaluate_with_llm_judge(question, reference_answer, predicted_answer):
    prompt = f"""
You are an impartial AI evaluator. Your task is to assess the quality of a Predicted Answer based on a given Question and a Reference Answer.

Consider the following:
Question: "{question}"
Reference Answer: "{reference_answer}"
Predicted Answer: "{predicted_answer}"

Please evaluate the Predicted Answer based on the following criteria:
1.  Factual Alignment: Does the Predicted Answer align factually with the Reference Answer? Consider if the core information is the same, even if phrased differently or if additional correct information is provided.
2.  Relevance: Is the Predicted Answer relevant and responsive to the Question?
3.  Completeness: Does the Predicted Answer sufficiently address the Question, considering the Reference Answer as a guide to what's important?

Evaluate answers based on semantic equivalence, not language. If answers in different languages convey the same meaning and accuracy, they should be scored equally.

Based on these criteria, provide an overall quality score from 1 to 5:
1: Very Poor - The answer is largely incorrect, irrelevant, or nonsensical.
2: Poor - The answer has significant inaccuracies or relevance issues.
3: Fair - The answer is partially correct and relevant but has notable flaws or omissions.
4: Good - The answer is mostly correct and relevant, with only minor issues.
5: Excellent - The answer is accurate, relevant, comprehensive, and well-written exact match.

Provide your evaluation as a JSON object with one key: "score" (an integer from 1 to 5).

Example #1:
```
{{"score": 1}} Explanation: The answer stated, "The capital of France is Berlin." This is factually incorrect and names the capital of a different country.
```

Example #2:
```
{{"score": 2}} Explanation: The answer stated, "France has many important cities like Lyon and Marseille, and one of them is the capital." This is vague, doesn't directly answer the question, and omits the actual capital.
```

Example #3:
```
{{"score": 3}} Explanation: The answer "As of my last update in 2023, there are two capital cities in France: Paris and Berlin." is partially incorrect. Only Paris is the capital of France.
```

Example #4:
```
{{"score": 4}} Explanation: The answer stated, "Pari." While this is a common colloquial name for the capital, it's not the complete and formal name, and the spelling is incorrect.
```

Example #5:
```
{{"score": 5}} Explanation: The answer "Paris" is accurate. It directly answers the question and is exacty matching the reference answer.
```

JSON Evaluation:
"""

    output = pipe(prompt, **generation_args)
    result = output[0]["generated_text"]
    pattern = r"\"score\":\s*(\d)"
    match = re.search(pattern, result)
    if not match:
        return -1
    return {
        "score": int(match.group(1)),
        "result": result,
    }


def evaluate_chrf(true_answer, pred_answer):
    chrf = evaluate.load("chrf")
    results = chrf.compute(predictions=[pred_answer], references=[true_answer])
    return results


def evaluate_rougeL(true_answer, pred_answer):
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(pred_answer, true_answer)
    s1 = scores["rouge1"]._asdict()
    s2 = scores["rougeL"]._asdict()
    return {
        "rouge1": s1,
        "rougeL": s2,
    }


def handle(question: str, true_answer: str, pred_answer: str) -> dict:
    """
    Args:
        question: Question to be answered.
        true_answer: True answer.
        pred_answer: Generated answer.

    Returns:
        Dictionary of evaluation metrics.
    """

    bertscore = evaluate_rag_answers_bertscore(pred_answer, true_answer)
    llmscore = evaluate_with_llm_judge(question, true_answer, pred_answer)
    chrfscore = evaluate_chrf(true_answer, pred_answer)
    rougelscore = evaluate_rougeL(true_answer, pred_answer)

    return {
        "BERTScore": bertscore,
        "Phi4MiniLLMScore": llmscore,
        "chrF": chrfscore,
        "rougeL": rougelscore,
    }
