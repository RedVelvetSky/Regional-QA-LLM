#! /usr/bin/env python3

import jsonlines
from collections import defaultdict
files = [
    "outputs/no_rag.jsonl",
    "outputs/perfect_rag.jsonl",
    "outputs/yes_rag.jsonl",
]

LLM_JUDGE_THRESHOLD = 5

def parse_llm_judge(evaluation):
    if evaluation == -1:
        return 0
        
    return evaluation["score"] >= LLM_JUDGE_THRESHOLD

SCORE_FUNCTIONS = {
    "BERTScore": lambda x: x["F1"],
    "Phi4MiniLLMScore": parse_llm_judge,
    "chrF": lambda x: x["score"],
    "rougeL": lambda x: x["rougeL"]["fmeasure"],
}


def compare_strange_results(results):
    for perfect, yes in zip (results["perfect_rag.jsonl"], results["yes_rag.jsonl"]):
        perfect_score = parse_llm_judge(perfect["evaluation"]["Phi4MiniLLMScore"])
        yes_score = parse_llm_judge(yes["evaluation"]["Phi4MiniLLMScore"])
        if perfect_score < yes_score:
            ...


def main():
    results = {}
    for file in files:
        results[file.split("/")[-1]] = list(jsonlines.open(file))

    for file in results:
        print(file)
        lines = results[file]

        scores = defaultdict(int)
        count = 0
        for line, perfect in zip(lines, results["perfect_rag.jsonl"]):
            if not perfect["context"][0]:
                continue

            for score_name, score_func in SCORE_FUNCTIONS.items():
                scores[score_name] += score_func(line["evaluation"][score_name])
            count += 1

        for score_name, score in scores.items():
            print(f"{score_name}: {score / count:.4f}")
        print(f"Count: {count}")
        print()

if __name__ == "__main__":
    main()
