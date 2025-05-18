#! /usr/bin/env python3

import jsonlines
from collections import defaultdict
files = [
    "outputs/no_rag.jsonl",
    "outputs/no_rag_en.jsonl",
    "outputs/perfect_rag-2.jsonl",
    "outputs/perfect_rag_en.jsonl",
    "outputs/yes_rag.jsonl",
]

LLM_JUDGE_THRESHOLD = 5

def parse_llm_judge(evaluation):
    if evaluation == -1:
        return 0
        
    return evaluation["score"] >= LLM_JUDGE_THRESHOLD

SCORE_FUNCTIONS = {
    "Phi4MiniLLMScore": lambda x: parse_llm_judge(x["Phi4MiniLLMScore"]),
    "BERTScore_F1": lambda x: x["BERTScore"]["F1"],
    "BERTScore_P": lambda x: x["BERTScore"]["P"],
    "BERTScore_R": lambda x: x["BERTScore"]["R"],
    "chrF": lambda x: x["chrF"]["score"],
    "rougeL_F1": lambda x: x["rougeL"]["rougeL"]["fmeasure"],
    "rougeL_P": lambda x: x["rougeL"]["rougeL"]["precision"],
    "rougeL_R": lambda x: x["rougeL"]["rougeL"]["recall"],
}


def compare_strange_results(results, expected_better, expected_worse):
    total = 0
    for better, worse in zip (results[expected_better], results[expected_worse]):
        better_score = parse_llm_judge(better["evaluation"]["Phi4MiniLLMScore"])
        worse_score = parse_llm_judge(worse["evaluation"]["Phi4MiniLLMScore"])
        if better_score != worse_score:
            total += 1

    print(f"Total: {total}")


def find_missing_context(results):
    czech_data = list(jsonlines.open("data/CZ.dev.jsonl"))

    for line, czech in zip(results["perfect_rag-2.jsonl"], czech_data):
        if not line["context"][0]:
            print(czech["wikititle"])


def has_context(line):
    context = line["context"]
    return (isinstance(context, str) and context != "") or \
           (isinstance(context, list) and all(isinstance(c, str) and c != "" for c in context))


def main():
    results = {}
    for file in files:
        results[file.split("/")[-1]] = list(jsonlines.open(file))

    # compare_strange_results(results, "no_rag_en.jsonl", "no_rag.jsonl")
    # find_missing_context(results)

    for file in results:
        print(file)
        lines = results[file]

        scores = defaultdict(int)
        count = 0
        for line, perfect in zip(lines, results["perfect_rag_en.jsonl"]):
            if not has_context(perfect):
                continue

            for score_name, score_func in SCORE_FUNCTIONS.items():
                scores[score_name] += score_func(line["evaluation"])
            count += 1

        for score_name, score in scores.items():
            print(f"{score_name}: {score / count:.3f}")
        print(f"Count: {count}")
        print()

if __name__ == "__main__":
    main()
