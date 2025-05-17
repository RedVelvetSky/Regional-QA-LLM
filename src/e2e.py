from registry import answering_methods, evaluation_methods, retrieval_methods
from retrieval.rag_pipeline import _get_article_from_jsonl, get_online_wikipedia_content
import jsonlines
import json
import sys
import os


config_file = sys.argv[1]
with open(config_file, "r") as f:
    config = json.load(f)

retrieve_fn = retrieval_methods[config["retrieval_method"]]
answer_fn = answering_methods[config["answering_method"]]
eval_fn = evaluation_methods[config["evaluation_method"]]

q_key = config["data"]["question_key"]
a_key = config["data"]["answer_key"]

os.makedirs(os.path.dirname(config["output_path"]), exist_ok=True)

with jsonlines.open(config["data"]["path"]) as reader:
    with jsonlines.open(config["output_path"], mode="w", flush=True) as writer:

        for item in reader:
            question = item[q_key]
            true_answer = item[a_key]
            true_article = item["wikititle"]

            if config["rag_type"] == "correct":
                if config.get("lang", "cs") == "en":
                    context = get_online_wikipedia_content(true_article, lang="en")
                else:
                    context = _get_article_from_jsonl("data/all_wiki_articles.jsonl", true_article)
            else:
                context = retrieve_fn(question)
            context = context[:16384]

            pred_answer = answer_fn(question, context)
            evaluation = eval_fn(question, true_answer, pred_answer)

            o = {
                "question": question,
                "true_answer": true_answer,
                "context": context,
                "pred_answer": pred_answer,
                "evaluation": evaluation,
            }

            print(o)
            writer.write(o)
