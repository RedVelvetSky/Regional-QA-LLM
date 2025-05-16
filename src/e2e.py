from registry import answering_methods, evaluation_methods, retrieval_methods
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
    with jsonlines.open(config["output_path"], mode="w") as writer:
        for item in reader:
            question = item[q_key]
            true_answer = item[a_key]

            context = retrieve_fn(question)
            pred_answer = answer_fn(question, context)
            evaluation = eval_fn(question, true_answer, pred_answer)

            writer.write(
                {
                    "question": question,
                    "true_answer": true_answer,
                    # "context": context,
                    "pred_answer": pred_answer,
                    "evaluation": evaluation,
                }
            )
