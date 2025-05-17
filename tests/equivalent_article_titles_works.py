import requests
import jsonlines

from src.retrieval.rag_pipeline import get_article_equivalent

def test():
    czech_data = list(jsonlines.open("data/CZ.dev.jsonl"))
    
    for item in czech_data:
        title = item["wikititle"]
        new_title = get_article_equivalent(title)
        print(title, new_title)

    # NO ERROR


if __name__ == "__main__":
    test()
