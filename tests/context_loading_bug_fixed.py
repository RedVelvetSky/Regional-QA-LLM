import json
import jsonlines
import re

# Copied because of import/dependencies difficulties
def _get_article_from_jsonl(jsonl_path: str, title: str):
    """
    Retrieves the full text of an article by its exact title from a JSONL file.

    :param jsonl_path: Path to the JSONL file containing articles.
    :type jsonl_path: str
    :param title: Exact title of the article to retrieve.
    :type title: str
    :return: Tuple (title, content) if found, otherwise (None, None).
    :rtype: tuple (str, str)
    """

    title = re.sub(r"[^\w\d.-]", "_", title)
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj["title"] == title:
                return obj["title"], obj["content"]
    return None, None

def test():
    czech_data = list(jsonlines.open("data/CZ.dev.jsonl"))

    not_loaded = 0

    for item in czech_data:
        title = item["wikititle"]
        context = _get_article_from_jsonl("data/all_wiki_articles.jsonl", title)
        if context[0] is None:
            print(title)
            not_loaded += 1

    assert not_loaded == 0

if __name__ == "__main__":
    test()
