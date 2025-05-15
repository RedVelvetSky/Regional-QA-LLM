import functools


def rag_to_string(func):
    """
    Decorator to transform the output of answer_query into a single string.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result_dict = func(*args, **kwargs)

        local = result_dict["local"]
        wiki_fallback = result_dict["wiki_fallback"]

        print(wiki_fallback)
        if local:
            return local[0]["full_text"]

        if wiki_fallback:
            return wiki_fallback[0]["full_text"]

        return "no context"

    return wrapper
