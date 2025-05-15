import functools


def rag_to_string(func):
    """
    Decorator to transform the output of answer_query into a single string.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result_dict = func(*args, **kwargs)

        local = result_dict.get("local")
        wiki_fallback = result_dict.get("wiki_fallback")

        if local:
            return "\n\n".join(local)

        if wiki_fallback:
            return "\n\n".join(wiki_fallback)

        return "no context"

    return wrapper
