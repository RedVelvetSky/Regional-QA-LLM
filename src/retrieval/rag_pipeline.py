import json
import os
import yaml
import re

import numpy as np
import requests
import faiss
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 0


def load_config(config_path=None):
    """
    Loads configuration parameters from a YAML file.

    If no path is provided, loads 'config.yaml' from the same directory as the current script.

    :param config_path: Path to the YAML config file. If None, uses the default location.
    :type config_path: str or None

    :return: Parsed configuration as a dictionary.
    :rtype: dict
    """
    if config_path is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_dir, "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

# === CONFIG ===
PASSAGE_FILE = CONFIG["PASSAGE_FILE"]
LANGUAGE = CONFIG["LANGUAGE"]
ARTICLE_JSONL = CONFIG["ARTICLE_JSONL"]

# === LOAD MODEL & TOKENIZER ===
MODEL_NAME = "intfloat/multilingual-e5-large"
_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
_model = AutoModel.from_pretrained(MODEL_NAME).eval()
_device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
_model.to(_device)


# === UTILS ===
def detect_language(text):
    """
    Detects language code of the input text using langdetect.
    Returns ISO 639-1 code: 'en', 'cs', etc.
    """
    try:
        lang = detect(text)
        return lang if lang in ("cs", "en") else "cs"
    except Exception:
        return "cs"


def _encode_query(query: str) -> np.ndarray:
    """
    Encodes a query string into a normalized embedding vector using the multilingual-e5 model.

    :param query: Input question or query string.
    :type query: str

    :return: Normalized embedding vector of shape [embedding_dim].
    :rtype: np.ndarray
    """

    inp = _tokenizer(
        f"query: {query}", return_tensors="pt", truncation=True, max_length=512
    ).to(_device)
    with torch.no_grad():
        out = _model(**inp).last_hidden_state  # (1, T, H)
    mask = inp["attention_mask"].unsqueeze(-1).bool()
    out = out.masked_fill(~mask, 0.0)
    pooled = (out.sum(1) / mask.sum(1)).cpu()
    return F.normalize(pooled, p=2, dim=1)[0].numpy()


def _load_faiss_index(jsonl_path: str):
    """
    Loads a FAISS index and corresponding passage embeddings from a .jsonl file.

    This function uses global variables to cache the FAISS index and passage metadata
    for future retrievals, enabling efficient nearest-neighbor search.

    :param jsonl_path: Path to the passage embeddings in .jsonl format.
    :type jsonl_path: str

    :return: None
    """

    # one-time global
    global _faiss_index, _passages
    if hasattr(_load_faiss_index, "_done"):
        return
    passages, vecs = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            passages.append(obj)
            vecs.append(obj["embedding"])
    mat = np.stack(vecs).astype("float32")
    faiss.normalize_L2(mat)
    _faiss_index = faiss.IndexFlatIP(mat.shape[1])
    _faiss_index.add(mat)
    _passages = passages
    _load_faiss_index._done = True


def _search_faiss(qvec: np.ndarray, top_k: int):
    """
    Performs a nearest-neighbor search in the FAISS index given a query embedding.

    :param qvec: Normalized embedding vector for the input query.
    :type qvec: np.ndarray

    :param top_k: Number of top passages to retrieve.
    :type top_k: int

    :return: List of top retrieved passages, each as a dictionary containing title, passage, and similarity score.
    :rtype: list of dict
    """

    _load_faiss_index(PASSAGE_FILE)
    q = qvec.astype("float32")[None, :]
    faiss.normalize_L2(q)
    scores, ids = _faiss_index.search(q, top_k)
    results = []
    for score, idx in zip(scores[0], ids[0]):
        item = _passages[idx]
        results.append(
            {"title": item["title"], "passage": item["passage"], "score": float(score)}
        )
    return results


# def _get_page_from_xml(xml_path: str, exact_title: str):
#     """
#     Retrieves the full text of a Wikipedia page by its exact title from a MediaWiki XML dump.
#
#     :param xml_path: Path to the Wikipedia XML dump file.
#     :type xml_path: str
#
#     :param exact_title: Exact title of the Wikipedia page to retrieve (case-sensitive).
#     :type exact_title: str
#
#     :return: Tuple (title, page_text) if the page is found, otherwise (None, None).
#     :rtype: tuple (str, str)
#     """
#
#     context = etree.iterparse(xml_path, events=("end",), tag="{*}page")
#     for _, elem in context:
#         title = elem.findtext("{*}title")
#         if title == exact_title:
#             txt = elem.findtext(".//{*}revision/{*}text") or ""
#             return title, txt
#         elem.clear()
#         while elem.getprevious() is not None:
#             del elem.getparent()[0]
#     return None, None


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


def _wiki_search_fallback(query: str, lang: str, top_k: int = 1):
    """
    Searches Wikipedia using the REST API for relevant page titles matching a query.

    :param query: Search query for Wikipedia.
    :type query: str

    :param lang: Wikipedia language code (e.g., 'cs' for Czech, 'en' for English).
    :type lang: str

    :param top_k: Number of top candidate titles to return.
    :type top_k: int

    :return: List of Wikipedia page titles matching the query.
    :rtype: list of str
    """

    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": top_k,
    }
    data = requests.get(url, params=params).json()
    return [r["title"] for r in data.get("query", {}).get("search", [])]


def get_online_wikipedia_content(title, lang="cs"):
    """
    Fetches the full wikitext content of a Wikipedia article by title from Wikipedia API.
    :param title: Wikipedia article title (string)
    :param lang: Language code (default: "cs" for Czech)
    :return: str (wikitext or empty string)
    """
    url = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "titles": title,
        "format": "json",
        "explaintext": True,  # Get plain text (no markup)
        "redirects": 1,
    }
    resp = requests.get(url, params=params).json()
    pages = resp.get("query", {}).get("pages", {})
    if not pages:
        return ""
    page = next(iter(pages.values()))
    return page.get("extract", "") or ""


def get_article_equivalent(
    wikititle: str, orig_lang: str = "cs", target_lang: str = "en"
):
    """
    Returns the equivalent article title in the target language.

    :param wikititle: Title of the article in the original language.
    :type wikititle: str

    :param orig_lang: Language code of the original language (e.g., "cs" for Czech).
    :type orig_lang: str

    :param target_lang: Language code of the target language (e.g., "en" for English).
    :type target_lang: str

    :return: Equivalent article title in the target language or None if not found.
    :rtype: str or None
    """
    url = f"https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbgetentities",
        "sites": orig_lang + "wiki",
        "titles": wikititle,
        "languages": target_lang,
        "format": "json",
    }
    resp = requests.get(url, params=params).json()
    try:
        new_title = list(resp["entities"].values())[0]["sitelinks"][
            target_lang + "wiki"
        ]["title"]
    except:
        return None
    return new_title


# === PUBLIC API ===
def answer_query(query: str, top_k_passages: int = 3) -> dict:
    lang = detect_language(query)
    qvec = _encode_query(query)
    top_hits = _search_faiss(qvec, top_k_passages)
    results = []
    fallback_candidates = []

    if lang == "en":
        for hit in top_hits:
            # We find an English equivalent using Wikidata API
            eng_title = get_article_equivalent(
                hit["title"], orig_lang="cs", target_lang="en"
            )
            if eng_title:
                # Fetch live English Wikipedia content
                full_text = get_online_wikipedia_content(eng_title, lang="en")
                if full_text:
                    results.append(
                        {
                            "title": eng_title,
                            "score": hit["score"],
                            "full_text": full_text,
                        }
                    )
        # Fallback search in English Wikipedia
        for title in _wiki_search_fallback(query, lang="en", top_k=top_k_passages):
            txt = get_online_wikipedia_content(title, lang="en")
            if txt:
                fallback_candidates.append({"title": title, "full_text": txt})

    else:
        for hit in top_hits:
            full_title, full_text = _get_article_from_jsonl(ARTICLE_JSONL, hit["title"])
            results.append(
                {
                    "title": full_title,
                    "passage": hit["passage"],
                    "score": hit["score"],
                    "full_text": full_text,
                }
            )
        for title in _wiki_search_fallback(query, lang="cs", top_k=top_k_passages):
            txt = get_online_wikipedia_content(title, lang="cs")
            if txt:
                fallback_candidates.append({"title": title, "full_text": txt})

    return {
        "query": query,
        "local": results,
        "wiki_fallback": fallback_candidates,
    }
