from src.retrieval.rag_pipeline import answer_query

result = answer_query("What is the name of the secretary of state of the United States originally from the Czech Republic?")
print("Local RAG answers:")
for item in result["local"]:
    print(f"Title: {item['title']} | Score: {item['score']:.3f}")
    snippet = item.get("passage") or item.get("full_text") or ""
    print(snippet[:200], "\n")

print("\nWikipedia Fallback:")
for item in result["wiki_fallback"]:
    print(f"Title: {item['title']}")
    print(item['full_text'][:200], "\n")
