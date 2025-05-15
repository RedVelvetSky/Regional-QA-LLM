from src.retrieval.rag_pipeline import answer_query

result = answer_query("Které řeky protékají Brnem?")
print("Local RAG answers:")
for item in result["local"]:
    print(f"Title: {item['title']} | Score: {item['score']:.3f}")
    print(item['passage'][:200], "\n")

print("\nWikipedia Fallback:")
for item in result["wiki_fallback"]:
    print(f"Title: {item['title']}")
    print(item['full_text'][:200], "\n")
