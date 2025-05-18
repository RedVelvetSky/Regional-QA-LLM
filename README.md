# 🌍 Regional RAG-based QA over Wikipedia

This project for NPFL140 implements a **Retrieval-Augmented Generation (RAG)** pipeline for answering questions based on **English and Czech Wikipedia** content mostly. It supports both Czech and English inputs and outputs, using **multilingual embeddings** and **large language models (LLMs)**.

---

## 🧠 Overview

The system is composed of the following key parts:

1. **Wikipedia Dump Preprocessing**  
   Downloads and processes Wikipedia XML dumps for English and Czech.

2. **Text Chunking**  
   Splits articles into smaller, searchable text chunks (e.g., paragraphs or sections) and stores associated metadata (e.g., title, language).

3. **Embedding & Indexing**  
   Uses a multilingual embedding model (e.g., `intfloat/multilingual-e5`) to embed chunks. Stores both embeddings and raw text in a vector database (e.g., **Qdrant**).

4. **Query & Retrieval**  
   Embeds user queries and retrieves top-K relevant chunks from the vector store using semantic search.

5. **Prompt Construction**  
   Constructs prompts by combining the original question with retrieved context

6. **Processing Query**     
Query is being processed by LLM and output to user-friendly GUI

7. **Evaluation**     
Answers are compared to the golden ones and evaluated using different suitable metrics like `BERTScore`,
   `Phi4MiniLLMScore`, `chrF` and `rougeL`.

---

## 🔧 Setup

```bash
pip install -r requirements.txt
```

```bash
streamlit run src/app.py
```

---

## 📊 Dataset Format

Example item in `data/queries/cz.dev.jsonl`:

```aiignore
{
  "question_orig": "Kdy vzniklo Československo?",
  "answer_orig": "V roce 1918.",
  "question_en": "When was Czechoslovakia founded?",
  "answer_en": "In 1918.",
  "wikititles": ["Československo"],
  "generated_orig": [],
  "generated_en": []
}
```

---

## 📌 TODO

- [x] Select models, download Wiki dataset, and configure storage place
- [x] Preprocess Wiki database 
- [x] Implement RAG pipeline
- [x] Implement LLM answering to the queries
- [ ] Fine-tune multilingual LLM on regional datasets
- [x] Add a friendly interface for querying
- [x] Add benchmarks on retrieval accuracy and answer quality
- [x] Support for an English version of the questions and English RAG
- [ ] Support additional languages (e.g., SK, UA)

---

## 📜 License

MIT License. See `LICENSE` file for details.

---

## 🤝 Contributions

Feel free to open issues or submit pull requests! All contributions to improving multilingual support, model performance, and usability are welcome.
