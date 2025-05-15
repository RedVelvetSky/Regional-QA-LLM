import answering.distilbert
import evaluation.method_a
import retrieval.rag_pipeline

retrieval_methods = {
    "Vlad's RAG": retrieval.rag_pipeline.answer_query,
}

answering_methods = {
    "DistilBERT base SQuAD": answering.distilbert.handle,
}

evaluation_methods = {
    "Evaluation Method A": evaluation.method_a.handle,
}
