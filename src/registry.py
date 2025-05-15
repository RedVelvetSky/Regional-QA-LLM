import answering.method_a
import evaluation.method_a
import retrieval.rag_pipeline

retrieval_methods = {
    "Vlad's RAG": retrieval.rag_pipeline.answer_query,
}

answering_methods = {
    "QA Method A": answering.method_a.handle,
}

evaluation_methods = {
    "Evaluation Method A": evaluation.method_a.handle,
}
