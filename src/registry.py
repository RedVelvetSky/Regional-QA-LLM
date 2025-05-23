import answering.phi
import evaluation.method_a
import retrieval.rag_pipeline
import retrieval.rag_utils
import retrieval.no_rag

retrieval_methods = {
    "Vlad's RAG": retrieval.rag_utils.rag_to_string(retrieval.rag_pipeline.answer_query),
    "No RAG": retrieval.no_rag.answer_query,
}

answering_methods = {
    "Phi-4-mini-instruct": answering.phi.handle,
}

evaluation_methods = {
    "Method A": evaluation.method_a.handle,
}
