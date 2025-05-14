import answering.distilbert
import answering.llama_v3_2
import evaluation.method_a
import retrieval.rag_pipeline

retrieval_methods = {
    "Retrieval Method A": retrieval.method_a.handle,
}

answering_methods = {
    "DistilBERT base SQuAD": answering.distilbert.handle,
    "Llama 3.2 3B Instruct": answering.llama_v3_2.handle,
}

evaluation_methods = {
    "Evaluation Method A": evaluation.method_a.handle,
}
