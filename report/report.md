# Regional-Specific Question Answering

## Related Work

Large language models (LLMs) [5, 6, 7] have achieved a great success in the text generation tasks in recent years.
A classical approach is to train some parametric model on the training data only, however this has several drawbacks.
Although the language itself looks almost perfect,
the factual part of the answers often lags behind.
In these cases the model starts hallucinating, 
i.e. generating factual nonsense [8].
This can have several causes.
One obvious could be the obsolete training data.
In the scenario we described earlier the model's knowledge is limited by the time of the training.
This means that for recent events the model has to be retrained each time, which is costly and time consuming.
Other problem is that even though the information is in the training data, it is stored in the model's weights [1] and
the model does not always handle such information correctly during the inference.

To overcome this issue, several methods have been proposed.
The hybrid models combine parametric LLM with a database, 
which is being queried during inference.
That way the data information is separated from the LLM itself in a modular manner and the parametric model can concentrate purely on the language since the information will be taken elswhere.
Therefore the database can be updated without the need for any retraining or fine-tuning.

Lewis et al [1] proposed two ways of such hybrid model.
A single document is queried either for the full answer (RAG-Sequence Model) or for each token separately (RAG-Token Model).
Similarly to RAG-Token Model, Jiang et al [2] introduced a generic method called Forward-Looking Active REtrieval augmented generation (FLARE).

TODO: add evaluation metrics if not sufficient


## Method

In this project we applied RAG to a regional specific question answering - a standard question answering where the questions are specific for the locals in that region and might be ambiguous otherwise.

We downloaded the text and metadata from English and Czech Wikipedia.
The articles were split into smaller chunks and embedded using intfloat/multilingual-e5 model [13].
The embeddings and the texts were stored in a vector database.
TODO: go deeper into FAISS

During inference the RAG-Sequence Model is being used.
The question is embedded and the $K$ most relevant chunks of text according to cosine similarity is taken from the database.
Then a prompt is constructed based on the original question and the retrieved context.
The prompt is then passed to a multilingual LLM to generate the final answer.

The results were compared using BERTScore [9], LLM-as-a-judge [10] (using Phi-4-mini-instruct [3, 4]), chrF [11], ROUGE-L [12] metrics.

TODO: describe each metric

## Experiments

We tested our approach on a czech dataset consisting of 530 regional questions.
We also tried to use english instead of czech.
These experiments used the same set of questions translated to english and the RAG queried english wikipedia instead of the czech one.

For the answering LLM we used the Phi-4-mini-instruct model [3, 4].
In all experiments we used the same number of retrieved chunks of text $K = 10$.
We compared three settings to measure the impact of our RAG pipeline on the overall model's performance: 

- **No RAG**: no context was used

- **Perfect RAG**: perfect context from the data

- **RAG**: context retrieved by the RAG

Complete code is available in the [github repository](https://gitlab.mff.cuni.cz/chuto/npfl140/), all experiments are reproducible.


## Results

|              |BERTScore|LLM-as-a-judge|chrF   |ROUGE-L|
|--------------|:-------:|:------------:|:-----:|:-----:|
|No RAG        |0.8318   |0.3408        |14.0305|0.0868 |
|Perfect RAG   |0.8947   |0.9013        |53.0164|0.4837 |
|RAG           |0.8895   |0.8430        |49.8094|0.4521 |
||||||
|No RAG en     |0.8486   |0.3275        |20.6784|0.1763 |
|Perfect RAG en|0.8804   |0.6921        |42.0455|0.3962 |
|RAG en        |   |        || |

## Conclusion

## Contributions and Acknowledgements

- Vladyslav Furda – RAG: embedding and retrieving, managing Wikipedia files, spokesperson
- Tommy Chu – answering and evaluation, running experiments, Streamlit presentation
- Petr Kašpárek – coordination, slides, testing
- Vilém Pech – report


We are thankful to ÚFAL for the ÚFAL Grid Engine (LRC) which allowed us to run all our experiments.
Special gratitude goes to our supervisiors Jindřich Helcl, Jindřich Libovický for their support during the semester.

## Sources

- [1] **RAG**: https://arxiv.org/pdf/2005.11401
- [2] **FLARE**: https://aclanthology.org/2023.emnlp-main.495.pdf
- [3] **Phi-4-mini-instruct**:
@misc{microsoft_phi4miniinstruct_2025,
  title        = {{Phi-4-mini-instruct}},
  author       = {{Microsoft}},
  year         = {2025},
  howpublished = {\url{https://huggingface.co/microsoft/Phi-4-mini-instruct}},
  note         = {Accessed: 2025-05-17}
}
- [4] **Phi-4-mini**: https://arxiv.org/abs/2503.01743
- [5] **Attention Is All You Need**: https://arxiv.org/abs/1706.03762
- [6] **A Survey of Large Language Models**: https://arxiv.org/abs/2303.18223
- [7] **Emergent Abilities of Large Language Models**: https://arxiv.org/abs/2206.07682
- [8] **Hallucinations**: https://arxiv.org/abs/2005.00661
- [9] **BERTScore**: https://arxiv.org/abs/1904.09675
- [10] **LLM-as-a-judge**: https://arxiv.org/abs/2411.15594
- [11] **chrF**: https://arxiv.org/abs/2311.02692
- [12] **ROUGE-L**: https://scispace.com/papers/rouge-a-package-for-automatic-evaluation-of-summaries-2tymbd14i8
- [13] **Multilingual E5 Embeddings**: @article{wang2024multilingual,
  title={Multilingual E5 Text Embeddings: A Technical Report},
  author={Wang, Liang and Yang, Nan and Huang, Xiaolong and Yang, Linjun and Majumder, Rangan and Wei, Furu},
  journal={arXiv preprint arXiv:2402.05672},
  year={2024}
}


