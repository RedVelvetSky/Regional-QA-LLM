# Regional-Specific Question Answering

## Related Work

Large language models (LLMs) [celý svět] have achieved a great success in the text generation tasks in recent years.
A classical approach is to train some parametric model on the training data only, however this has several drawbacks.
Although the language itself looks almost perfect,
the factual part of the answers often lags behind.
In these cases the model starts hallucinating, 
i.e. generating factual nonsense [někdo].
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


## Method

In this project we applied RAG to a regional specific question answering - a standard question answering where the questions are specific for the locals in that region and might be ambiguous otherwise.

We have downloaded the text and metadata from English and Czech Wikipedia.
The articles were split into smaller chunks and embeded using intfloat/multilingual-e5 model.
The embeddings and the texts were stored into a vector database.

During inference the RAG-Sequence Model is being used.
The question is embeded and the $K$ most relevant chunks of text is taken from the database.
Then a prompt is constructed based on the original question and the retrieved context.
The prompt is then passed to a multilingual LLM to generate the final answer.


## Experiments

## Results

## Conclusion

## Contributions

## Sources

- [1] RAG: https://arxiv.org/pdf/2005.11401
- [2] FLARE: https://aclanthology.org/2023.emnlp-main.495.pdf
