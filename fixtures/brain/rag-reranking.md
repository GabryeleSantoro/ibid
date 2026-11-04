# Reranking Search Results

## A second opinion

First-stage retrieval is built for speed: it compares a query vector with millions of chunk vectors that were computed ahead of time. It is fast but coarse, because the query and the chunk never meet. A reranker adds a second, more careful pass over a short list.

## The cross-encoder

The reranker is a cross-encoder. Instead of embedding the question and each passage separately, it reads them together as a single input and outputs a relevance score. That lets it notice when a passage really answers the question, rather than merely sharing its topic. In our pipeline the cross-encoder rescores the top 40 candidates from hybrid search and keeps 6, the passages that go into the answer prompt.

## The trade-off

Reranking is slower but more precise. Every candidate needs its own forward pass through the model, so cost grows with the size of the candidate list, and it cannot be precomputed. That is why it only ever sees the top few dozen results, never the whole index. Raising the candidate count can recover a relevant passage that the first stage ranked low, at the price of more latency. A minimum score can drop weak passages entirely.
