# Embeddings for Search

## Meaning as vectors

An embedding model turns a piece of text into a list of numbers, a dense vector of a few hundred dimensions. Dense vectors capture meaning rather than exact words: "how do I reset my password" and "I forgot my login" land close together even though they share no keywords. Search then becomes geometry. The question is embedded, and the chunks whose vectors sit nearest to it, usually by cosine similarity, are retrieved.

## One embedder for both sides

Use one embedder for queries and documents. Each model builds its own vector space, and vectors from two different models cannot be compared at all; the distances are meaningless. Some models also expect a short prefix that marks a text as a query or as a document, and leaving it out quietly hurts quality. Keep the model name and dimension stored next to the index.

## Re-index when it changes

Re-index when the embedder changes. A new model, or even a new version of the same one, means every stored chunk vector belongs to the old space. Mixing them gives nonsense results without any error message. The safe approach is to record which model built the index and refuse to search until everything has been embedded again with the new one.
