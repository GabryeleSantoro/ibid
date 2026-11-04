# Hybrid Search

## Two kinds of retrieval

Dense and keyword retrieval fail in different ways, so it pays to run both. BM25 is the classic keyword ranking: it scores documents by how often the query's terms appear, weighted by how rare each term is across the corpus. BM25 is excellent for exact terms such as error codes, product names, function names and rare jargon, where embeddings tend to blur close variants together.

## Dense for paraphrase

Dense retrieval with embeddings handles paraphrase: the user asks in their own words and the document uses different ones. "How do I get my money back" finds a passage about the refund policy even though neither "money" nor "back" appears in it. Each method covers the other's blind spot.

## Fusing the rankings

Reciprocal rank fusion merges them. Each document gets a score of one divided by a constant plus its rank in each list, commonly with the constant at 60, and the scores are summed. Fusion uses only ranks, not raw scores, so there is no need to normalise BM25 scores against cosine similarities. A chunk ranked well by both methods rises to the top, and the fused list then goes to the reranker.
