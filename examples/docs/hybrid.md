# Hybrid search

Hybrid search runs keyword search (BM25) and vector search side by side and merges the two
rankings, usually with reciprocal rank fusion. Keyword matching catches exact terms such as
error codes, while embeddings catch paraphrases.
