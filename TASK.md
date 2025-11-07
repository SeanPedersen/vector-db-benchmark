# Implement retrieval benchmark (implemented in retrieval_benchmark.py)

The goal of this benchmark to compare different vector storage and index configurations of JINA CLIP v2 text embeddings - to find a configuration that maximizes recall and minimizes storage and retrieval times.

Eval following knobs (vs bruteforce baseline via recall):
- Matryoshka dimensions (vector storage dimensions): 128, 512, 768, 1024
- Storage float precision (halfvector float16)

`CREATE TABLE items (id bigserial PRIMARY KEY, embedding halfvec(1024));`

- Index precision (full,  halfvector, binary with overfetching: fetch 10X results, then compute real distance)

`CREATE INDEX ON items USING hnsw ((embedding::bit(1024)) bit_cosine_ops);`
