# Implement adaptive binary quantization (in retrieval_benchmark.py)

Exploit information increase along vector dim of matryoshka embeddings: represent first dims with more bits than last dims:
- 0 - 128 dims: 128x16=2048bits - quasi uint16 binary mean threshold quantisation
- 128 - 256 dims: 128x8=1024bits - quasi uint8 binary mean threshold quantisation
- 256 - 512 dims: 256x4=1024bits - quasi uint4 binary mean threshold quantisation
- 512 - 1024 dims: 512x2=1024bits - binary mean threshold quantisation

Total vector size: 5120bits
