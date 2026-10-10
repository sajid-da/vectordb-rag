# Metrics

## Synthetic HNSW vs FAISS Vector Search Benchmark
Workload: 500 random 32-dimensional vectors, 25 query vectors.
Reproduce: `.venv/Scripts/python scripts/benchmark.py`.

- **Custom HNSW Index**:
  - Recall@10: **0.964** (96.4% top-10 retrieval accuracy against exact brute-force search).
  - Build Duration: 0.9945 s.
  - Query Batch Duration (25 queries): 0.0478 s (~1.91 ms per query).
- **FAISS Baseline (`IndexFlatIP`)**:
  - Build Duration: 0.000030 s.
  - Query Batch Duration (25 queries): 0.0011 s (~0.044 ms per query).
- **Comparative Note**: FAISS is ~43x faster due to C++ SIMD hardware acceleration. HNSW demonstrates custom graph-search algorithm implementation correctness.

## Quantitative RAG Retrieval & Abstention Evaluation
Dataset: `data/eval_cases.jsonl` (100 total cases: 80 answerable, 20 unanswerable).
Reproduce: `.venv/Scripts/python scripts/evaluate_rag.py`.

- **Citation Verbatim Validity**: **100.0%** (100% of generated citations match source text verbatim).
- **Answerable Response Accuracy**: **100.0%** (80/80 answerable cases received grounded context responses).
- **Retrieval Recall (Mock Character Frequency Model)**: Recall@1 = 0.05, Recall@3 = 0.15, Recall@5 = 0.25, MRR = 0.0917. (Evaluates retrieval evaluation seam; production retrieval quality requires a trained sentence transformer embedding model).
