# Metrics

No baseline quality number has been recorded: the V3 set is self-made synthetic
data and is not yet connected to a real embedding provider. Reproduce metric
calculation with `PYTHONPATH=src py -3.14 -m pytest tests/test_evaluation.py`.

V7 self-made benchmark (500 random 32-D vectors, 25 queries): HNSW recall@10
0.964; build 0.9434542 s; query batch 0.0449936 s. FAISS build 0.0000304 s;
query batch 0.0012869 s. Reproduce with `.venv\\Scripts\\python.exe scripts\\benchmark.py`.
