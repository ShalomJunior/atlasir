# 0007: pinned E5 encoder and isolated optional dense profile

Date: 17 October 2026. Status: accepted for the synthetic Day 6 pilot.

Use E5-small-v2 at full model/tokenizer commit
`ffb93f3bd4047442299a41ebb6fa998a38507c52`. Follow the explicit Transformers
model-card path: query/passage prefixes, masked mean pooling, float32 normalized
vectors and a 512-token limit. The research method and later exact FAISS plan
remain E5 dense retrieval; this chooses the implementation API rather than a
new retriever. All preprocessing and token-limit choices enter resolved config.

Keep torch/Transformers/safetensors in a locked optional `dense` extra. Test it
inside an isolated Kaggle Python 3.11 environment. The existing shared local
environment is an optional CPU check profile with separately recorded versions;
its results do not establish that the new locked GPU environment works.

Do not use mixed precision/TF32 or implicit device fallback in the initial
pilot. Use eval/inference mode, deterministic torch algorithms and explicit CUDA
workspace configuration. Compare repeated encodings with rtol 1e-5 and atol 1e-6;
this is a within-environment check, not a cross-version/device bitwise guarantee.

Use schema 3 for small encoding pilots, preserving vectors, IDs, truncation
counts, model-file hashes and exact source/config/input/runtime. Continue to
verify schema-1 E00 and schema-2 BM25 runs. The pilot is synthetic and exploratory;
no quality or throughput benchmark is claimed. Weight files are revision/hash
identified but not duplicated in these small bundles; durable shared model
objects and embedding checkpoints belong to the next storage stage.
