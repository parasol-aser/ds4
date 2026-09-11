# Reproduction scripts

Paths are anchored on each script's own location: the repository root is two
levels up (`DS4_ROOT` overrides it), outputs go to `out/` next to the scripts
(`OUT` overrides it), and llama.cpp binaries are taken from `LLAMA_BIN`
(default `~/github/llama.cpp/build/bin`). Scripts stop on the first missing
prerequisite or failed command. Order: `make_prompts.py`, `run_q40.sh`,
`run_all.sh`, `run_correctness.sh`, `run_eval.sh`, then `analyze.py` and
`figures.py`.

| Script | Purpose |
| --- | --- |
| `make_prompts.py` | Builds the 1.4k--33k-token ChatML prompts from `prompt_1439.txt`. |
| `run_all.sh` | Headline runs with matched timing definitions (5 repeats), context sweep (3 repeats, three engines), ablation (3 repeats), stage profiles, perplexity, prefill, DeepSeek V4 Flash. |
| `run_q40.sh` | Builds the pure Q4_K and Q4_0 files with `llama-quantize` and prints their SHA-256. |
| `run_correctness.sh` | Reference-logit comparison at decode steps 0/4/16/63 on three prompts. |
| `run_eval.sh` | ds4-eval downstream check (10 questions, 2500-token budget). |
| `logits_dump.cpp` | llama.cpp reference: prefill, greedy decode, dump the logits of step k. |
| `cmp.py`, `top_logits.py` | Compare a ds4 logit dump with the reference. |
| `stage_agg.py` | Aggregate `DS4_METAL_DECODE_STAGE_PROFILE` + `DS4_METAL_CB_TIMES` output into per-stage GPU time. |
| `analyze.py`, `figures.py` | Tables and figures from the logs. |
