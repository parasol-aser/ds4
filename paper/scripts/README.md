# Reproduction scripts

Paths are anchored on each script's own location: the repository root is the
nearest ancestor directory containing `ds4.c` (`DS4_ROOT` overrides it),
outputs go to `out/` next to the scripts (`OUT` overrides it), and llama.cpp
binaries are taken from `LLAMA_BIN` (default `~/github/llama.cpp/build/bin`).
Scripts stop on the first missing prerequisite or failed command; an argmax
mismatch in the correctness suite is recorded, not fatal. Order:
`make_prompts.py`, `run_q40.sh`, `run_all.sh`, `run_correctness.sh`,
`run_eval.sh`, then `analyze.py`, `tables.py`, `figures.py`, and the LaTeX build
in the parent directory.

| Script | Purpose |
| --- | --- |
| `make_prompts.py` | Builds the 1.4k--33k-token ChatML prompts from `prompt_1439.txt`. |
| `run_all.sh` | Headline runs with matched timing definitions (5 repeats), context sweep (3 repeats, three engines, prompt files passed unchanged to all three), ablation (3 repeats), stage profiles, perplexity, prefill, DeepSeek V4 Flash. Writes `out/logs/prompt_sha256.txt` (SHA-256 of the prompts and the wikitext-2 input). |
| `run_q40.sh` | Builds the pure Q4_K and Q4_0 files with `llama-quantize` and prints their SHA-256. |
| `run_correctness.sh` | Reference-logit comparison at nine (prompt, step) points: steps 0/4/16/40/63 on the 20-token prompt, 0/4/63 on the 1,439-token prompt, 4 on the 8,590-token prompt. |
| `run_eval.sh` | ds4-eval downstream check (10 questions, 2500-token budget). |
| `logits_dump.cpp` | llama.cpp reference: prefill, greedy decode, dump the logits of step k. |
| `cmp.py`, `top_logits.py` | Compare a ds4 logit dump with the reference. |
| `stage_agg.py` | Aggregate `DS4_METAL_DECODE_STAGE_PROFILE` + `DS4_METAL_CB_TIMES` output into per-stage GPU time. |
| `analyze.py`, `tables.py`, `figures.py` | `out/results.json` from the logs; LaTeX table bodies in `../tables/`; figures in `../figs/`. |
