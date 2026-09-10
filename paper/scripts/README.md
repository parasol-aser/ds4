# Reproduction scripts

All scripts take the repository root and an output directory from the
environment (`DS4_ROOT`, default `..`; `OUT`, default `./out`). They assume the
GGUF files under `$DS4_ROOT/gguf/` and a llama.cpp build at `$LLAMA_BIN`
(default `~/github/llama.cpp/build/bin`). Run them from this directory.

| Script | Purpose |
| --- | --- |
| `make_prompts.py` | Builds the 1.4k--33k-token ChatML prompts from `prompt_1439.txt`. |
| `run_all.sh` | Context sweep (ds4, llama.cpp), single-switch ablation, stage profiles, sampled decode, wikitext-2 perplexity, DeepSeek V4 Flash. |
| `run_q40.sh` | Pure Q4_0 build with `llama-quantize`, its benchmarks, logits and perplexity. |
| `run_mlx.sh` | MLX baseline: download of the community 4-bit conversion, context sweep, perplexity (`mlx_ppl.py`). |
| `run_eval.sh` | ds4-eval downstream check (10 questions, 2500-token budget). |
| `logits_dump.cpp` | llama.cpp reference: prefill, greedy decode, dump the logits of step k. |
| `cmp.py`, `top_logits.py` | Compare a ds4 logit dump with the reference. |
| `stage_agg.py` | Aggregate `DS4_METAL_DECODE_STAGE_PROFILE` + `DS4_METAL_CB_TIMES` output into per-stage GPU time. |
| `analyze.py`, `figures.py` | Tables and figures from the logs. |
