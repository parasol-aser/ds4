#!/bin/zsh
# Headline decode comparison (matched timing definitions, 5 repeats), context sweep (3 repeats),
# single-switch ablation (3 repeats), per-stage GPU profiles, wikitext-2 perplexity, DeepSeek V4 Flash.
# Paths are anchored on this script's directory; the repository root is two levels up.
set -eu
HERE=${0:A:h}
# repository root: nearest ancestor of this script that contains ds4.c (works for paper/scripts and misc/paper/scripts)
find_root() { d=$1; while [ "$d" != "/" ]; do [ -f "$d/ds4.c" ] && { echo "$d"; return; }; d=${d:h}; done; echo "$1"; }
DS4_ROOT=${DS4_ROOT:-$(find_root "$HERE")}; OUT=${OUT:-$HERE/out}; L=$OUT/logs; mkdir -p "$L"
LL=${LLAMA_BIN:-$HOME/github/llama.cpp/build/bin}
for x in "$DS4_ROOT/ds4" "$LL/llama-completion" "$LL/llama-perplexity" "$OUT/prompt_32k.txt" "$OUT/wikitext-2-raw/wiki.test.raw"; do
  [ -e "$x" ] || { echo "missing: $x (build ds4 and llama.cpp, run make_prompts.py, fetch wikitext-2)"; exit 1; }
done
Q4=$DS4_ROOT/gguf/Qwen3.8-27B-Q4_K.gguf; Q40=$DS4_ROOT/gguf/Qwen3.8-27B-Q4_0.gguf
UD=$DS4_ROOT/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf; Q8=$DS4_ROOT/gguf/Qwen3.8-27B-Q8_0.gguf
DS=$DS4_ROOT/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf
for x in $Q4 $Q40 $UD $Q8; do [ -f "$x" ] || { echo "missing model: $x"; exit 1; }; done
SHORT=$'<|im_start|>user\nWrite a short paragraph about the ocean.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
printf '%s' "$SHORT" > "$L/prompt_short.txt"
MLX=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('mlx-community/Qwen3.8-27B-4bit'))")
cd "$DS4_ROOT"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$L/progress.txt"; }
ctx() { case $1 in 1k) echo 2048;; 4k) echo 5120;; 8k) echo 9216;; 16k) echo 18432;; 32k) echo 34816;; esac; }

# 1. headline: ds4 greedy and sampled (DS4_TOKEN_TIMING gives the per-forward and per-token split),
#    llama.cpp greedy, sampled with ds4's sampler settings, and sampled with its own defaults.
for f in Q4 Q40 UD Q8; do
  eval m=\$$f
  for i in 1 2 3 4 5; do
    DS4_TOKEN_TIMING=1 ./ds4 -m $m -p "$SHORT" -n 64 --temp 0 -c 4096 > "$L/ds4_greedy_${f}_$i.out" 2> "$L/ds4_greedy_${f}_$i.log"
    DS4_TOKEN_TIMING=1 ./ds4 -m $m -p "$SHORT" -n 64 -c 4096 > "$L/ds4_sampled_${f}_$i.out" 2> "$L/ds4_sampled_${f}_$i.log"
    $LL/llama-completion -m $m -p "$SHORT" -n 64 --temp 0 -c 4096 -no-cnv --no-warmup > "$L/llama_greedy_${f}_$i.out" 2> "$L/llama_greedy_${f}_$i.log"
    $LL/llama-completion -m $m -p "$SHORT" -n 64 --temp 1 --top-p 0.95 --top-k 0 --min-p 0 -c 4096 -no-cnv --no-warmup > /dev/null 2> "$L/llama_sampled_matched_${f}_$i.log"
  done
  for i in 1 2 3; do
    $LL/llama-completion -m $m -p "$SHORT" -n 64 -c 4096 -no-cnv --no-warmup > /dev/null 2> "$L/llama_sampled_default_${f}_$i.log"
  done
  log "headline $f done"
done
for i in 1 2 3 4 5; do
  python3 -m mlx_lm generate --model "$MLX" --prompt "$SHORT" --max-tokens 64 --temp 0.0 --ignore-chat-template > "$L/mlx_short_$i.log" 2>&1
done
log "mlx short done"

# 2. context sweep, three repeats, all three engines on the same prompt files
for rep in 1 2 3; do
  for n in 1k 4k 8k 16k 32k; do
    ./ds4 -m $Q4 --prompt-file "$OUT/prompt_$n.txt" -n 64 --temp 0 -c $(ctx $n) > /dev/null 2> "$L/ds4_ctx_${n}_$rep.log"
    $LL/llama-completion -m $Q4 -f "$OUT/prompt_$n.txt" -n 64 --temp 0 -c $(ctx $n) -no-cnv --no-warmup > /dev/null 2> "$L/llama_ctx_${n}_$rep.log"
    python3 -m mlx_lm generate --model "$MLX" --prompt "$(cat "$OUT/prompt_$n.txt")" --max-tokens 64 --temp 0.0 --ignore-chat-template > "$L/mlx_ctx_${n}_$rep.log" 2>&1
  done
  log "context rep $rep done"
done

# 3. single-switch ablation, three repeats
abl() { name=$1; shift; for i in 1 2 3; do env "$@" ./ds4 -m $Q4 -p "$SHORT" -n 64 --temp 0 -c 4096 > /dev/null 2> "$L/abl_${name}_$i.log"; done; }
abl baseline X=1
abl no_fast_mv DS4_QWEN_DISABLE_FAST_MV=1
abl no_ksplit DS4_METAL_Q4K_KSPLIT_NSG=0
abl multi_nr0_2 DS4_METAL_Q4K_MULTI_NR0=2
abl no_multi_proj DS4_QWEN_DISABLE_MULTI_PROJ=1
abl fa_nwg32 DS4_METAL_QWEN_FA_NWG=32
abl no_fused_swiglu DS4_QWEN_DISABLE_FUSED_SWIGLU=1
abl no_fused_norm DS4_QWEN_DISABLE_FUSED_NORM=1
ALLOFF=(DS4_QWEN_DISABLE_FAST_MV=1 DS4_METAL_Q4K_KSPLIT_NSG=0 DS4_QWEN_DISABLE_MULTI_PROJ=1 DS4_METAL_QWEN_FA_NWG=32 DS4_QWEN_DISABLE_FUSED_SWIGLU=1 DS4_QWEN_DISABLE_FUSED_NORM=1)
abl all_off $ALLOFF
abl q4k_classic DS4_QWEN_DISABLE_FAST_MV=1 DS4_METAL_Q4K_KSPLIT_NSG=0 DS4_QWEN_DISABLE_MULTI_PROJ=1 DS4_QWEN_DISABLE_FUSED_SWIGLU=1
for i in 1 2 3; do DS4_QWEN_DISABLE_FAST_MV=1 DS4_QWEN_DISABLE_MULTI_PROJ=1 DS4_QWEN_DISABLE_FUSED_SWIGLU=1 ./ds4 -m $Q40 -p "$SHORT" -n 64 --temp 0 -c 4096 > /dev/null 2> "$L/abl_q40_classic_$i.log"; done
log "ablation done"

# 4. per-stage GPU profiles (positions 20-31 of the short prompt; 8590-8601 of the 8k prompt)
PROF=(DS4_METAL_CB_TIMES=1 DS4_METAL_DECODE_STAGE_PROFILE=1)
env $PROF ./ds4 -m $Q4 -p "$SHORT" -n 12 --temp 0 -c 4096 > /dev/null 2> "$L/stage_after.log"
env $ALLOFF $PROF ./ds4 -m $Q4 -p "$SHORT" -n 12 --temp 0 -c 4096 > /dev/null 2> "$L/stage_alloff.log"
env $PROF ./ds4 -m $Q4 --prompt-file "$OUT/prompt_8k.txt" -n 12 --temp 0 -c 9216 > /dev/null 2> "$L/stage_after_8k.log"
env $PROF ./ds4 -m $Q40 -p "$SHORT" -n 12 --temp 0 -c 4096 > /dev/null 2> "$L/stage_q40.log"
log "stage profiles done"

# 5. perplexity (llama-perplexity; per-chunk running values in the logs give the paired analysis)
for f in Q4 Q40 UD Q8; do
  eval m=\$$f
  $LL/llama-perplexity -m $m -f "$OUT/wikitext-2-raw/wiki.test.raw" -c 2048 --chunks 40 > "$L/ppl_$f.log" 2>&1
  $LL/llama-completion -m $m -f "$HERE/prompt_1439.txt" -n 8 --temp 0 -c 4096 -no-cnv --no-warmup > /dev/null 2> "$L/llama_prefill_$f.log"
  ./ds4 -m $m --prompt-file "$HERE/prompt_1439.txt" -n 8 --temp 0 -c 4096 > /dev/null 2> "$L/ds4_prefill_$f.log"
  log "ppl/prefill $f done"
done
MLX_MODEL="$MLX" python3 "$HERE/mlx_ppl.py" "$OUT/wikitext-2-raw/wiki.test.raw" 40 > "$L/mlx_ppl.log" 2>&1

# 6. DeepSeek V4 Flash (skipped if the file is absent)
if [ -f "$DS" ]; then
  ./ds4 -m "$DS" -p "Write a short paragraph about the ocean." -n 64 --temp 0 -c 4096 > /dev/null 2> "$L/ds4_dsv4.log"
  $LL/llama-completion -m "$DS" -p "Write a short paragraph about the ocean." -n 64 --temp 0 -c 4096 -no-cnv --no-warmup > /dev/null 2> "$L/llama_dsv4.log"
  env $PROF ./ds4 -m "$DS" -p "Write a short paragraph about the ocean." -n 12 --temp 0 -c 4096 > /dev/null 2> "$L/stage_dsv4.log"
fi
log "ALL DONE"
