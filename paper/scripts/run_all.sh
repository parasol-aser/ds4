#!/bin/zsh
# Context sweep, ablation, stage profiles, sampled decode, perplexity, MoE.
# Needs: ds4 built at $DS4_ROOT, llama.cpp at $LLAMA_BIN, prompts from make_prompts.py,
# wikitext-2 test file at $OUT/wikitext-2-raw/wiki.test.raw.
set -u
DS4_ROOT=${DS4_ROOT:-$(cd .. && pwd)}; OUT=${OUT:-$PWD/out}; L=$OUT/logs; mkdir -p $L
LL=${LLAMA_BIN:-$HOME/github/llama.cpp/build/bin}
Q4=$DS4_ROOT/gguf/Qwen3.8-27B-Q4_K.gguf; UD=$DS4_ROOT/gguf/Qwen3.8-27B-UD-Q4_K_M.gguf; Q8=$DS4_ROOT/gguf/Qwen3.8-27B-Q8_0.gguf
DS=$DS4_ROOT/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
SHORT=$'<|im_start|>user\nWrite a short paragraph about the ocean.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
cd $DS4_ROOT
log() { echo "[$(date +%H:%M:%S)] $*" >> $L/progress.txt; }
ctx() { case $1 in 1k) echo 2048;; 4k) echo 5120;; 8k) echo 9216;; 16k) echo 18432;; 32k) echo 34816;; esac; }

for n in 1k 4k 8k 16k 32k; do
  ./ds4 -m $Q4 --prompt-file $OUT/prompt_$n.txt -n 64 --temp 0 -c $(ctx $n) > /dev/null 2> $L/ds4_ctx_$n.log
  $LL/llama-completion -m $Q4 -f $OUT/prompt_$n.txt -n 64 --temp 0 -c $(ctx $n) -no-cnv --no-warmup > /dev/null 2> $L/llama_ctx_$n.log
  log "ctx $n done"
done

abl() { name=$1; shift; for i in 1 2; do env "$@" ./ds4 -m $Q4 -p "$SHORT" -n 64 --temp 0 -c 4096 > /dev/null 2> $L/abl_${name}_$i.log; done; log "abl $name done"; }
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

PROF=(DS4_METAL_CB_TIMES=1 DS4_METAL_DECODE_STAGE_PROFILE=1)
env $PROF ./ds4 -m $Q4 -p "$SHORT" -n 12 --temp 0 -c 4096 > /dev/null 2> $L/stage_after.log
env $ALLOFF $PROF ./ds4 -m $Q4 -p "$SHORT" -n 12 --temp 0 -c 4096 > /dev/null 2> $L/stage_alloff.log
env $PROF ./ds4 -m $Q4 --prompt-file $OUT/prompt_8k.txt -n 12 --temp 0 -c 9216 > /dev/null 2> $L/stage_after_8k.log
log "stage profiles done"

for f in Q4 UD Q8; do
  eval m=\$$f
  ./ds4 -m $m -p "$SHORT" -n 64 -c 4096 > /dev/null 2> $L/ds4_sampled_$f.log
  ./ds4 -m $m -p "$SHORT" -n 64 --temp 0 -c 4096 > /dev/null 2> $L/ds4_greedy_$f.log
  $LL/llama-completion -m $m -p "$SHORT" -n 64 -c 4096 -no-cnv --no-warmup > /dev/null 2> $L/llama_sampled_$f.log
  $LL/llama-completion -m $m -p "$SHORT" -n 64 --temp 0 -c 4096 -no-cnv --no-warmup > /dev/null 2> $L/llama_greedy_$f.log
  $LL/llama-completion -m $m -f $PWD/prompt_1439.txt -n 8 --temp 0 -c 4096 -no-cnv --no-warmup > /dev/null 2> $L/llama_prefill_$f.log
  $LL/llama-perplexity -m $m -f $OUT/wikitext-2-raw/wiki.test.raw -c 2048 --chunks 40 > $L/ppl_$f.log 2>&1
  log "file $f done"
done

./ds4 -m $DS -p "Write a short paragraph about the ocean." -n 64 --temp 0 -c 4096 > /dev/null 2> $L/ds4_dsv4.log
$LL/llama-completion -m $DS -p "Write a short paragraph about the ocean." -n 64 --temp 0 -c 4096 -no-cnv --no-warmup > /dev/null 2> $L/llama_dsv4.log
env $PROF ./ds4 -m $DS -p "Write a short paragraph about the ocean." -n 12 --temp 0 -c 4096 > /dev/null 2> $L/stage_dsv4.log
log "ALL DONE"
