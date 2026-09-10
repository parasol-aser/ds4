#!/bin/zsh
# Pure Q4_0 file (same imatrix and Q8_0 head as the pure Q4_K file), then its benchmarks.
set -u
DS4_ROOT=${DS4_ROOT:-$(cd .. && pwd)}; OUT=${OUT:-$PWD/out}; L=$OUT/logs; mkdir -p $L
LL=${LLAMA_BIN:-$HOME/github/llama.cpp/build/bin}
cd $DS4_ROOT/gguf
[ -f Qwen3.8-27B-Q4_0.gguf ] || $LL/llama-quantize --allow-requantize --pure --imatrix imatrix_unsloth.gguf \
  --output-tensor-type q8_0 --token-embedding-type q4_0 Qwen3.8-27B-Q8_0.gguf Qwen3.8-27B-Q4_0.gguf Q4_0 8
[ -f Qwen3.8-27B-Q4_K.gguf ] || $LL/llama-quantize --allow-requantize --pure --imatrix imatrix_unsloth.gguf \
  --output-tensor-type q8_0 --token-embedding-type q4_k Qwen3.8-27B-Q8_0.gguf Qwen3.8-27B-Q4_K.gguf Q4_K_S 8
cd $DS4_ROOT
Q40=gguf/Qwen3.8-27B-Q4_0.gguf
SHORT=$'<|im_start|>user\nWrite a short paragraph about the ocean.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
FR=$'<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
$LL/llama-bench -m $Q40 -p 0 -n 32 > $L/llama_bench_q40.log 2>&1      # warms the page cache too
for i in 1 2; do ./ds4 -m $Q40 -p "$SHORT" -n 64 --temp 0 -c 4096 > /dev/null 2> $L/ds4_q40_greedy_$i.log; done
DS4_QWEN_DISABLE_FAST_MV=1 DS4_QWEN_DISABLE_MULTI_PROJ=1 DS4_QWEN_DISABLE_FUSED_SWIGLU=1 \
  ./ds4 -m $Q40 -p "$SHORT" -n 64 --temp 0 -c 4096 > /dev/null 2> $L/ds4_q40_classic.log
$LL/llama-completion -m $Q40 -p "$SHORT" -n 64 --temp 0 -c 4096 -no-cnv --no-warmup > /dev/null 2> $L/llama_q40_greedy.log
DS4_METAL_CB_TIMES=1 DS4_METAL_DECODE_STAGE_PROFILE=1 ./ds4 -m $Q40 -p "$SHORT" -n 12 --temp 0 -c 4096 > /dev/null 2> $L/stage_q40.log
DS4_QWEN_LOGIT_DUMP_STEP=4 DS4_GLM_LOGIT_DUMP=$L/ds4_q40_france_s4.bin ./ds4 -m $Q40 -p "$FR" -n 8 --temp 0 -c 4096 > /dev/null 2>&1
$OUT/logits_dump $Q40 "$FR" 8 5 4 $L/ref_q40_france_s4.bin > $L/ref_q40_france.txt 2>&1
python3 $OLDPWD/cmp.py $L/ds4_q40_france_s4.bin $L/ref_q40_france_s4.bin > $L/cmp_q40.txt
$LL/llama-perplexity -m $Q40 -f $OUT/wikitext-2-raw/wiki.test.raw -c 2048 --chunks 40 > $L/ppl_Q40.log 2>&1
echo "[$(date +%H:%M:%S)] Q40 DONE" >> $L/progress.txt
