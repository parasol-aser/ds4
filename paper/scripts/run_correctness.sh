#!/bin/zsh
# Reference-logit comparison at several decode steps and context lengths.
# Step k means: prefill, then k generated tokens fed back; the dumped logits are those that pick token k+1.
set -eu
HERE=${0:A:h}
# repository root: nearest ancestor of this script that contains ds4.c (works for paper/scripts and misc/paper/scripts)
find_root() { d=$1; while [ "$d" != "/" ]; do [ -f "$d/ds4.c" ] && { echo "$d"; return; }; d=${d:h}; done; echo "$1"; }
DS4_ROOT=${DS4_ROOT:-$(find_root "$HERE")}; OUT=${OUT:-$HERE/out}; L=$OUT/logs; mkdir -p "$L"
LL=${LLAMA_BIN:-$HOME/github/llama.cpp/build/bin}
[ -x "$OUT/logits_dump" ] || c++ -std=c++17 -O2 -I "$HOME/github/llama.cpp/include" -I "$HOME/github/llama.cpp/ggml/include" \
  "$HERE/logits_dump.cpp" -L "$LL" -lllama -lggml -lggml-base -Wl,-rpath,"$LL" -o "$OUT/logits_dump"
M=${MODEL:-$DS4_ROOT/gguf/Qwen3.8-27B-Q4_K.gguf}
SHORT=$'<|im_start|>user\nWrite a short paragraph about the ocean.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
printf '%s' "$SHORT" > "$L/prompt_short.txt"
cd "$DS4_ROOT"
cor() { tag=$1; pfile=$2; step=$3; c=$4
  DS4_QWEN_LOGIT_DUMP_STEP=$step DS4_GLM_LOGIT_DUMP="$L/ds4_${tag}_s$step.bin" ./ds4 -m "$M" --prompt-file "$pfile" -n 64 --temp 0 -c $c > "$L/ds4_${tag}_s$step.out" 2>/dev/null
  "$OUT/logits_dump" "$M" "@$pfile" 64 5 $step "$L/ref_${tag}_s$step.bin" > "$L/ref_${tag}_s$step.txt" 2>&1
  python3 "$HERE/cmp.py" "$L/ds4_${tag}_s$step.bin" "$L/ref_${tag}_s$step.bin" | tee "$L/cmp_${tag}_s$step.txt"
}
for st in 0 4 16 40 63; do cor short "$L/prompt_short.txt" $st 4096; done
for st in 0 4 63; do cor long "$HERE/prompt_1439.txt" $st 4096; done
cor ctx8k "$OUT/prompt_8k.txt" 4 9216
