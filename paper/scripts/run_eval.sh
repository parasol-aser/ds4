#!/bin/zsh
# ds4-eval downstream check: first 10 built-in questions, thinking on, 2500-token reply budget.
set -eu
HERE=${0:A:h}
# repository root: nearest ancestor of this script that contains ds4.c (works for paper/scripts and misc/paper/scripts)
find_root() { d=$1; while [ "$d" != "/" ]; do [ -f "$d/ds4.c" ] && { echo "$d"; return; }; d=${d:h}; done; echo "$1"; }
DS4_ROOT=${DS4_ROOT:-$(find_root "$HERE")}; OUT=${OUT:-$HERE/out}; L=$OUT/logs/eval; mkdir -p "$L"
cd "$DS4_ROOT"
for f in Q4_K Q4_0 Q8_0; do
  ./ds4-eval -m "gguf/Qwen3.8-27B-$f.gguf" -c 8192 -n 2500 --questions 10 --plain --trace "$L/trace_$f.jsonl" > "$L/eval_$f.log" 2>&1
  grep -a "ds4-eval:" "$L/eval_$f.log" | tail -1
done
