#!/bin/zsh
# ds4-eval downstream check: first 10 built-in questions, thinking on, 2500-token reply budget.
set -u
DS4_ROOT=${DS4_ROOT:-$(cd .. && pwd)}; OUT=${OUT:-$PWD/out}; L=$OUT/logs/eval; mkdir -p $L
cd $DS4_ROOT
for f in Q4_K Q4_0 Q8_0; do
  ./ds4-eval -m gguf/Qwen3.8-27B-$f.gguf -c 8192 -n 2500 --questions 10 --plain --trace $L/trace_$f.jsonl > $L/eval_$f.log 2>&1
  grep -a "ds4-eval:" $L/eval_$f.log | tail -1
done
