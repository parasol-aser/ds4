#!/bin/zsh
# MLX baseline (mlx-lm >= 0.31 for the qwen3_5 architecture).
set -u
OUT=${OUT:-$PWD/out}; L=$OUT/logs; mkdir -p $L
M=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('mlx-community/Qwen3.8-27B-4bit'))")
P="Write a short paragraph about the ocean."
python3 -m mlx_lm generate --model "$M" --prompt "$P" --max-tokens 64 --temp 0.0 > $L/mlx_short.log 2>&1
for n in 1k 4k 8k 16k 32k; do
  python3 -m mlx_lm generate --model "$M" --prompt "$(cat $OUT/prompt_$n.txt)" --max-tokens 64 --temp 0.0 --ignore-chat-template > $L/mlx_ctx_$n.log 2>&1
done
MLX_MODEL="$M" python3 mlx_ppl.py $OUT/wikitext-2-raw/wiki.test.raw 40 > $L/mlx_ppl.log 2>&1
echo "[$(date +%H:%M:%S)] MLX DONE" >> $L/progress.txt
