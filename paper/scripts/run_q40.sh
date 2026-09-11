#!/bin/zsh
# Build the two pure files (same Q8_0 source, same imatrix, same Q8_0 head) with llama-quantize.
# Their benchmarks, profile and perplexity are part of run_all.sh.
set -eu
HERE=${0:A:h}
# repository root: nearest ancestor of this script that contains ds4.c (works for paper/scripts and misc/paper/scripts)
find_root() { d=$1; while [ "$d" != "/" ]; do [ -f "$d/ds4.c" ] && { echo "$d"; return; }; d=${d:h}; done; echo "$1"; }
DS4_ROOT=${DS4_ROOT:-$(find_root "$HERE")}
LL=${LLAMA_BIN:-$HOME/github/llama.cpp/build/bin}
cd "$DS4_ROOT/gguf"
[ -f imatrix_unsloth.gguf ] || curl -LO https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/4ca720788d1e01f1bff70c033e0d0028fd02e502/imatrix_unsloth.gguf
# VERIFY_HASHES=1 checks existing files against the hashes recorded in ../logs/sha256.txt (slow: ~80 GB read)
if [ "${VERIFY_HASHES:-0}" = "1" ] && [ -f "$HERE/../logs/sha256.txt" ]; then shasum -a 256 -c "$HERE/../logs/sha256.txt"; fi
[ -f Qwen3.8-27B-Q4_K.gguf ] || "$LL/llama-quantize" --allow-requantize --pure --imatrix imatrix_unsloth.gguf \
  --output-tensor-type q8_0 --token-embedding-type q4_k Qwen3.8-27B-Q8_0.gguf Qwen3.8-27B-Q4_K.gguf Q4_K_S 8
[ -f Qwen3.8-27B-Q4_0.gguf ] || "$LL/llama-quantize" --allow-requantize --pure --imatrix imatrix_unsloth.gguf \
  --output-tensor-type q8_0 --token-embedding-type q4_0 Qwen3.8-27B-Q8_0.gguf Qwen3.8-27B-Q4_0.gguf Q4_0 8
shasum -a 256 Qwen3.8-27B-Q4_K.gguf Qwen3.8-27B-Q4_0.gguf
