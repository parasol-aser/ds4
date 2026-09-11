# Perplexity of an MLX model on wikitext-2 test with llama-perplexity's protocol:
# consecutive 2048-token chunks, score the second half of each chunk given the first.
import sys, math, glob, time
import mlx.core as mx
from mlx_lm import load
import os; model_path = os.environ['MLX_MODEL']
text = open(sys.argv[1]).read()
n_ctx, n_chunks = 2048, int(sys.argv[2]) if len(sys.argv) > 2 else 40
model, tok = load(model_path)
ids = tok.encode(text)
print('tokens', len(ids), 'chunks', n_chunks)
nll = 0.0; count = 0; t0 = time.time()
for c in range(n_chunks):
    chunk = mx.array(ids[c*n_ctx:(c+1)*n_ctx])[None]
    logits = model(chunk)                      # [1, n_ctx, vocab]
    logp = logits[0, :-1].astype(mx.float32)
    logp = logp - mx.logsumexp(logp, axis=-1, keepdims=True)
    tgt = chunk[0, 1:]
    tok_lp = mx.take_along_axis(logp, tgt[:, None], axis=-1)[:, 0]
    half = n_ctx // 2
    sel = tok_lp[half:]                        # targets at positions half+1 .. n_ctx-1, as llama-perplexity
    nll -= float(mx.sum(sel)); count += sel.shape[0]
    mx.eval(logits)
    if (c + 1) % 5 == 0:
        print(f'[{c+1}] ppl so far {math.exp(nll/count):.4f}  ({time.time()-t0:.0f}s)', flush=True)
print(f'Final estimate: PPL = {math.exp(nll/count):.4f} over {count} tokens')
