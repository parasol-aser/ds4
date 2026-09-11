#!/usr/bin/env python3
"""Generate the paper's LaTeX table bodies from out/results.json into ../tables/*.tex,
so table values and captions cannot drift from the logs."""
import json, os, re
OUT = os.environ.get('OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
TAB = os.environ.get('TABLES', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'tables'))
os.makedirs(TAB, exist_ok=True)
R = json.load(open(os.path.join(OUT, 'results.json')))
def w(name, text, cols=1):
    text = text.rstrip('\n')
    if not text: text = '(pending)' + ' &' * (cols - 1) + ' \\\\'
    open(os.path.join(TAB, name), 'w').write(text + '%\n')
def m(x, d=1): return 'n/a' if x is None else f"{x['mean']:.{d}f}"
def msd(x, d=1): return 'n/a' if x is None else (f"{x['mean']:.{d}f} $\\pm$ {x['sd']:.{d}f}" if x['n'] > 1 else f"{x['mean']:.{d}f}")

# headline
H = R['headline']; P = R.get('prefill_1439', {})
names = {'Q4': 'pure Q4\\_K', 'Q40': 'pure Q4\\_0', 'UD': 'UD-Q4\\_K\\_M', 'Q8': 'Q8\\_0'}
rows = []
for f in ['Q4', 'Q40', 'UD', 'Q8']:
    if f not in H: continue
    h = H[f]; p = P.get(f, {})
    pre = f"{p.get('ds4', float('nan')):.0f} / {p.get('llama', float('nan')):.0f}" if p else 'n/a'
    rows.append(f"{names[f]} & {m(h['ds4_greedy_forward'])} & {m(h['ds4_greedy_loop'])} & {m(h['ds4_sampled_loop'])} & {m(h['llama_greedy_forward'])} & {m(h['llama_default_loop'])} & {m(h['llama_matched_loop'])} & {pre} \\\\")
w('headline.tex', '\n'.join(rows), 8)
# sampler costs
s = H.get('Q4', {})
w('sampler.tex', f"{m(s.get('ds4_sampled_sampler'),2)} & {m(s.get('llama_default_sampler'),2)} & {m(s.get('llama_matched_sampler'),2)}\n")
if 'MLX' in H:
    w('mlx_short.tex', f"{m(H['MLX']['mlx_loop'])}\n")

# ablation
A = R['ablation']; base = A['baseline']['loop']['mean'] if 'baseline' in A else None
order = ['baseline', 'no_fused_swiglu', 'no_multi_proj', 'no_fast_mv', 'no_ksplit', 'fa_nwg32', 'no_fused_norm', 'multi_nr0_2', 'q4k_classic', 'all_off']
rows = []
for k in order:
    if k not in A: continue
    v = A[k]['loop']; d = v['mean'] - base if base is not None else 0
    rows.append(f"{A[k]['label'].replace('_', chr(92)+'_')} & {msd(v, 2)} & {'' if k == 'baseline' else f'{d:+.2f}'} \\\\")
w('ablation.tex', '\n'.join(rows), 3)

# context
C = R['context']; rows = []
for n in ['1k', '4k', '8k', '16k', '32k']:
    if n not in C: continue
    c = C[n]
    rows.append(f"{c['prompt_tokens']} & {msd(c['ds4_loop'])} & {msd(c['mlx_loop'])} & {msd(c['llama_forward'])} & {msd(c['ds4_prefill'], 0)} & {msd(c['mlx_prefill'], 0)} & {msd(c['llama_prefill'], 0)} \\\\")
w('context.tex', '\n'.join(rows), 7)

# correctness
K = R['correctness']; rows = []
labels = {'short': '20-token prompt', 'long': '1,439-token prompt', 'ctx8k': '8,590-token prompt'}
for tag in ['short', 'long', 'ctx8k']:
    for st in [0, 4, 16, 40, 63]:
        key = f'{tag}_s{st}'
        if key in K and 'mean' in K[key]:
            v = K[key]
            if not v['argmax']: continue   # histories diverged before this step (see the .out texts); not a like-for-like point
            rows.append(f"{labels[tag]} & {st} & {v['mean']:.4f} & {v['max']:.3f} & {v['kl']:.1e} & {'agrees' if v['argmax'] else 'DIFFERS'} \\\\")
w('correctness.tex', '\n'.join(rows), 6)

# perplexity
PP = R.get('ppl', {}); PD = R.get('ppl_paired', {})
def ppl(f): x = PP.get(f); return 'n/a' if not x else (f"${x['ppl']:.2f} \\pm {x['se']:.2f}$" if x.get('se') else f"{x['ppl']:.2f}")
def pd(k):   # relative perplexity change exp(d)-1 with the delta-method standard error exp(d)*se
    x = PD.get(k); return '' if not x else f"${100*(x['ppl_ratio']-1):+.1f}\\% \\pm {100*x['ppl_ratio']*x['se']:.1f}$"
rows = [f"pure Q4\\_K, imatrix & 0.5625 & {ppl('Q4')} & \\\\",
        f"UD-Q4\\_K\\_M & 0.58 & {ppl('UD')} & {pd('UD-Q4')} \\\\",
        f"Q8\\_0 & 1.0625 & {ppl('Q8')} & {pd('Q8-Q4')} \\\\",
        f"pure Q4\\_0, imatrix & 0.5625 & {ppl('Q40')} & {pd('Q40-Q4')} \\\\",
        f"MLX 4-bit affine, group 64 & 0.5625 & {ppl('MLX')} & \\\\"]
w('ppl.tex', '\n'.join(rows), 4)

# sha256
sha = os.path.join(OUT, 'sha256.txt')
if os.path.exists(sha):
    rows = []
    for line in open(sha):
        parts = line.split()
        if len(parts) == 2: rows.append(f"\\texttt{{{parts[1].replace('_', chr(92)+'_')}}} & \\texttt{{{parts[0][:16]}\\ldots{{}}{parts[0][-8:]}}} \\\\")
    w('sha256.tex', '\n'.join(rows), 2)
# stage table (Q4_K): off / on / on at 8k
S = R.get('stage', {})
def st(key, kind, name):
    x = S.get(key, {}).get('stages', {}).get(f'{kind}:{name}')
    return x
ROWS = [('both', 'FFN gate/up', ['ffn_gate_up']), ('both', 'FFN down', ['ffn_down']),
        ('DeltaNet', 'projections $qkv$, $z$, $\\alpha$, $\\beta$', ['gdn_proj']),
        ('attention', 'projections $q|$gate, $k$, $v$', ['attn_proj']),
        ('DeltaNet', 'output norm + out projection', ['gdn_out']), ('attention', 'gate + out projection', ['attn_out']),
        ('DeltaNet', 'conv + recurrence', ['gdn_recurrence']), ('attention', 'split-KV attention (+pad, reduce)', ['attention']),
        ('attention', '$q$/$k$ prologue, $V$ store', ['attn_prologue']),
        ('both', 'residual adds and norms', ['attn_residual', 'ffn_residual', 'attn_norm', 'ffn_norm']),
        ('', 'output head (Q8\\_0)', ['output_head'])]
def cell(key, names, layer):
    stg = S.get(key, {}).get('stages', {})
    kinds = ['gdn', 'attn'] if layer in ('both', '') else (['gdn'] if layer == 'DeltaNet' else ['attn'])
    per = [stg[f'{k}:{n}'] for k in kinds for n in names if f'{k}:{n}' in stg]
    if not per: return ('', '')
    ms = sum(x['ms_per_token'] for x in per)
    if layer == 'both':  # per-layer value: average over the 64 layers weighted by count
        us = ms * 1000 / 64
    elif layer == 'DeltaNet': us = ms * 1000 / 48
    elif layer == 'attention': us = ms * 1000 / 16
    else: us = None
    return (f'{us:.1f}' if us is not None else '', f'{ms:.2f}')
rows = []
for layer, label, names in ROWS:
    a = cell('stage_alloff', names, layer); b = cell('stage_after', names, layer); c = cell('stage_after_8k', names, layer)
    rows.append(f"{layer} & {label} & {a[0]} & {a[1]} & {b[0]} & {b[1]} & {c[1]} \\\\")
tot = [f"{S[k]['total_ms_per_token']:.2f}" if k in S else '' for k in ['stage_alloff', 'stage_after', 'stage_after_8k']]
rows.append('\\midrule')
rows.append(f" & sum of stage spans & & {tot[0]} & & {tot[1]} & {tot[2]} \\\\")
w('stages.tex', '\n'.join(rows), 7)
pos = {k: S[k]['positions'] for k in S}
w('stage_positions.tex', ' '.join(f"{k}: {v[0]}--{v[1]}" for k, v in pos.items()))
# MoE table
if 'stage_dsv4' in S:
    stg = S['stage_dsv4']['stages']; agg = {}
    for k, v in stg.items():
        name = k.split(':')[1]; agg.setdefault(name, [0.0, 0.0]); agg[name][0] += v['ms_per_token']
    # per-layer us: ms/token * 1000 / 43 layers
    order = ['routed_moe', 'q_path', 'attn_inv_rope', 'attn_output', 'router', 'kv_path', 'attn_hc_pre', 'ffn_hc_pre', 'shared_down']
    rows = []
    for name in order:
        if name in agg:
            ms = agg[name][0]; rows.append(f"\\texttt{{{name.replace('_', chr(92)+'_')}}} & {ms*1000/43:.0f} & {ms:.2f} \\\\")
    rest = sum(v[0] for k, v in agg.items() if k not in order)
    rows.append(f"other (compressor and indexer updates, norms; amortized) & & {rest:.2f} \\\\")
    rows.append('\\midrule'); rows.append(f"sum of stage spans & & {S['stage_dsv4']['total_ms_per_token']:.1f} \\\\")
    w('moe.tex', '\n'.join(rows), 3)
print('tables written to', TAB)
