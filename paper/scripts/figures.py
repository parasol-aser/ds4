#!/usr/bin/env python3
"""Figures from out/results.json (run analyze.py first)."""
import json, os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.environ.get('OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
FIGS = os.environ.get('FIGS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'figs'))
os.makedirs(FIGS, exist_ok=True)
R = json.load(open(os.path.join(OUT, 'results.json')))
plt.rcParams.update({'font.size': 9, 'figure.dpi': 150})

# 1. context sweep: ms per decode forward (mean +- sd over repeats), prefill t/s
ctx = R['context']; names = [n for n in ['1k', '4k', '8k', '16k', '32k'] if n in ctx]
toks = [ctx[n]['prompt_tokens'] for n in names]
def series(key):
    m = [ctx[n][key]['mean'] if ctx[n][key] else float('nan') for n in names]
    s = [ctx[n][key]['sd'] if ctx[n][key] else 0 for n in names]
    return m, s
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
for key, lab, mk in [('ds4_loop', 'ds4, Q4_K (generation loop)', 'o-'), ('mlx_loop', 'MLX, 4-bit affine (generation loop)', '^-.'), ('llama_forward', 'llama.cpp, Q4_K (forward only)', 's--')]:
    m, s = series(key); ax[0].errorbar(toks, m, yerr=s, fmt=mk, capsize=2, label=lab)
ax[0].set_xscale('log'); ax[0].set_xlabel('prompt tokens'); ax[0].set_ylabel('ms per decode forward'); ax[0].set_ylim(0, 55); ax[0].legend(fontsize=7); ax[0].set_title('Decode')
for key, lab, mk in [('ds4_prefill', 'ds4', 'o-'), ('mlx_prefill', 'MLX', '^-.'), ('llama_prefill', 'llama.cpp', 's--')]:
    m, s = series(key); ax[1].errorbar(toks, m, yerr=s, fmt=mk, capsize=2, label=lab)
ax[1].set_xscale('log'); ax[1].set_xlabel('prompt tokens'); ax[1].set_ylabel('prefill tokens/s'); ax[1].set_ylim(0, 320); ax[1].legend(fontsize=7); ax[1].set_title('Prefill')
fig.tight_layout(); fig.savefig(os.path.join(FIGS, 'context.pdf'))

# 2. ablation: leave-one-out delta in ms per decode forward, zero baseline, error bars from repeats
abl = R['ablation']; base = abl['baseline']['loop']
order = ['no_fused_swiglu', 'no_multi_proj', 'no_fast_mv', 'no_ksplit', 'fa_nwg32', 'no_fused_norm', 'multi_nr0_2', 'q4k_classic', 'all_off']
order = [k for k in order if k in abl]
labels = [abl[k]['label'] for k in order]
delta = [abl[k]['loop']['mean'] - base['mean'] for k in order]
err = [(abl[k]['loop']['sd'] ** 2 + base['sd'] ** 2) ** 0.5 for k in order]
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.barh(range(len(order)), delta, xerr=err, color='#666', capsize=2)
ax.axvline(0, color='black', lw=0.8)
ax.set_yticks(range(len(order))); ax.set_yticklabels(labels); ax.invert_yaxis()
ax.set_xlabel(f'added ms per decode forward vs. all on ({base["mean"]:.2f} ms)')
for i, v in enumerate(delta): ax.text(v + (0.05 if v >= 0 else -0.05), i, f'{v:+.2f}', va='center', ha='left' if v >= 0 else 'right', fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, 'ablation.pdf'))

# 3. stage breakdown (grouped)
groups = [('FFN gate/up', ['ffn_gate_up']), ('FFN down', ['ffn_down']), ('projections', ['gdn_proj', 'attn_proj']),
          ('out proj (+gate/norm)', ['gdn_out', 'attn_out']), ('conv+recurrence', ['gdn_recurrence']),
          ('attention (+prologue)', ['attention', 'attn_prologue']), ('residual/norm', ['attn_residual', 'ffn_residual', 'attn_norm', 'ffn_norm']),
          ('output head', ['output_head'])]
rows = []
for key, lab in [('stage_alloff', 'optimizations off'), ('stage_after', 'on'), ('stage_after_8k', 'on, 8.6k-token context')]:
    if key in R['stage']:
        stg = R['stage'][key]['stages']
        rows.append((lab, [sum(v['ms_per_token'] for k, v in stg.items() if k.split(':')[1] in names_) for _, names_ in groups]))
fig, ax = plt.subplots(figsize=(6.4, 2.4)); left = [0] * len(rows); colors = plt.cm.tab20.colors
for gi, (glab, _) in enumerate(groups):
    v = [r[1][gi] for r in rows]
    ax.barh([r[0] for r in rows], v, left=left, label=glab, color=colors[gi]); left = [a + b for a, b in zip(left, v)]
ax.set_xlabel('GPU ms per decode forward (sum of isolated stage spans)'); ax.invert_yaxis()
ax.legend(fontsize=6.5, ncol=4, loc='lower right', bbox_to_anchor=(1.0, 1.02), frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(FIGS, 'breakdown.pdf'))
print('figures written to', FIGS)
