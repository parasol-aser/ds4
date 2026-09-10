import re, os, sys, glob, collections
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os; S=os.environ.get('OUT', 'out')
L=S+'/logs'; OUT=os.environ.get('FIGS', '../figs'); os.makedirs(OUT, exist_ok=True)

def ds4_gen(path):
    t=open(path).read(); m=re.findall(r'prefill: ([\d.]+) t/s, generation: ([\d.]+) t/s', t); return (float(m[-1][0]), float(m[-1][1]))
def llama_gen(path):
    t=open(path).read()
    pp=re.search(r'prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens.*?([\d.]+) tokens per second', t)
    tg=re.search(r'\n[^\n]*\beval time =\s+([\d.]+) ms /\s+(\d+) runs.*?([\d.]+) tokens per second', t)
    return (float(pp.group(3)), int(pp.group(2)), float(tg.group(3)))
plt.rcParams.update({'font.size': 10, 'figure.dpi': 150})
# 1. context sweep
names=['1k','4k','8k','16k','32k']; toks=[]; d_dec=[]; l_dec=[]; d_pp=[]; l_pp=[]; m_dec=[]; m_pp=[]
def mlx_gen(path):
    t=open(path).read()
    pp=re.search(r'Prompt: (\d+) tokens, ([\d.]+) tokens-per-sec', t); tg=re.search(r'Generation: (\d+) tokens, ([\d.]+) tokens-per-sec', t)
    return (float(pp.group(2)), float(tg.group(2)))
for n in names:
    d=ds4_gen(f'{L}/ds4_ctx_{n}.log'); l=llama_gen(f'{L}/llama_ctx_{n}.log'); m=mlx_gen(f'{L}/mlx_ctx_{n}.log')
    toks.append(l[1]); d_dec.append(d[1]); l_dec.append(l[2]); d_pp.append(d[0]); l_pp.append(l[0]); m_dec.append(m[1]); m_pp.append(m[0])
print('mlx', list(zip(toks, m_dec, m_pp)))
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
ax[0].plot(toks, d_dec, 'o-', label='ds4 (Q4_K)'); ax[0].plot(toks, m_dec, '^-.', label='MLX (4-bit affine)'); ax[0].plot(toks, l_dec, 's--', label='llama.cpp (Q4_K)')
ax[0].set_xscale('log'); ax[0].set_xlabel('context length (tokens)'); ax[0].set_ylabel('decode t/s'); ax[0].set_ylim(0, max(d_dec+m_dec)*1.15); ax[0].legend(fontsize=8); ax[0].set_title('Decode')
ax[1].plot(toks, d_pp, 'o-', label='ds4'); ax[1].plot(toks, m_pp, '^-.', label='MLX'); ax[1].plot(toks, l_pp, 's--', label='llama.cpp')
ax[1].set_xscale('log'); ax[1].set_xlabel('prompt length (tokens)'); ax[1].set_ylabel('prefill t/s'); ax[1].set_ylim(0, max(d_pp+l_pp)*1.15); ax[1].legend(); ax[1].set_title('Prefill')
fig.tight_layout(); fig.savefig(OUT+'/context.pdf'); print('context.pdf', list(zip(toks, d_dec, l_dec, d_pp, l_pp)))
# 2. ablation bars
abl=[('baseline','all on'),('no_fused_swiglu','no fused gate/up'),('no_multi_proj','no multi-weight proj'),('no_fast_mv','no vectorized single mv'),('no_ksplit','no K-split (ffn down)'),('multi_nr0_2','multi kernel 2 rows'),('fa_nwg32','fixed 32 KV workgroups'),('no_fused_norm','no fused residual+norm'),('all_off','all off')]
vals=[]; labels=[]
for key,lab in abl:
    r=[ds4_gen(f'{L}/abl_{key}_{i}.log')[1] for i in (1,2) if os.path.exists(f'{L}/abl_{key}_{i}.log')]
    if r: vals.append(sum(r)/len(r)); labels.append(lab)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.barh(range(len(vals)), vals, color=['#444']+['#888']*(len(vals)-2)+['#444'] if len(vals)>1 else '#444')
ax.set_yticks(range(len(vals))); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.set_xlabel('decode t/s (Q4_K, greedy)'); ax.set_xlim(min(vals)*0.9, max(vals)*1.03)
for i,v in enumerate(vals): ax.text(v+0.05, i, f'{v:.1f}', va='center', fontsize=8)
fig.tight_layout(); fig.savefig(OUT+'/ablation.pdf'); print('ablation.pdf', list(zip(labels, [round(v,2) for v in vals])))
# 3. stage breakdown bars (after vs all-off) from stage logs
def stages(path):
    gpu=None; agg=collections.defaultdict(list); toks=set()
    for line in open(path):
        m=re.search(r"gpu span ([\d.]+) ms", line)
        if m: gpu=float(m.group(1)); continue
        m=re.search(r"part=qwen layer=(\d+) pos=(\d+) tokens=1 (\w+)=([\d.]+) ms", line)
        if m and gpu is not None:
            il=int(m.group(1)); pos=int(m.group(2)); st=m.group(3)
            if pos>=8: agg[st].append(gpu); toks.add(pos)
            gpu=None
    n=len(toks); return {k: sum(v)/n for k,v in agg.items()}
groups=[('FFN gate/up',['ffn_gate_up']),('FFN down',['ffn_down']),('projections',['gdn_proj','attn_proj']),('out proj (+gate/norm)',['gdn_out','attn_out']),('recurrence',['gdn_recurrence']),('attention',['attention','attn_prologue']),('residual/norm',['attn_residual','ffn_residual','attn_norm','ffn_norm']),('output head',['output_head'])]
rows=[]
for key,lab in [('stage_alloff','fusions off'),('stage_after','all on'),('stage_after_8k','all on, 8k ctx')]:
    p=f'{L}/{key}.log'
    if os.path.exists(p):
        st=stages(p); rows.append((lab,[sum(st.get(s,0) for s in ss) for _,ss in groups]))
fig, ax = plt.subplots(figsize=(6.4, 2.4)); left=[0]*len(rows)
colors=plt.cm.tab20.colors
for gi,(glab,_) in enumerate(groups):
    v=[r[1][gi] for r in rows]
    ax.barh([r[0] for r in rows], v, left=left, label=glab, color=colors[gi]); left=[a+b for a,b in zip(left,v)]
ax.set_xlabel('GPU ms per token'); ax.invert_yaxis(); ax.legend(fontsize=7, ncol=4, loc='lower right', bbox_to_anchor=(1.0,1.02), frameon=False)
fig.tight_layout(); fig.savefig(OUT+'/breakdown.pdf'); print('breakdown.pdf', [(r[0],[round(x,2) for x in r[1]], round(sum(r[1]),2)) for r in rows])
