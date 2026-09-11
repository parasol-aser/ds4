#!/usr/bin/env python3
"""Build the paper's result set (out/results.json) from the logs written by run_all.sh
and run_correctness.sh, and print the tables.

Timing conventions (see the paper's timing figure):
  forward  = milliseconds per decode forward (GPU eval of one token; 63 forwards in a 64-token run)
  loop     = milliseconds per decode forward of the whole generation loop (sampling and emission included)
ds4's forward is the host-elapsed time of the forward call (DS4_TOKEN_TIMING); llama.cpp's is its 'eval time'
per run and its loop proxy adds 'sampling time'; mlx-lm reports n output tokens over the time of n-1 forwards,
so its rate is converted with n/(n-1). Rates are 1000 / mean ms."""
import re, glob, os, json, math, statistics as st, collections

OUT = os.environ.get('OUT', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out'))
L = os.path.join(OUT, 'logs')
R = {}

def read(p):
    with open(p, errors='replace') as f: return f.read()

def mean_sd(v):
    v = [x for x in v if x is not None]
    if not v: return None
    return {'mean': st.mean(v), 'sd': st.stdev(v) if len(v) > 1 else 0.0, 'n': len(v), 'values': v}

def fmt(m, digits=2):
    return 'n/a' if m is None else (f"{m['mean']:.{digits}f}±{m['sd']:.{digits}f}" if m['n'] > 1 else f"{m['mean']:.{digits}f}")

# ---------------- headline ----------------
def ds4_greedy(p):
    t = read(p)
    ev = [float(x) for x in re.findall(r'decode eval \d+ took ([\d.]+) ms', t)]
    g = re.search(r'generation: ([\d.]+) t/s', t)
    fwd = st.mean(ev) if ev else None      # all 63 decode forwards, as for llama.cpp
    loop = (64.0 / float(g.group(1))) / 63 * 1000 if g else None
    return fwd, loop

def ds4_sampled(p):
    t = read(p)
    m = re.search(r'sample ([\d.]+) ms, emit ([\d.]+) ms, eval ([\d.]+) ms', t)
    g = re.search(r'generation: ([\d.]+) t/s', t)
    if not (m and g): return None, None, None
    return float(m.group(3)), float(m.group(1)), (64.0 / float(g.group(1))) / 63 * 1000

def llama(p):
    t = read(p)
    ev = re.search(r'eval time =\s+([\d.]+) ms /\s+(\d+) runs', t)
    sa = re.search(r'sampling time =\s+([\d.]+) ms', t)
    pp = re.search(r'prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens.*?([\d.]+) tokens per second', t)
    if not ev: return None
    runs = int(ev.group(2))
    return {'forward': float(ev.group(1)) / runs,
            'loop': (float(ev.group(1)) + (float(sa.group(1)) if sa else 0.0)) / runs,
            'sampling_per_token': (float(sa.group(1)) / (runs + 1)) if sa else None,
            'prefill_tps': float(pp.group(3)) if pp else None, 'prompt_tokens': int(pp.group(2)) if pp else None}

def mlx(p):
    t = read(p)
    g = re.search(r'Generation: (\d+) tokens, ([\d.]+) tokens-per-sec', t)
    q = re.search(r'Prompt: (\d+) tokens, ([\d.]+) tokens-per-sec', t)
    if not g: return None
    n = int(g.group(1))   # mlx-lm reports n output tokens over the time of n-1 decode forwards
    return {'loop': 1000 * n / ((n - 1) * float(g.group(2))) if n > 1 else None, 'prefill_tps': float(q.group(2)) if q else None,
            'prompt_tokens': int(q.group(1)) if q else None, 'gen_tokens': n}

def tail_text(x): return x.split('</think>')[-1].strip()

R['headline'] = {}
for f in ['Q4', 'Q40', 'UD', 'Q8']:
    g = [ds4_greedy(p) for p in sorted(glob.glob(f'{L}/ds4_greedy_{f}_*.log'))]
    if not g: continue
    s = [ds4_sampled(p) for p in sorted(glob.glob(f'{L}/ds4_sampled_{f}_*.log'))]
    lg = [x for x in (llama(p) for p in sorted(glob.glob(f'{L}/llama_greedy_{f}_*.log'))) if x]
    lm = [x for x in (llama(p) for p in sorted(glob.glob(f'{L}/llama_sampled_matched_{f}_*.log'))) if x]
    ld = [x for x in (llama(p) for p in sorted(glob.glob(f'{L}/llama_sampled_default_{f}_*.log'))) if x]
    outs = [tail_text(read(p)) for p in sorted(glob.glob(f'{L}/ds4_greedy_{f}_*.out'))]
    louts = [tail_text(read(p)) for p in sorted(glob.glob(f'{L}/llama_greedy_{f}_*.out'))]
    same = (outs[0] == louts[0]) if outs and louts else None
    R['headline'][f] = {
        'ds4_greedy_forward': mean_sd([x[0] for x in g]), 'ds4_greedy_loop': mean_sd([x[1] for x in g]),
        'ds4_sampled_forward': mean_sd([x[0] for x in s]), 'ds4_sampled_sampler': mean_sd([x[1] for x in s]),
        'ds4_sampled_loop': mean_sd([x[2] for x in s]),
        'llama_greedy_forward': mean_sd([x['forward'] for x in lg]), 'llama_greedy_loop': mean_sd([x['loop'] for x in lg]),
        'llama_matched_forward': mean_sd([x['forward'] for x in lm]), 'llama_matched_loop': mean_sd([x['loop'] for x in lm]),
        'llama_matched_sampler': mean_sd([x['sampling_per_token'] for x in lm]),
        'llama_default_loop': mean_sd([x['loop'] for x in ld]), 'llama_default_sampler': mean_sd([x['sampling_per_token'] for x in ld]),
        'prompt_tokens_llama': lg[0]['prompt_tokens'] if lg else None,
        'greedy_text_identical': same, 'ds4_runs_identical': (len(set(outs)) == 1) if outs else None,
    }
mx = [x for x in (mlx(p) for p in sorted(glob.glob(f'{L}/mlx_short_*.log'))) if x]
if mx:
    R['headline']['MLX'] = {'mlx_loop': mean_sd([x['loop'] for x in mx]), 'prompt_tokens': mx[0]['prompt_tokens'],
                            'gen_tokens': sorted(set(x['gen_tokens'] for x in mx))}

# ---------------- prefill on the 1439-token prompt ----------------
R['prefill_1439'] = {}
for f in ['Q4', 'Q40', 'UD', 'Q8']:
    d = glob.glob(f'{L}/ds4_prefill_{f}.log'); l = glob.glob(f'{L}/llama_prefill_{f}.log')
    if d:
        m = re.search(r'prefill: ([\d.]+) t/s', read(d[0])); lm_ = llama(l[0]) if l else None
        R['prefill_1439'][f] = {'ds4': float(m.group(1)) if m else None, 'llama': lm_['prefill_tps'] if lm_ else None}

# ---------------- context sweep ----------------
R['context'] = {}
for n in ['1k', '4k', '8k', '16k', '32k']:
    d = []; dp = []; l = []; lp = []; m = []; mp = []; mg = []; mt = []; ntok = None
    for p in sorted(glob.glob(f'{L}/ds4_ctx_{n}_*.log')):
        t = read(p); g = re.search(r'prefill: ([\d.]+) t/s, generation: ([\d.]+) t/s', t)
        if g: d.append((64.0 / float(g.group(2))) / 63 * 1000); dp.append(float(g.group(1)))
    for p in sorted(glob.glob(f'{L}/llama_ctx_{n}_*.log')):
        x = llama(p)
        if x: l.append(x['forward']); lp.append(x['prefill_tps']); ntok = x['prompt_tokens']
    for p in sorted(glob.glob(f'{L}/mlx_ctx_{n}_*.log')):
        x = mlx(p)
        if x: m.append(x['loop']); mp.append(x['prefill_tps']); mg.append(x['gen_tokens']); mt.append(x['prompt_tokens'])
    if d or l:
        R['context'][n] = {'prompt_tokens': ntok, 'ds4_loop': mean_sd(d), 'ds4_prefill': mean_sd(dp),
                           'llama_forward': mean_sd(l), 'llama_prefill': mean_sd(lp), 'mlx_loop': mean_sd(m), 'mlx_prefill': mean_sd(mp),
                           'mlx_prompt_tokens': sorted(set(mt)), 'mlx_gen_tokens': sorted(set(mg))}

# ---------------- ablation ----------------
ABL = [('baseline', 'all optimizations on'), ('no_fused_swiglu', 'no fused gate/up SwiGLU'),
       ('no_multi_proj', 'no multi-weight projections'), ('no_fast_mv', 'no vectorized single-weight matvec'),
       ('no_ksplit', 'no K-split (FFN down)'), ('fa_nwg32', 'fixed 32 split-KV workgroups'),
       ('no_fused_norm', 'no fused residual+norm'), ('multi_nr0_2', 'multi-weight kernel, 2 rows/simdgroup'),
       ('q4k_classic', 'ggml-style Q4_K matvecs, generic fusions'), ('q40_classic', 'Q4_0: ggml-style matvecs, generic fusions'), ('all_off', 'all off')]
R['ablation'] = {}
for key, label in ABL:
    v = []
    for p in sorted(glob.glob(f'{L}/abl_{key}_*.log')):
        g = re.search(r'generation: ([\d.]+) t/s', read(p))
        if g: v.append((64.0 / float(g.group(1))) / 63 * 1000)
    if v: R['ablation'][key] = {'label': label, 'loop': mean_sd(v)}

# ---------------- stage profiles ----------------
def stages(path):
    gpu = None; agg = collections.defaultdict(list); toks = set(); positions = []
    for line in open(path, errors='replace'):
        m = re.search(r'gpu span ([\d.]+) ms', line)
        if m: gpu = float(m.group(1)); continue
        m = re.search(r'part=(\w+) layer=(\d+) pos=(\d+) tokens=1 (\w+)=([\d.]+) ms', line)
        if m and gpu is not None:
            pos = int(m.group(3)); st_ = m.group(4); il = int(m.group(2))
            positions.append(pos)
            if len(positions) > 0 and pos >= min(positions) + 1:   # skip the first decode position (pipeline warm)
                agg[(il, st_)].append(gpu); toks.add(pos)
            gpu = None
    n = len(toks)
    per = {}
    for (il, st_), v in agg.items():
        kind = 'attn' if il % 4 == 3 else 'gdn'
        per.setdefault(st_, {}).setdefault(kind, []).extend(v)
    res = {'positions': [min(positions), max(positions)] if positions else None, 'tokens': n, 'stages': {}}
    for st_, kinds in per.items():
        for kind, v in kinds.items():
            res['stages'][f'{kind}:{st_}'] = {'us_per_layer': 1000 * st.mean(v), 'ms_per_token': sum(v) / n}
    res['total_ms_per_token'] = sum(x['ms_per_token'] for x in res['stages'].values())
    return res
R['stage'] = {}
for key in ['stage_after', 'stage_alloff', 'stage_after_8k', 'stage_q40', 'stage_dsv4']:
    p = f'{L}/{key}.log'
    if os.path.exists(p): R['stage'][key] = stages(p)

# ---------------- perplexity (with paired chunk differences) ----------------
def chunk_nll(path):
    vals = [float(v) for _, v in re.findall(r'\[(\d+)\]([\d.]+)', read(path))]
    nll = []; prev = 0.0
    for k, ppl in enumerate(vals, start=1):
        cum = k * math.log(ppl); nll.append(cum - prev); prev = cum
    return nll
R['ppl'] = {}
nll = {}
for f in ['Q4', 'Q40', 'UD', 'Q8']:
    p = f'{L}/ppl_{f}.log'
    if os.path.exists(p):
        m = re.search(r'Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)', read(p))
        nll[f] = chunk_nll(p)
        R['ppl'][f] = {'ppl': float(m.group(1)) if m else None, 'se': float(m.group(2)) if m else None, 'chunks': len(nll[f])}
p = f'{L}/mlx_ppl.log'
if os.path.exists(p):
    m = re.search(r'Final estimate: PPL = ([\d.]+) over (\d+) tokens', read(p))
    if m: R['ppl']['MLX'] = {'ppl': float(m.group(1)), 'scored_tokens': int(m.group(2))}
R['ppl_paired'] = {}
for a, b in [('UD', 'Q4'), ('Q8', 'Q4'), ('Q40', 'Q4'), ('Q40', 'Q8')]:
    if a in nll and b in nll and len(nll[a]) == len(nll[b]) > 1:
        d = [x - y for x, y in zip(nll[a], nll[b])]
        mu = st.mean(d); se = st.stdev(d) / math.sqrt(len(d))
        R['ppl_paired'][f'{a}-{b}'] = {'mean_nll_diff': mu, 'se': se, 'ppl_ratio': math.exp(mu), 'chunks': len(d)}

# ---------------- correctness ----------------
R['correctness'] = {}
for p in sorted(glob.glob(f'{L}/cmp_*.txt')):
    t = read(p); m = re.search(r'max\|diff\|=([\d.]+) mean\|diff\|=([\d.]+) KL\(ds4\|\|ref\)=([\d.e+-]+) argmax_match=(\w+)', t)
    key = os.path.basename(p)[4:-4]
    if m: R['correctness'][key] = {'max': float(m.group(1)), 'mean': float(m.group(2)), 'kl': float(m.group(3)), 'argmax': m.group(4) == 'True'}
    else: R['correctness'][key] = {'error': t.strip()[:200]}

# ---------------- sampler built with -fno-finite-math-only ----------------
hm = []
for p in sorted(glob.glob(f'{L}/ds4_sampled_Q4_honormath_*.log')):
    m_ = re.search(r'sample ([\d.]+) ms', read(p))
    if m_: hm.append(float(m_.group(1)))
if hm: R['sampler_honormath'] = mean_sd(hm)

# ---------------- DeepSeek ----------------
d = glob.glob(f'{L}/ds4_dsv4.log'); l = glob.glob(f'{L}/llama_dsv4.log')
if d:
    g = re.search(r'generation: ([\d.]+) t/s', read(d[0])); x = llama(l[0]) if l else None
    R['dsv4'] = {'ds4_loop_ms': (64.0 / float(g.group(1))) / 63 * 1000 if g else None, 'llama_forward_ms': x['forward'] if x else None}

with open(os.path.join(OUT, 'results.json'), 'w') as f: json.dump(R, f, indent=1)

# ---------------- print ----------------
def rate(m): return 'n/a' if m is None else f"{1000/m['mean']:.1f}"
print('== headline (ms per decode forward; rate in t/s)')
for f, h in R['headline'].items():
    if f == 'MLX': print(f"  MLX  loop {fmt(h['mlx_loop'])} -> {rate(h['mlx_loop'])} t/s, prompt {h['prompt_tokens']} tokens"); continue
    print(f"  {f:4s} ds4 fwd {fmt(h['ds4_greedy_forward'])} ({rate(h['ds4_greedy_forward'])})  loop {fmt(h['ds4_greedy_loop'])} | sampled: fwd {fmt(h['ds4_sampled_forward'])} sampler {fmt(h['ds4_sampled_sampler'])} loop {fmt(h['ds4_sampled_loop'])} ({rate(h['ds4_sampled_loop'])})")
    print(f"       llama fwd {fmt(h['llama_greedy_forward'])} ({rate(h['llama_greedy_forward'])}) loop {fmt(h['llama_greedy_loop'])} | matched sampler {fmt(h['llama_matched_sampler'])} loop {fmt(h['llama_matched_loop'])} ({rate(h['llama_matched_loop'])}) | default sampler {fmt(h['llama_default_sampler'])} loop {fmt(h['llama_default_loop'])}")
    print(f"       greedy text identical: {h['greedy_text_identical']}  prompt tokens {h['prompt_tokens_llama']}")
print('== prefill 1439:', {k: v for k, v in R['prefill_1439'].items()})
print('== context (loop ms ds4 / fwd ms llama / loop ms mlx; prefill t/s)')
for n, c in R['context'].items():
    print(f"  {n:4s} {c['prompt_tokens']} tok: ds4 {fmt(c['ds4_loop'])} ({rate(c['ds4_loop'])}) llama {fmt(c['llama_forward'])} ({rate(c['llama_forward'])}) mlx {fmt(c['mlx_loop'])} ({rate(c['mlx_loop'])}) | prefill {fmt(c['ds4_prefill'],0)} / {fmt(c['llama_prefill'],0)} / {fmt(c['mlx_prefill'],0)}")
print('== ablation (loop ms per forward)')
base = R['ablation'].get('baseline', {}).get('loop')
for k, a in R['ablation'].items():
    d = a['loop']['mean'] - base['mean'] if base else 0
    print(f"  {a['label']:45s} {fmt(a['loop'])} ({rate(a['loop'])} t/s)  delta {d:+.2f} ms")
for k, s in R['stage'].items():
    print(f"== {k}: positions {s['positions']}, total {s['total_ms_per_token']:.2f} ms/token")
    for st_, v in sorted(s['stages'].items(), key=lambda kv: -kv[1]['ms_per_token'])[:12]:
        print(f"     {st_:24s} {v['us_per_layer']:8.1f} us/layer {v['ms_per_token']:6.2f} ms/token")
print('== ppl', R['ppl']); print('== ppl paired', {k: f"{v['mean_nll_diff']:+.4f}±{v['se']:.4f} (x{v['ppl_ratio']:.4f})" for k, v in R['ppl_paired'].items()})
print('== correctness', R['correctness']); print('== dsv4', R.get('dsv4'))
