import re, glob, os, collections
import os; L=os.environ.get('OUT', 'out')+'/logs'
def ds4_gen(path):
    try: t=open(path).read()
    except FileNotFoundError: return None
    m=re.findall(r'prefill: ([\d.]+) t/s, generation: ([\d.]+) t/s', t)
    return (float(m[-1][0]), float(m[-1][1])) if m else None
def llama_gen(path):
    try: t=open(path).read()
    except FileNotFoundError: return None
    pp=re.search(r'prompt eval time =\s+([\d.]+) ms /\s+(\d+) tokens.*?([\d.]+) tokens per second', t)
    tg=re.search(r'\n[^\n]*\beval time =\s+([\d.]+) ms /\s+(\d+) runs.*?([\d.]+) tokens per second', t)
    return (float(pp.group(3)) if pp else None, int(pp.group(2)) if pp else None, float(tg.group(3)) if tg else None)
print('== context sweep (prefill t/s, decode t/s)')
for n in ['1k','4k','8k','16k','32k']:
    d=ds4_gen(f'{L}/ds4_ctx_{n}.log'); l=llama_gen(f'{L}/llama_ctx_{n}.log')
    print(f'{n:4s} ds4 {d} llama {l}')
print('== ablations')
for f in sorted(glob.glob(f'{L}/abl_*_1.log')):
    name=os.path.basename(f)[4:-6]
    r=[ds4_gen(f'{L}/abl_{name}_{i}.log') for i in (1,2)]
    r=[x[1] for x in r if x]
    print(f'{name:18s} {sum(r)/len(r) if r else None:.2f}' if r else f'{name} n/a')
print('== sampled')
for f in ['Q4','UD','Q8']:
    d=ds4_gen(f'{L}/ds4_sampled_{f}.log'); ls=llama_gen(f'{L}/llama_sampled_{f}.log'); lg=llama_gen(f'{L}/llama_greedy_{f}.log')
    print(f'{f}: ds4 sampled {d} llama sampled {ls} llama greedy {lg}')
print('== ppl')
for f in ['Q4','UD','Q8']:
    try: print(f, re.findall(r'Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)', open(f'{L}/ppl_{f}.log').read()))
    except FileNotFoundError: print(f, 'n/a')
print('== dsv4')
print('ds4', ds4_gen(f'{L}/ds4_dsv4.log'), 'llama', llama_gen(f'{L}/llama_dsv4.log'))
