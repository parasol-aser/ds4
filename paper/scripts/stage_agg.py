import re, collections, sys
L=sys.argv[1]
gpu=None; agg=collections.defaultdict(list); toks=set()
for line in open(L):
    m=re.search(r"gpu span ([\d.]+) ms", line)
    if m: gpu=float(m.group(1)); continue
    m=re.search(r"part=qwen layer=(\d+) pos=(\d+) tokens=1 (\w+)=([\d.]+) ms", line)
    if m and gpu is not None:
        il=int(m.group(1)); pos=int(m.group(2)); st=m.group(3)
        if pos>=8:
            kind='gdn' if il%4!=3 else 'attn'
            agg[(kind,st)].append(gpu); toks.add(pos)
        gpu=None
ntok=len(toks); tot=0
print("kind  stage            n/tok  mean_us  per_token_us")
for (k,st),v in sorted(agg.items(), key=lambda kv:-sum(kv[1])):
    per_tok=sum(v)/ntok; tot+=per_tok
    print(f"{k:5s} {st:16s} {len(v)/ntok:5.1f} {1000*sum(v)/len(v):8.1f} {1000*per_tok:10.0f}")
print("total layer-stage GPU per token: %.2f ms" % tot)
