import sys, numpy as np
a = np.fromfile(sys.argv[1], dtype=np.float32); b = np.fromfile(sys.argv[2], dtype=np.float32)
if a.size == 0 or a.size != b.size or not (np.isfinite(a).all() and np.isfinite(b).all()):
    print("cmp: bad dump (sizes %d/%d, finite %s/%s)" % (a.size, b.size, np.isfinite(a).all(), np.isfinite(b).all())); sys.exit(2)
a = a.astype(np.float64); b = b.astype(np.float64)
ia = np.argsort(-a)[:5]; ib = np.argsort(-b)[:5]
print("ds4 top5:", " ".join("%d:%.4f" % (i, a[i]) for i in ia))
print("ref top5:", " ".join("%d:%.4f" % (i, b[i]) for i in ib))
d = np.abs(a - b)
la = a - (a.max() + np.log(np.exp(a - a.max()).sum()))   # log-softmax in float64
lb = b - (b.max() + np.log(np.exp(b - b.max()).sum()))
kl = float(np.sum(np.exp(la) * (la - lb)))
print("max|diff|=%.4f mean|diff|=%.5f KL(ds4||ref)=%.2e argmax_match=%s" % (d.max(), d.mean(), kl, ia[0] == ib[0]))
sys.exit(0 if ia[0] == ib[0] else 1)
