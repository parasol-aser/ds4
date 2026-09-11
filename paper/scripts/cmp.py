import sys, numpy as np
a = np.fromfile(sys.argv[1], dtype=np.float32); b = np.fromfile(sys.argv[2], dtype=np.float32)
if a.size == 0 or a.size != b.size or not (np.isfinite(a).all() and np.isfinite(b).all()):
    print("cmp: bad dump (sizes %d/%d, finite %s/%s)" % (a.size, b.size, np.isfinite(a).all(), np.isfinite(b).all())); sys.exit(2)
ia = np.argsort(-a)[:5]; ib = np.argsort(-b)[:5]
print("ds4 top5:", " ".join("%d:%.4f" % (i, a[i]) for i in ia))
print("ref top5:", " ".join("%d:%.4f" % (i, b[i]) for i in ib))
d = np.abs(a - b)
pa = np.exp(a - a.max()); pa /= pa.sum(); pb = np.exp(b - b.max()); pb /= pb.sum()
kl = float(np.sum(np.where(pa > 0, pa * (np.log(pa + 1e-30) - np.log(pb + 1e-30)), 0.0)))
print("max|diff|=%.4f mean|diff|=%.5f KL(ds4||ref)=%.2e argmax_match=%s" % (d.max(), d.mean(), kl, ia[0] == ib[0]))
sys.exit(0 if ia[0] == ib[0] else 1)
