import sys, struct, numpy as np
a = np.fromfile(sys.argv[1], dtype=np.float32)
k = int(sys.argv[2]) if len(sys.argv) > 2 else 10
idx = np.argsort(-a)[:k]
print("top%d:" % k, " ".join("%d:%.4f" % (i, a[i]) for i in idx))
print("n=%d finite=%d max=%.4f min=%.4f" % (a.size, np.isfinite(a).sum(), a.max(), a.min()))
