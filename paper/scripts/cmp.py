import sys, numpy as np
a = np.fromfile(sys.argv[1], dtype=np.float32); b = np.fromfile(sys.argv[2], dtype=np.float32)
ia = np.argsort(-a)[:5]; ib = np.argsort(-b)[:5]
print("ds4 top5:", " ".join("%d:%.4f" % (i, a[i]) for i in ia))
print("ref top5:", " ".join("%d:%.4f" % (i, b[i]) for i in ib))
d = np.abs(a - b); print("max|diff|=%.4f mean|diff|=%.5f argmax_match=%s" % (d.max(), d.mean(), ia[0] == ib[0]))
