import numpy as np, json
from sklearn.metrics import average_precision_score
base = "/home/ibm-user/queueguard/"
fc = json.load(open(base+"artifacts/feature_contract.json"))
y = np.load(base+"data/y.npy")[fc["i2"]:]
sl = np.load(base+"artifacts/score_test_light.npy")
sa = np.load(base+"artifacts/score_test_adv.npy")
casc = np.where(sl >= 0.0166, 1+sa, sl)
def rec(yy, s, f):
    k = int(len(s)*f); top = np.argpartition(-s, k)[:k]
    return yy[top].sum()/yy.sum()
rng = np.random.default_rng(42); n = len(y); B = 200
d_pr, d5, d20 = [], [], []
for b in range(B):
    ix = rng.integers(0, n, n)
    yy, a, c = y[ix], sa[ix], casc[ix]
    d_pr.append(average_precision_score(yy, c) - average_precision_score(yy, a))
    d5.append(rec(yy, c, 0.05) - rec(yy, a, 0.05))
    d20.append(rec(yy, c, 0.20) - rec(yy, a, 0.20))
def ci(x): x = np.array(x); return f"ort={x.mean():+.4f}  %95 aralik=[{np.percentile(x,2.5):+.4f}, {np.percentile(x,97.5):+.4f}]"
print("Fark = kaskad - gelismis  (B=200 bootstrap)")
print("PR-AUC      :", ci(d_pr))
print("recall @ %5 :", ci(d5))
print("recall @ %20:", ci(d20))
print("BITTI")
