import numpy as np, json
base = "/home/ibm-user/queueguard/"
fc = json.load(open(base+"artifacts/feature_contract.json"))
y = np.load(base+"data/y.npy")[fc["i2"]:]
sl = np.load(base+"artifacts/score_test_light.npy")
sa = np.load(base+"artifacts/score_test_adv.npy")
T = 0.0166
casc = np.where(sl >= T, 1+sa, sl)
def rec(s, f):
    k = int(len(s)*f); top = np.argsort(-s)[:k]
    return y[top].sum()/y.sum()
print("butce  hafif  gelismis  kaskad  fark(kaskad-gelismis)")
for f in [0.01, 0.02, 0.05, 0.10, 0.20]:
    a, b, c = rec(sl, f), rec(sa, f), rec(casc, f)
    print(f"%{f*100:<4.0f} {a:.3f}  {b:.3f}     {c:.3f}   {c-b:+.3f}")
print("BITTI")
