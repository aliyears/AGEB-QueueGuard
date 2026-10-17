import numpy as np, json, time, joblib, os, psutil
from sklearn.metrics import average_precision_score
base = "/home/ibm-user/queueguard/"
fc = json.load(open(base+"artifacts/feature_contract.json"))
i2 = fc["i2"]
X = np.load(base+"data/X_full.npy")[i2:]
y = np.load(base+"data/y.npy")[i2:]
Xl = np.ascontiguousarray(X[:, fc["light_idx"]])
ml = joblib.load(base+"artifacts/model_light.joblib")
ma = joblib.load(base+"artifacts/model_adv.joblib")
sl = np.load(base+"artifacts/score_test_light.npy")
sa = np.load(base+"artifacts/score_test_adv.npy")
T = 0.0166
print("OMP_NUM_THREADS =", os.environ.get("OMP_NUM_THREADS", "ayarsiz"))

def rec(s, frac=0.02):
    k = int(len(s)*frac); top = np.argsort(-s)[:k]
    return y[top].sum()/y.sum(), y[top].sum()/k

print("\n--- TEST BASARISI (esik val'den secildi) ---")
for name, s in [("B hafif", sl), ("C gelismis", sa)]:
    r, p = rec(s)
    print(f"{name:11s} PR-AUC={average_precision_score(y,s):.4f} recall@2%={r:.3f} precision@2%={p:.3f}")
for t in [0.0136, 0.0166, 0.0219]:
    routed = sl >= t
    final = np.where(routed, 1+sa, sl)
    r, p = rec(final)
    tag = "ANA" if t == T else "duyarlilik"
    print(f"Kaskad t={t} ({tag}): atlanan=%{(~routed).mean()*100:.1f} PR-AUC={average_precision_score(y,final):.4f} "
          f"recall@2%={r:.3f} precision@2%={p:.3f} atlananda_hirsiz=%{y[~routed].sum()/y.sum()*100:.1f}")

def single(mode, N=3000):
    for i in range(20):
        ml.predict_proba(Xl[i:i+1]); ma.predict_proba(X[i:i+1])
    ts = []; routed = 0
    for i in range(N):
        t0 = time.perf_counter()
        if mode == "C":
            ma.predict_proba(X[i:i+1])
        else:
            a = ml.predict_proba(Xl[i:i+1])[0,1]
            if mode == "Casc" and a >= T:
                ma.predict_proba(X[i:i+1]); routed += 1
        ts.append((time.perf_counter()-t0)*1000)
    ts = np.array(ts)
    print(f"{mode:5s} ort={ts.mean():.2f}ms p50={np.percentile(ts,50):.2f} p95={np.percentile(ts,95):.2f} "
          f"p99={np.percentile(ts,99):.2f} | islem/sn={1000/ts.mean():.0f} | ikinci modele giden=%{routed/N*100:.1f}")

def batch(mode, n=20000):
    t0 = time.perf_counter()
    if mode == "C":
        ma.predict_proba(X[:n])
    else:
        s = ml.predict_proba(Xl[:n])[:,1]
        if mode == "Casc":
            ma.predict_proba(X[:n][s >= T])
    return n/(time.perf_counter()-t0)

print("\n--- GERCEK SURE: TEK ISLEM (3000 islem) ---")
for m in ["B", "C", "Casc"]: single(m)
print("\n--- GERCEK SURE: TOPLU (20000 islem, 3 deneme ortancasi) ---")
for m in ["B", "C", "Casc"]:
    print(f"{m:5s} islem/sn = {np.median([batch(m) for _ in range(3)]):.0f}")
print("\nBellek (MB):", round(psutil.Process().memory_info().rss/1e6))
print("BITTI")
