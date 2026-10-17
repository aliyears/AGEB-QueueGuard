import numpy as np, json, time, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

base = "/home/ibm-user/queueguard/"
fc = json.load(open(base+"artifacts/feature_contract.json"))
X = np.load(base+"data/X_full.npy")
y = np.load(base+"data/y.npy")
i1, i2 = fc["i1"], fc["i2"]
li = fc["light_idx"]

def recall_at_budget(y_true, s, frac=0.02):
    k = int(len(s)*frac)
    top = np.argsort(-s)[:k]
    tp = y_true[top].sum()
    return tp / y_true.sum(), tp / k

def bench(model, Xv):
    model.predict_proba(Xv[:5])                       # isinma
    t = time.perf_counter()
    for i in range(300):
        model.predict_proba(Xv[i:i+1])
    single_ms = (time.perf_counter()-t)/300*1000
    B = Xv[:20000]
    t = time.perf_counter()
    model.predict_proba(B)
    thr = len(B)/(time.perf_counter()-t)
    return single_ms, thr

configs = {
  "light": dict(cols=li, cat=fc["cat_idx_light"],
                params=dict(max_iter=60, max_depth=4, learning_rate=0.1)),
  "adv":   dict(cols=list(range(X.shape[1])), cat=fc["cat_idx_full"],
                params=dict(max_iter=300, max_leaf_nodes=63, learning_rate=0.08)),
}

for name, cfg in configs.items():
    Xs = np.ascontiguousarray(X[:, cfg["cols"]])
    Xtr, ytr = Xs[:i1], y[:i1]
    Xva, yva = Xs[i1:i2], y[i1:i2]
    Xte = Xs[i2:]
    m = HistGradientBoostingClassifier(categorical_features=cfg["cat"], early_stopping=False,
                                       random_state=42, **cfg["params"])
    t = time.time(); m.fit(Xtr, ytr); tr = time.time()-t
    sv = m.predict_proba(Xva)[:,1]
    st = m.predict_proba(Xte)[:,1]
    rec, prec = recall_at_budget(yva, sv)
    single_ms, thr = bench(m, Xva)
    print(f"\n=== {name.upper()} ({len(cfg['cols'])} sutun) ===")
    print(f"Egitim suresi (sn): {tr:.1f}")
    print(f"VAL PR-AUC: {average_precision_score(yva, sv):.4f} | ROC-AUC: {roc_auc_score(yva, sv):.4f}")
    print(f"VAL en supheli %2: recall={rec:.3f} precision={prec:.3f}")
    print(f"Tek islem suresi (ms): {single_ms:.2f} | Toplu hiz (islem/sn): {thr:.0f}")
    joblib.dump(m, base+f"artifacts/model_{name}.joblib")
    np.save(base+f"artifacts/score_val_{name}.npy", sv)
    np.save(base+f"artifacts/score_test_{name}.npy", st)
print("\nBITTI")
