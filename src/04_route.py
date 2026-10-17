import numpy as np, json
from sklearn.metrics import average_precision_score
base = "/home/ibm-user/queueguard/"
fc = json.load(open(base+"artifacts/feature_contract.json"))
y = np.load(base+"data/y.npy")[fc["i1"]:fc["i2"]]
sl = np.load(base+"artifacts/score_val_light.npy")
sa = np.load(base+"artifacts/score_val_adv.npy")
L1, A1 = 1.99, 4.55                          # tek islem ms (olculdu)
Lb, Ab = 1e6/508338, 1e6/96040               # toplu: mikrosaniye/islem (olculdu)

def rec(s, frac=0.02):
    k = int(len(s)*frac); top = np.argsort(-s)[:k]
    return y[top].sum()/y.sum()

print(f"Sadece HAFIF : PR-AUC={average_precision_score(y,sl):.4f} recall@2%={rec(sl):.3f}")
print(f"Sadece GELISMIS: PR-AUC={average_precision_score(y,sa):.4f} recall@2%={rec(sa):.3f}\n")
print("atla%  esik    PR-AUC  recall@2%  kacan_hirsiz%  tek_islem_ms  toplu_us")
rows = []
for p in [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    t = np.quantile(sl, p) if p > 0 else -1
    routed = sl >= t
    final = np.where(routed, 1+sa, sl)       # yonlendirilenler hep ustte, atlananlar altta
    missed = y[~routed].sum()/y.sum()*100
    single = L1 + (1-p)*A1
    batch = Lb + (1-p)*Ab
    r = (p*100, t, average_precision_score(y,final), rec(final), missed, single, batch)
    rows.append(r)
    print(f"{r[0]:4.0f}  {r[1]:.4f}  {r[2]:.4f}  {r[3]:.3f}      {r[4]:5.1f}          {r[5]:.2f}         {r[6]:.2f}")
np.savetxt(base+"artifacts/pareto_val.csv", np.array(rows), delimiter=",",
           header="skip_pct,threshold,pr_auc,recall_at_2pct,missed_fraud_pct,single_ms,batch_us")
print("\nBITTI")
