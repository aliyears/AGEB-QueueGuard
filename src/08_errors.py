import numpy as np, pandas as pd, json
base = "/home/ibm-user/queueguard/"
fc = json.load(open(base+"artifacts/feature_contract.json"))
i2 = fc["i2"]; T = 0.0166
df = pd.read_pickle(base+"data/clean.pkl").iloc[i2:].reset_index(drop=True)
sl = np.load(base+"artifacts/score_test_light.npy")
sa = np.load(base+"artifacts/score_test_adv.npy")
df["skipped"] = sl < T
df["sa"] = sa
top2 = sa >= np.quantile(sa, 0.98)
fr = df[df.isFraud == 1]
print("Test fraud:", len(fr), "| atlanan:", int(fr.skipped.sum()), f"(%{fr.skipped.mean()*100:.1f})")
sk = fr[fr.skipped]
print("Atlanan fraudlardan gelismisin en supheli %2'sine girenler:", int((sk.sa >= np.quantile(sa, 0.98)).sum()), "/", len(sk))
print("\nAtlanan fraud / tum fraud, ProductCD'ye gore:")
g = fr.groupby("ProductCD").agg(fraud=("isFraud","size"), atlanan=("skipped","sum"))
g["atlanan_%"] = (g.atlanan/g.fraud*100).round(1); print(g.to_string())
print("\ncard6'ya gore (kodlu):")
g = fr.groupby("card6").agg(fraud=("isFraud","size"), atlanan=("skipped","sum"))
g["atlanan_%"] = (g.atlanan/g.fraud*100).round(1); print(g.to_string())
print("\nTutar araligina gore:")
fr = fr.assign(amt=pd.cut(fr.TransactionAmt, [0,25,50,100,250,1000,1e9]))
g = fr.groupby("amt", observed=True).agg(fraud=("isFraud","size"), atlanan=("skipped","sum"))
g["atlanan_%"] = (g.atlanan/g.fraud*100).round(1); print(g.to_string())
print("\nAtlanan fraud tutar medyani:", round(sk.TransactionAmt.median(),1), "| yakalanan fraud tutar medyani:", round(fr[~fr.skipped].TransactionAmt.median(),1))
print("BITTI")
