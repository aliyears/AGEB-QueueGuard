import pandas as pd, numpy as np, json, time
t0 = time.time()
base = "/home/ibm-user/queueguard/"
df = pd.read_pickle(base+"data/clean.pkl")
sp = json.load(open(base+"artifacts/split.json"))
i1, i2 = sp["i1"], sp["i2"]

cat_cols = ["ProductCD","card4","card6","P_emaildomain","R_emaildomain"] + [f"M{i}" for i in range(1,10)]
cat_maps = {}
for c in cat_cols:
    cats = sorted(df.iloc[:i1][c].dropna().unique().tolist())   # sadece egitimden ogren
    cat_maps[c] = cats
    df[c] = pd.Categorical(df[c], categories=cats).codes.astype("float32")
    df.loc[df[c] < 0, c] = np.nan                               # bos veya gorulmemis -> NaN

df["hour"] = ((df.TransactionDT // 3600) % 24).astype("float32")

drop = ["TransactionID","isFraud","TransactionDT"]
full = [c for c in df.columns if c not in drop]
light = (["TransactionAmt","hour","ProductCD","card4","card6","card1","card2","addr1",
          "P_emaildomain","M4","M6","D1","D4","D10","D15"] + [f"C{i}" for i in range(1,15)])
assert all(c in full for c in light)
light_idx = [full.index(c) for c in light]
cat_idx_full = [full.index(c) for c in cat_cols]
cat_idx_light = [light.index(c) for c in light if c in cat_cols]

X = df[full].to_numpy(dtype="float32")
y = df["isFraud"].to_numpy(dtype="int8")
np.save(base+"data/X_full.npy", X)
np.save(base+"data/y.npy", y)

json.dump({"full":full,"light":light,"light_idx":light_idx,
           "cat_idx_full":cat_idx_full,"cat_idx_light":cat_idx_light,
           "cat_maps":cat_maps,"i1":i1,"i2":i2},
          open(base+"artifacts/feature_contract.json","w"), indent=1)

print("X sekli:", X.shape, "| hafif sutun:", len(light), "| gelismis sutun:", len(full))
print("Metin sutunlari (kategori):", len(cat_idx_full))
print("Egitim hirsizlik:", int(y[:i1].sum()), "| val:", int(y[i1:i2].sum()), "| test:", int(y[i2:].sum()))
print("Sure (sn):", round(time.time()-t0,1))
