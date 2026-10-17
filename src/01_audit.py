import pandas as pd, numpy as np, json, time
t0 = time.time()
path = "/home/ibm-user/queueguard/data/train_transaction.csv"

cols = pd.read_csv(path, nrows=0).columns.tolist()
cat_cols = ["ProductCD","card4","card6","P_emaildomain","R_emaildomain"] + [f"M{i}" for i in range(1,10)]
dtypes = {}
for c in cols:
    if c in cat_cols: dtypes[c] = "object"
    elif c == "TransactionID": dtypes[c] = "int32"
    elif c == "isFraud": dtypes[c] = "int8"
    elif c == "TransactionDT": dtypes[c] = "int64"
    else: dtypes[c] = "float32"

df = pd.read_csv(path, dtype=dtypes)
print("Okuma suresi (sn):", round(time.time()-t0,1))
print("Satir, sutun:", df.shape)
print("RAM (GB):", round(df.memory_usage(deep=True).sum()/1e9,2))
print("Hirsizlik orani:", round(df.isFraud.mean()*100,2), "%  | adet:", int(df.isFraud.sum()))
print("Zamana gore sirali mi:", bool(df.TransactionDT.is_monotonic_increasing))
print("TransactionDT aralik (gun):", round(df.TransactionDT.min()/86400,1), "-", round(df.TransactionDT.max()/86400,1))

if not df.TransactionDT.is_monotonic_increasing:
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    print("-> Zamana gore siralandi.")

n = len(df)
i1, i2 = int(n*0.70), int(n*0.85)
parts = {"train": df.iloc[:i1], "val": df.iloc[i1:i2], "test": df.iloc[i2:]}
print("\nBolmeler:")
for k, p in parts.items():
    print(f"  {k:5s} satir={len(p):7d}  hirsizlik={int(p.isFraud.sum()):5d}  oran={p.isFraud.mean()*100:.2f}%  gun {p.TransactionDT.min()/86400:.0f}-{p.TransactionDT.max()/86400:.0f}")

miss = (df.isna().mean()*100).sort_values(ascending=False)
print("\nSutunlarin %50'den fazlasi bos olan sayisi:", int((miss>50).sum()), "/", len(miss))
print("Hic bos olmayan sutun sayisi:", int((miss==0).sum()))
print("En cok bos 8 sutun:\n", miss.head(8).round(1).to_string())
print("\nMetin sutunlari ve kac farkli deger:")
for c in cat_cols: print(f"  {c}: {df[c].nunique()} deger, %{df[c].isna().mean()*100:.0f} bos")

df.to_pickle("/home/ibm-user/queueguard/data/clean.pkl")
json.dump({"n_train":i1,"n_val":i2-i1,"n_test":n-i2,"i1":i1,"i2":i2}, open("/home/ibm-user/queueguard/artifacts/split.json","w"))
print("\nKaydedildi. Toplam sure (sn):", round(time.time()-t0,1))
