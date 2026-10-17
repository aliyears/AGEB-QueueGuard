import sys, json, time, numpy as np, requests
base = "/home/ibm-user/queueguard/"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
fc = json.load(open(base+"artifacts/feature_contract.json"))
i2 = fc["i2"]
X = np.load(base+"data/X_full.npy", mmap_mode="r")[i2:i2+N]
y = np.load(base+"data/y.npy")[i2:i2+N]
url = f"http://127.0.0.1:{PORT}/score"
s = requests.Session()
mode = s.get(url.replace("/score", "/")).json()["mode"]
for i in range(30):                                   # isinma
    s.post(url, json={"id": -1, "x": [None if np.isnan(v) else float(v) for v in X[i]]})
lat, res = [], []
t_all = time.perf_counter()
for i in range(N):
    body = {"id": i, "x": [None if np.isnan(v) else float(v) for v in X[i]]}
    t0 = time.perf_counter()
    r = s.post(url, json=body).json()
    lat.append((time.perf_counter()-t0)*1000); res.append(r)
total = time.perf_counter()-t_all
lat = np.array(lat)
dec = np.array([r["decision"] for r in res])
mdl = np.array([r["model_ms"] for r in res])
print(f"\n=== SIM mod={mode} N={N} ===")
print(f"Uctan uca (ms): ort={lat.mean():.2f} p50={np.percentile(lat,50):.2f} p95={np.percentile(lat,95):.2f} p99={np.percentile(lat,99):.2f}")
print(f"Sadece model (ms): ort={mdl.mean():.2f} p50={np.percentile(mdl,50):.2f} p95={np.percentile(mdl,95):.2f} p99={np.percentile(mdl,99):.2f}")
print(f"Islem/sn (HTTP dahil): {N/total:.0f}")
print(f"Ikinci modele giden: %{np.mean([r['routed'] for r in res])*100:.1f}")
for k in ["normal", "ek_dogrulama_onerisi", "incelemeye_yonlendir"]:
    m = dec == k
    print(f"  {k:22s} adet={m.sum():5d} icindeki hirsiz={int(y[m].sum())}")
print(f"Toplam hirsiz (bu {N} islemde): {int(y.sum())}")
print("BITTI")
