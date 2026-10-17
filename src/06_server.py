import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
import json, time, joblib, numpy as np
from http.server import BaseHTTPRequestHandler, HTTPServer

base = "/home/ibm-user/queueguard/"
MODE = os.environ.get("MODE", "casc")          # casc | adv | light
PORT = int(os.environ.get("PORT", "8000"))
T = 0.0166                                      # dogrulamada secildi, sabit
fc = json.load(open(base+"artifacts/feature_contract.json"))
li = fc["light_idx"]
ml = joblib.load(base+"artifacts/model_light.joblib")
ma = joblib.load(base+"artifacts/model_adv.joblib")

ref = np.load(base+("artifacts/score_val_light.npy" if MODE == "light" else "artifacts/score_val_adv.npy"))
T_HIGH, T_UNC = np.quantile(ref, 0.98), np.quantile(ref, 0.95)
log = open(base+f"results/log_{MODE}.jsonl", "w")
ma.predict_proba(np.zeros((1, 392), dtype=np.float32)); ml.predict_proba(np.zeros((1, len(li)), dtype=np.float32))

def decide(s):
    if s >= T_HIGH: return "incelemeye_yonlendir"
    if s >= T_UNC: return "ek_dogrulama_onerisi"
    return "normal"

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    wbufsize = 65536
    disable_nagle_algorithm = True
    def log_message(self, *a): pass
    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b); self.wfile.flush()
    def do_GET(self):
        self._send({"ok": True, "mode": MODE, "t": T})
    def do_POST(self):
        t_in = time.perf_counter()
        d = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        x = np.array([np.nan if v is None else v for v in d["x"]], dtype=np.float32).reshape(1, -1)
        t0 = time.perf_counter()
        s1 = None; routed = False
        if MODE == "adv":
            s = ma.predict_proba(x)[0, 1]; used = "adv"; reason = "sadece gelismis"
        else:
            s1 = ml.predict_proba(x[:, li])[0, 1]
            if MODE == "light":
                s = s1; used = "light"; reason = "sadece hafif"
            elif s1 < T:
                s = None; used = "light"; reason = f"s1={s1:.4f} < {T}: bypass"
            else:
                s = ma.predict_proba(x)[0, 1]; used = "light+adv"; routed = True
                reason = f"s1={s1:.4f} >= {T}: gelismise yonlendirildi"
        model_ms = (time.perf_counter()-t0)*1000
        decision = "normal" if s is None else decide(s)
        out = {"id": d["id"], "s1": None if s1 is None else float(s1), "s2": None if s is None else float(s),
               "used": used, "routed": routed, "decision": decision, "reason": reason,
               "model_ms": round(model_ms, 3)}
        out["server_ms"] = round((time.perf_counter()-t_in)*1000, 3)
        log.write(json.dumps(out)+"\n"); log.flush()
        self._send(out)

print(f"MODE={MODE} PORT={PORT} T_HIGH={T_HIGH:.4f} T_UNC={T_UNC:.4f}", flush=True)
HTTPServer(("127.0.0.1", PORT), H).serve_forever()
