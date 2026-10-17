import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
base = "/home/ibm-user/queueguard/"
d = np.loadtxt(base+"artifacts/pareto_val.csv", delimiter=",")
skip, rec, miss = d[:,0], d[:,3], d[:,4]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(skip, rec, "o-")
ax[0].axvline(60, ls="--", c="gray"); ax[0].text(61, rec.min(), "T=0.0166 (val: %60)", fontsize=8)
ax[0].set_xlabel("Atlanan islem % (dogrulama)"); ax[0].set_ylabel("Recall @ en supheli %2")
ax[0].set_title("Atlama arttikca recall@%2")
ax[1].plot(skip, miss, "o-", c="C3")
ax[1].set_xlabel("Atlanan islem % (dogrulama)"); ax[1].set_ylabel("Atlananlardaki fraud % (tum fraudlarin)")
ax[1].set_title("Kacirilan fraud payi")
plt.tight_layout(); plt.savefig(base+"results/pareto.png", dpi=150)
print("BITTI")
