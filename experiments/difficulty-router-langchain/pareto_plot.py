"""Accuracy vs cost-per-successful-task Pareto chart.

Run inside the notebook with `%run -i pareto_plot.py` after the results cell (needs `rows`).
"""

import matplotlib.pyplot as plt

pts = sorted(rows, key=lambda r: (r["per_success"], -r["acc"]))
frontier, best = [], -1.0
for r in pts:
    if r["acc"] > best:
        frontier.append(r)
        best = r["acc"]

fig, ax = plt.subplots(figsize=(8, 5.5))
for r in pts:
    on = r in frontier
    ax.errorbar(r["per_success"], r["acc"], yerr=[[r["acc"] - r["acc_lo"]], [r["acc_hi"] - r["acc"]]],
                fmt="o", ms=9 if on else 6, capsize=4, color="tab:red" if on else "tab:gray")
    ax.annotate(f"{r['policy']}\n{r['acc']:.1%} @ ${r['per_success']:.4f}", (r["per_success"], r["acc"]),
                textcoords="offset points", xytext=(8, -4), fontsize=9)
ax.step([r["per_success"] for r in frontier], [r["acc"] for r in frontier], where="post",
        color="tab:red", alpha=0.6, label="Pareto frontier (nothing is both cheaper and more accurate)")
ax.set_xscale("log")
ax.set_xlabel("$ per successful task (log, lower is better)")
ax.set_ylabel("accuracy (higher is better)")
bench = globals().get("BENCH_NAME", "Mixed GSM8K / MATH L3 / AIME")
ax.set_title(f"Pareto: accuracy vs cost per success\n{bench}, {N_QUESTIONS} questions x {N_SAMPLES} draws")
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig("fig_pareto.png", dpi=150)
plt.show()

print("on the frontier:", [r["policy"] for r in frontier])
print("dominated:", [r["policy"] for r in pts if r not in frontier])
