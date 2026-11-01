"""Stats and Forge-themed charts for the FloorPlanCAD benchmark. Bootstrap resamples buildings."""

from __future__ import annotations

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw_utils as fv  # noqa: E402  (theme, palette, title helper)

from fp_common import MODELS, load_results  # noqa: E402

ALPHA = 0.05
NAME = {k: v["label"] for k, v in MODELS.items()}
NAME.update({"glm_enum": "GLM, list-then-count prompt", "glm_enum_vote": "GLM, list-then-count, majority of 5",
             "glm_sft": "GLM + vision LoRA (SFT on train buildings)"})
COLOR = {"glm": fv.CORAL, "muse": fv.MAGENTA, "kimi": fv.ORANGE, "gemini4": "#F2B880", "opus55": "#C75D7A",
         "gpt61": "#4F2A3A", "glm_enum": fv.PEACH, "glm_enum_vote": "#D9636B", "glm_sft": fv.INK}


def name(tag: str) -> str:
    return NAME.get(tag, tag)


def color(tag: str) -> str:
    return COLOR.get(tag, fv.INK)


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    n = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    return {"only_a": only_a, "only_b": only_b, "p": p}


def block_bootstrap(items: list[tuple[str, bool, bool]], B: int = 5000, seed: int = 42) -> dict:
    by = defaultdict(list)
    for img, a, b in items:
        by[img].append((a, b))
    keys = list(by)

    def delta(ks):
        pairs = [p for k in ks for p in by[k]]
        return (sum(a for a, _ in pairs) - sum(b for _, b in pairs)) / max(len(pairs), 1)

    obs = delta(keys)
    rng = random.Random(seed)
    s = sorted(delta([rng.choice(keys) for _ in keys]) for _ in range(B))
    p = min(1.0, 2 * sum(x <= 0 for x in s) / B) if obs >= 0 else min(1.0, 2 * sum(x >= 0 for x in s) / B)
    return {"delta": obs, "lo": s[int(0.025 * B)], "hi": s[int(0.975 * B) - 1], "p": p, "samples": s}


def compare(a: str, b: str, B: int = 5000) -> dict | None:
    A, Bm = load_results(a), load_results(b)
    ids = sorted(set(A) & set(Bm))
    if not ids:
        return None
    ac = [A[i]["score"] >= 1 for i in ids]
    bc = [Bm[i]["score"] >= 1 for i in ids]
    return {"a": a, "b": b, "n": len(ids), "acc_a": sum(ac) / len(ids), "acc_b": sum(bc) / len(ids),
            "mcnemar": {**mcnemar(ac, bc), "both_right": sum(x and y for x, y in zip(ac, bc)),
                        "both_wrong": sum(not x and not y for x, y in zip(ac, bc))}, "boot": block_bootstrap(units := [(A[i]["building"], x, y) for i, x, y in zip(ids, ac, bc)], B),
            "units": units}


def significance_table(pairs: list[tuple[str, str]], B: int = 5000) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        r = compare(a, b, B)
        if not r:
            continue
        mc_sig, bs_sig = r["mcnemar"]["p"] < ALPHA, r["boot"]["lo"] > 0 or r["boot"]["hi"] < 0
        d = 100 * (r["acc_a"] - r["acc_b"])
        rows.append({"pair": f"{name(a)} vs {name(b)}", "n": r["n"], "acc_a": f"{r['acc_a']:.1%}", "acc_b": f"{r['acc_b']:.1%}",
                     "Δpp": round(d, 1), "only_a": r["mcnemar"]["only_a"], "only_b": r["mcnemar"]["only_b"],
                     "McNemar p": round(r["mcnemar"]["p"], 4), "boot 95% CI": f"[{100*r['boot']['lo']:+.1f}, {100*r['boot']['hi']:+.1f}]",
                     "significant": mc_sig and bs_sig,
                     "notes": (f"{'GAIN' if d > 0 else 'loss'} holds under both tests" if mc_sig and bs_sig else "within noise")})
    df = pd.DataFrame(rows).set_index("pair") if rows else pd.DataFrame()
    display(df)
    return df


def leaderboard(tags: list[str], baseline: str, title: str, subtitle: str = "", save: str | None = None):
    accs = {t: (lambda r: sum(x["score"] >= 1 for x in r.values()) / len(r))(load_results(t)) for t in tags if load_results(t)}
    order = sorted(accs, key=accs.get)
    fig, ax = fv.plt.subplots(figsize=(11, 0.6 * len(order) + 2.2))
    fig.subplots_adjust(left=0.24, right=0.8)
    fv._title(fig, title, subtitle)
    for i, t in enumerate(order):
        ax.barh(i, accs[t] * 100, color=color(t), height=0.6, hatch="//" if t == baseline else None,
                edgecolor=fv.INK if t == baseline else "none")
        ax.text(accs[t] * 100 + 0.6, i, f"{accs[t]:.1%}", va="center")
        if t != baseline and baseline in accs:
            r = compare(t, baseline, B=2000)
            sig = r["mcnemar"]["p"] < ALPHA and (r["boot"]["lo"] > 0 or r["boot"]["hi"] < 0)
            d = 100 * (r["acc_a"] - r["acc_b"])
            ax.text(1.02, i, f"{d:+.1f}pp  {'significant' if sig else 'n.s.'}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=10, clip_on=False, fontweight="bold" if sig else "normal",
                    color=(fv.MAGENTA if d > 0 else fv.CORAL) if sig else fv.MUTED)
    ax.set_yticks(range(len(order)), [name(t) + ("  (baseline)" if t == baseline else "") for t in order])
    ax.set_xlim(0, 100); ax.set_xlabel("accuracy on the FloorPlanCAD test set (%)")
    ax.grid(axis="x", color=fv.GRID); ax.grid(axis="y", visible=False)
    fv.show(fig, save)


def class_heatmap(tags: list[str], min_n: int = 30, save: str | None = None):
    tags = [t for t in tags if load_results(t)]
    if not tags:
        print("class_heatmap: no results yet")
        return
    data = {}
    for t in tags:
        res = load_results(t)
        if res:
            df = pd.DataFrame(res.values())
            data[name(t)] = df.groupby(["kind", "cls"]).score.mean()
    counts = pd.DataFrame(load_results(tags[0]).values()).groupby(["kind", "cls"]).size()
    keep = counts[counts >= min_n].index
    mat = pd.DataFrame(data).loc[keep].sort_values(name(tags[0]))
    fig, ax = fv.plt.subplots(figsize=(11, 0.42 * len(mat) + 2.4))
    fv._title(fig, "Accuracy by question type and class", f"Groups with at least {min_n} test questions · darker = better")
    ax.imshow(mat.values * 100, cmap=fv.SUNSET, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(mat.shape[1]), mat.columns, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(mat.shape[0]), [f"{k} {c}  (n={counts[(k, c)]})" for k, c in mat.index], fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j] * 100
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8.5, color=fv.BG if v > 70 else fv.INK)
    ax.grid(False)
    fv.show(fig, save)


def cost_table(tags: list[str]) -> pd.DataFrame:
    rows = []
    for t in tags:
        res = list(load_results(t).values())
        if res:
            ok = sum(r["score"] >= 1 for r in res)
            cost = sum(r.get("cost_usd", 0) for r in res)
            rows.append({"system": name(t), "accuracy": f"{ok / len(res):.1%}", "errors": sum(bool(r.get("error")) for r in res),
                         "cost $": round(cost, 2), "$ per 1k questions": round(1000 * cost / len(res), 2),
                         "median latency s": round(float(np.median([r["latency_s"] for r in res])), 1)})
    df = pd.DataFrame(rows).set_index("system")
    display(df)
    return df
