"""Stats + Forge charts for the engineering drawings. Field-level McNemar (each gold field right/wrong) and drawing-level bootstrap on F1."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fw_utils as fv  # noqa: E402

from eng_common import MODELS, leaf_pairs, load_results, load_split  # noqa: E402

ALPHA = 0.05
NAME = {k: v["label"] for k, v in MODELS.items()}
NAME.update({
    "glm_zero": "GLM 5.3 Flash, zero-shot",
    "glm_oneshot": "GLM 5.3 Flash, one-shot schema (baseline)",
    "glm_fewshot": "GLM, 3-shot schema",
    "glm_fewshot_vote": "GLM, 3-shot schema, medoid of 5",
    "glm_sft": "GLM + vision LoRA (SFT on 480 drawings)",
    "kimi_oneshot": "Kimi K3, one-shot", "muse_oneshot": "Muse Spark, one-shot",
    "gemini4_oneshot": "Gemini 4 Argon, one-shot", "opus55_oneshot": "Claude Opus 5.5, one-shot",
    "gpt61_oneshot": "GPT-6.1 Terra Pro, one-shot",
})
COLOR = {"glm_zero": fv.BLUSH, "glm_oneshot": fv.CORAL, "glm_fewshot": fv.ORANGE, "glm_fewshot_vote": "#D9636B",
         "glm_sft": fv.INK, "kimi_oneshot": fv.PEACH, "muse_oneshot": fv.MAGENTA,
         "gemini4_oneshot": "#F2B880", "opus55_oneshot": "#C75D7A", "gpt61_oneshot": "#4F2A3A"}


def name(t):
    return NAME.get(t, t)


def _gold_fields() -> dict[str, list[str]]:
    from eng_common import _norm  # noqa: F401
    out = {}
    for r in load_split("test"):
        out[r["id"]] = sorted(f"{k}={v}" for k, v in leaf_pairs(r["gold"]))
    return out


def compare(a: str, b: str, B: int = 5000, seed: int = 42) -> dict | None:
    A, Bm = load_results(a), load_results(b)
    ids = sorted(set(A) & set(Bm))
    if not ids:
        return None
    gold = _gold_fields()
    fa, fb = [], []
    for i in ids:
        ha, hb = set(A[i].get("hits", [])), set(Bm[i].get("hits", []))
        for g in gold[i]:
            fa.append(g in ha)
            fb.append(g in hb)
    only_a = sum(x and not y for x, y in zip(fa, fb))
    only_b = sum(y and not x for x, y in zip(fa, fb))
    n = only_a + only_b
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
    da = np.array([A[i]["f1"] for i in ids])
    db = np.array([Bm[i]["f1"] for i in ids])
    rng = random.Random(seed)
    s = sorted(float(np.mean([da[j] - db[j] for j in (rng.randrange(len(ids)) for _ in ids)])) for _ in range(B))
    obs = float(np.mean(da - db))
    bp = min(1.0, 2 * sum(x <= 0 for x in s) / B) if obs >= 0 else min(1.0, 2 * sum(x >= 0 for x in s) / B)
    return {"n_receipts": len(ids), "n_fields": len(fa), "f1_a": float(da.mean()), "f1_b": float(db.mean()),
            "field_acc_a": sum(fa) / len(fa), "field_acc_b": sum(fb) / len(fb),
            "mcnemar": {"only_a": only_a, "only_b": only_b, "p": p,
                        "both_right": sum(x and y for x, y in zip(fa, fb)), "both_wrong": sum(not x and not y for x, y in zip(fa, fb))},
            "boot": {"delta": obs, "lo": s[int(0.025 * B)], "hi": s[int(0.975 * B) - 1], "p": bp, "samples": s},
            "units": [(i, float(x), float(y)) for i, x, y in zip(ids, da, db)]}


def significance_table(pairs, B: int = 5000) -> pd.DataFrame:
    rows = []
    for a, b in pairs:
        r = compare(a, b, B)
        if not r:
            continue
        mc_sig, bs_sig = r["mcnemar"]["p"] < ALPHA, r["boot"]["lo"] > 0 or r["boot"]["hi"] < 0
        d = 100 * r["boot"]["delta"]
        rows.append({"pair": f"{name(a)} vs {name(b)}", "drawings": r["n_receipts"], "fields": r["n_fields"],
                     "F1 a": f"{r['f1_a']:.1%}", "F1 b": f"{r['f1_b']:.1%}", "ΔF1 pp": round(d, 1),
                     "fields only a": r["mcnemar"]["only_a"], "fields only b": r["mcnemar"]["only_b"],
                     "McNemar p (fields)": f"{r['mcnemar']['p']:.2e}" if r["mcnemar"]["p"] < 1e-3 else round(r["mcnemar"]["p"], 4),
                     "bootstrap 95% CI (drawings)": f"[{100*r['boot']['lo']:+.1f}, {100*r['boot']['hi']:+.1f}]",
                     "significant": mc_sig and bs_sig,
                     "notes": (f"{'GAIN' if d > 0 else 'loss'} holds under both tests" if mc_sig and bs_sig else "within noise")})
    df = pd.DataFrame(rows).set_index("pair") if rows else pd.DataFrame()
    display(df)
    return df


def leaderboard(tags, baseline, title, subtitle="", save: str | None = None):
    f1 = {t: float(np.mean([r["f1"] for r in load_results(t).values()])) for t in tags if load_results(t)}
    order = sorted(f1, key=f1.get)
    fig, ax = fv.plt.subplots(figsize=(11, 0.6 * len(order) + 2.2))
    fig.subplots_adjust(left=0.3, right=0.8)
    fv._title(fig, title, subtitle)
    for i, t in enumerate(order):
        ax.barh(i, f1[t] * 100, color=COLOR.get(t, fv.MUTED), height=0.6, hatch="//" if t == baseline else None,
                edgecolor=fv.INK if t == baseline else "none")
        ax.text(f1[t] * 100 + 0.8, i, f"{f1[t]:.1%}", va="center")
        if t != baseline and baseline in f1:
            r = compare(t, baseline, B=2000)
            sig = r["mcnemar"]["p"] < ALPHA and (r["boot"]["lo"] > 0 or r["boot"]["hi"] < 0)
            d = 100 * r["boot"]["delta"]
            ax.text(1.02, i, f"{d:+.1f}pp  {'significant' if sig else 'n.s.'}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=10, clip_on=False, fontweight="bold" if sig else "normal",
                    color=(fv.MAGENTA if d > 0 else fv.CORAL) if sig else fv.MUTED)
    ax.set_yticks(range(len(order)), [name(t) for t in order])
    ax.set_xlim(0, 106); ax.set_xlabel("mean JSON F1 on engineering-drawings test (%)")
    ax.grid(axis="x", color=fv.GRID); ax.grid(axis="y", visible=False)
    fv.show(fig, save)


def before_after(before: str, after: str, title: str, subtitle: str = "", save: str | None = None):
    """Per-drawing F1, before vs after, sorted: every drawing is one arrow."""
    A, Bm = load_results(after), load_results(before)
    ids = sorted(set(A) & set(Bm), key=lambda i: Bm[i]["f1"])
    fig, ax = fv.plt.subplots(figsize=(12, 4.8))
    fv._title(fig, title, subtitle)
    x = np.arange(len(ids))
    b = np.array([Bm[i]["f1"] for i in ids]) * 100
    a = np.array([A[i]["f1"] for i in ids]) * 100
    for xi, bi, ai in zip(x, b, a):
        ax.plot([xi, xi], [bi, ai], color=fv.MAGENTA if ai > bi else (fv.CORAL if ai < bi else fv.GRID), lw=1.4)
    ax.scatter(x, b, s=16, color=fv.BG, edgecolor=COLOR.get(before, fv.CORAL), lw=1.4, label=f"before: {name(before)}", zorder=3)
    ax.scatter(x, a, s=18, color=COLOR.get(after, fv.INK), label=f"after: {name(after)}", zorder=4)
    ax.set_xlabel("test drawings (sorted by before-F1)"); ax.set_ylabel("JSON F1 (%)"); ax.set_ylim(-3, 103)
    ax.legend(loc="lower right")
    fv.show(fig, save)


def cost_table(tags) -> pd.DataFrame:
    rows = []
    for t in tags:
        res = list(load_results(t).values())
        if res:
            cost = sum(r.get("cost_usd", 0) for r in res)
            rows.append({"system": name(t), "mean F1": f"{np.mean([r['f1'] for r in res]):.1%}",
                         "valid JSON": f"{sum(r['valid_json'] for r in res)}/{len(res)}",
                         "errors": sum(bool(r.get("error")) for r in res), "cost $": round(cost, 3),
                         "$ per 1k drawings": round(1000 * cost / len(res), 2),
                         "median latency s": round(float(np.median([r.get("latency_s", 0) for r in res])), 1)})
    df = pd.DataFrame(rows).set_index("system")
    display(df)
    return df
