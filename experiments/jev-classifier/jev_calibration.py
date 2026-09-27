"""Temperature scaling for the Jev classifiers.

One number T > 0 divides every option's log-probability before renormalizing. T > 1 softens
overconfident probabilities; the top option never changes, so accuracy is untouched. T is fit on
our own dev rows (tasks we trained on, unseen rows), never on a benchmark we report.
"""

import json
import math
from pathlib import Path

import numpy as np


def scale(probs, temperature):
    """Apply temperature to a probability vector (list/array) or a {key: prob} dict."""
    if isinstance(probs, dict):
        keys = list(probs)
        out = scale([probs[k] for k in keys], temperature)
        return dict(zip(keys, out))
    logp = np.log(np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)) / temperature
    logp -= logp.max()
    p = np.exp(logp)
    return (p / p.sum()).tolist()


def nll(rows, temperature):
    return float(np.mean([-math.log(max(scale(r["probs"], temperature)[r["gold"]], 1e-12)) for r in rows]))


def fit_temperature(rows, lo=0.25, hi=10.0, iters=60):
    """Minimize mean negative log-likelihood of the gold option over T (golden-section on log T)."""
    rows = [r for r in rows if not r.get("format_miss") and r.get("probs")]
    a, b = math.log(lo), math.log(hi)
    phi = (math.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(rows, math.exp(c)), nll(rows, math.exp(d))
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = nll(rows, math.exp(c))
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = nll(rows, math.exp(d))
    t = math.exp((a + b) / 2)
    return {"temperature": t, "nll_before": nll(rows, 1.0), "nll_after": nll(rows, t), "n": len(rows)}


def fit_from_eval_file(path, split="dev"):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return fit_temperature([r for r in rows if r.get("split") == split])


def ece(rows, temperature=1.0, bins=15):
    rows = [r for r in rows if not r.get("format_miss") and r.get("probs")]
    conf, corr = [], []
    for r in rows:
        p = scale(r["probs"], temperature)
        conf.append(max(p)); corr.append(int(int(np.argmax(p)) == r["gold"]))
    conf, corr = np.array(conf), np.array(corr)
    idx = np.clip(np.digitize(conf, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    return float(sum(abs(corr[idx == b].mean() - conf[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def rescale_results(src, dst, temperature):
    """Apply temperature to every answer in a Decision Index results.jsonl (no new requests).
    Choices keep their argmax; yes/no ("noul") probabilities are scaled as a two-option distribution."""
    with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "ok":
                for a in r["response"]["answers"].values():
                    if a["type"] == "choice":
                        a["probabilities"] = scale(a["probabilities"], temperature)
                    else:
                        a["noul"] = scale([a["noul"], 1 - a["noul"]], temperature)[0]
            out.write(json.dumps(r) + "\n")


def save(result, path):
    Path(path).write_text(json.dumps(result, indent=2))
