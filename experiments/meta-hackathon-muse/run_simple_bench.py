#!/usr/bin/env python3
"""UiPad-style short-answer subset: Muse Spark Open vs DeepSeek V4.1 Flash.

Answer-type analogs (text-only, no screenshots / no coordinates):
  - number  -> GSM8K
  - yes/no  -> BoolQ
  - string  -> Natural Questions Open (nq_open)
  - choice  -> ARC-Easy (A–D)

Deterministic grading after a light answer extract. Resumable JSONL.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

from datasets import load_dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

MODELS = [
    {
        "label": "Muse Spark Open",
        "short": "muse",
        "model": "accounts/fireworks/models/muse-spark-open",
        # Muse always reasons; valid ladder is low / medium / high / xhigh (no "none").
        "efforts": ["low", "medium", "high", "xhigh"],
    },
    {
        "label": "DeepSeek V4.1 Flash",
        "short": "ds",
        "model": "accounts/fireworks/models/deepseek-v4p1-flash",
        "efforts": ["low", "high"],
    },
]

N_PER_BENCH = 40
SEED = 7
CONCURRENCY = 6
MAX_TOKENS = 2048
REQUEST_TIMEOUT = 300
RESULTS = Path(__file__).parent / "results.jsonl"

SYSTEM = (
    "You answer short factual / reasoning questions. "
    "Think briefly if needed, then put the final answer alone on the last line.\n"
    "- math / numeric -> a number (e.g. 18 or 3.5)\n"
    "- yes/no -> Yes or No\n"
    "- open factoid -> a short exact answer string\n"
    "- multiple choice -> a single letter A, B, C, or D"
)


def _norm_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _last_number(s: str) -> str | None:
    nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return nums[-1] if nums else None


def _extract_yes_no(s: str) -> str | None:
    t = s.strip().lower()
    # prefer last yes/no mention
    hits = list(re.finditer(r"\b(yes|no)\b", t))
    if hits:
        return hits[-1].group(1)
    return None


def _extract_choice(s: str) -> str | None:
    hits = list(re.finditer(r"\b([ABCD])\b", s.upper()))
    if hits:
        return hits[-1].group(1)
    m = re.search(r"(?:answer|choice)\s*[:is]*\s*([ABCD])\b", s.upper())
    return m.group(1) if m else None


def grade(kind: str, pred_text: str, gold) -> float:
    if not pred_text:
        return 0.0
    if kind == "number":
        p, g = _last_number(pred_text), _last_number(str(gold))
        if p is None or g is None:
            return 0.0
        try:
            return float(abs(float(p) - float(g)) <= max(1e-6, 0.01 * abs(float(g))))
        except ValueError:
            return float(p == g)
    if kind == "yes_no":
        p = _extract_yes_no(pred_text)
        if p is None:
            return 0.0
        g = "yes" if bool(gold) else "no"
        return float(p == g)
    if kind == "string":
        # gold is list of acceptable answers
        pred = _norm_text(pred_text.split("\n")[-1])
        if not pred:
            return 0.0
        for a in gold:
            na = _norm_text(str(a))
            if not na:
                continue
            if pred == na or na in pred or pred in na:
                return 1.0
        return 0.0
    if kind == "choice":
        p = _extract_choice(pred_text)
        return float(p == str(gold).upper()) if p else 0.0
    raise ValueError(kind)


def build_rows(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    idxs = list(range(len(gsm)))
    rng.shuffle(idxs)
    for i in idxs[:n]:
        ex = gsm[i]
        gold = ex["answer"].split("####")[-1].strip()
        rows.append(
            {
                "bench": "gsm8k",
                "kind": "number",
                "id": f"gsm8k-{i}",
                "prompt": ex["question"],
                "gold": gold,
            }
        )

    bq = load_dataset("google/boolq", split="validation")
    idxs = list(range(len(bq)))
    rng.shuffle(idxs)
    for i in idxs[:n]:
        ex = bq[i]
        prompt = (
            f"Passage:\n{ex['passage']}\n\n"
            f"Question: {ex['question']}?\n"
            f"Answer Yes or No."
        )
        rows.append(
            {
                "bench": "boolq",
                "kind": "yes_no",
                "id": f"boolq-{i}",
                "prompt": prompt,
                "gold": bool(ex["answer"]),
            }
        )

    nq = load_dataset("google-research-datasets/nq_open", split="validation")
    idxs = list(range(len(nq)))
    rng.shuffle(idxs)
    for i in idxs[:n]:
        ex = nq[i]
        rows.append(
            {
                "bench": "nq_open",
                "kind": "string",
                "id": f"nq-{i}",
                "prompt": ex["question"] + "\nGive a short factual answer.",
                "gold": list(ex["answer"]),
            }
        )

    arc = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
    idxs = list(range(len(arc)))
    rng.shuffle(idxs)
    for i in idxs[:n]:
        ex = arc[i]
        choices = "\n".join(
            f"{lab}. {txt}" for lab, txt in zip(ex["choices"]["label"], ex["choices"]["text"])
        )
        rows.append(
            {
                "bench": "arc_easy",
                "kind": "choice",
                "id": f"arc-{ex['id']}",
                "prompt": f"{ex['question']}\n\n{choices}\n\nReply with the letter only.",
                "gold": ex["answerKey"],
            }
        )

    return rows


def load_done(path: Path) -> set[tuple]:
    done = set()
    if not path.exists():
        return done
    with path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["model"], r["effort"], r["id"], r.get("sample", 0)))
    return done


async def call_one(client: AsyncOpenAI, model: str, effort: str, prompt: str) -> dict:
    t0 = time.perf_counter()
    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=MAX_TOKENS,
            temperature=0.0,
            reasoning_effort=effort,
        )
        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        usage = resp.usage
        return {
            "ok": True,
            "text": text,
            "latency_s": time.perf_counter() - t0,
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "reasoning_tokens": (
                getattr(getattr(usage, "completion_tokens_details", None), "reasoning_tokens", None)
            ),
            "error": None,
        }
    except Exception as e:  # noqa: BLE001 — keep sweep going
        return {
            "ok": False,
            "text": "",
            "latency_s": time.perf_counter() - t0,
            "prompt_tokens": None,
            "completion_tokens": None,
            "reasoning_tokens": None,
            "error": f"{type(e).__name__}: {e}",
        }


async def run(efforts_filter: list[str] | None, n: int, models_filter: list[str] | None) -> None:
    key = os.environ.get("FIREWORKS_API_KEY")
    if not key:
        raise SystemExit("Set FIREWORKS_API_KEY in repo-root .env")

    client = AsyncOpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=key,
        timeout=REQUEST_TIMEOUT,
    )
    rows = build_rows(n, SEED)
    done = load_done(RESULTS)
    sem = asyncio.Semaphore(CONCURRENCY)

    models = MODELS
    if models_filter:
        models = [m for m in models if m["short"] in models_filter or m["label"] in models_filter]

    jobs = []
    for m in models:
        efforts = m["efforts"]
        if efforts_filter:
            efforts = [e for e in efforts if e in efforts_filter]
        for effort in efforts:
            for row in rows:
                key_t = (m["label"], effort, row["id"], 0)
                if key_t in done:
                    continue
                jobs.append((m, effort, row))

    print(f"rows={len(rows)} pending={len(jobs)} results={RESULTS}")
    if not jobs:
        print("nothing to run")
        return

    lock = asyncio.Lock()

    async def one(m, effort, row):
        async with sem:
            out = await call_one(client, m["model"], effort, row["prompt"])
        score = grade(row["kind"], out["text"], row["gold"]) if out["ok"] else 0.0
        rec = {
            "model": m["label"],
            "model_id": m["model"],
            "effort": effort,
            "bench": row["bench"],
            "kind": row["kind"],
            "id": row["id"],
            "sample": 0,
            "score": score,
            "gold": row["gold"],
            "pred": out["text"],
            "ok": out["ok"],
            "error": out["error"],
            "latency_s": out["latency_s"],
            "prompt_tokens": out["prompt_tokens"],
            "completion_tokens": out["completion_tokens"],
            "reasoning_tokens": out["reasoning_tokens"],
        }
        async with lock:
            with RESULTS.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    with tqdm(total=len(jobs), desc="sweep") as bar:
        tasks = [asyncio.create_task(one(m, e, r)) for m, e, r in jobs]
        for fut in asyncio.as_completed(tasks):
            await fut
            bar.update(1)


def summarize() -> None:
    if not RESULTS.exists():
        print("no results yet")
        return
    rows = [json.loads(l) for l in RESULTS.open() if l.strip()]
    from collections import defaultdict

    agg = defaultdict(list)
    for r in rows:
        if not r.get("ok", True) and not r.get("pred"):
            continue
        agg[(r["bench"], r["model"], r["effort"])].append(r["score"])

    print("\npass@1 (%) by bench × model@effort")
    benches = sorted({b for b, _, _ in agg})
    cfgs = sorted({(m, e) for _, m, e in agg})
    header = f"{'bench':12}" + "".join(f" | {m[:12]}@{e:4}" for m, e in cfgs)
    print(header)
    for b in benches:
        cells = []
        for m, e in cfgs:
            xs = agg.get((b, m, e), [])
            cells.append(f"{100 * sum(xs) / len(xs):5.1f}" if xs else "  n/a")
        print(f"{b:12}" + "".join(f" | {c:>17}" for c in cells))

    print("\noverall")
    for m, e in cfgs:
        xs = [s for (b, mm, ee), vals in agg.items() if mm == m and ee == e for s in vals]
        print(f"  {m} @ {e}: {100 * sum(xs) / len(xs):.1f}%  (n={len(xs)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N_PER_BENCH, help="samples per benchmark")
    ap.add_argument("--efforts", nargs="*", default=None, help="e.g. low high")
    ap.add_argument("--models", nargs="*", default=None, help="muse ds")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    if args.summarize_only:
        summarize()
        return
    asyncio.run(run(args.efforts, args.n, args.models))
    summarize()


if __name__ == "__main__":
    main()
