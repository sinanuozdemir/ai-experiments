"""FloorPlanCAD counting / presence benchmark built from the official SVG annotations, with a cached async eval.

Images come from Voxel51/FloorPlanCAD on Hugging Face. Its class names are scrambled, so gold counts come from the
original FloorPlanCAD SVGs instead (data/official_counts.json, written by 01_official_labels.ipynb)."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from openai import AsyncOpenAI
from PIL import Image
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / ".env")
DATA = HERE / "data"
SPLITS = DATA / "splits"
RESULTS = HERE / "results"
OFFICIAL_NAMES = ["wall", "curtain wall", "single door", "double door", "sliding door", "folding door", "revolving door",
                  "rolling door", "window", "bay window", "blind window", "opening symbol", "sofa", "bed", "chair", "table",
                  "TV cabinet", "wardrobe", "cabinet", "gas stove", "sink", "refrigerator", "airconditioner", "bath",
                  "bath tub", "washing machine", "squat toilet", "urinal", "toilet", "stairs", "elevator", "escalator",
                  "railing", "row chairs", "parking spot"]
OFFICIAL_GROUPS = {"door": ["single door", "double door", "sliding door", "folding door", "revolving door", "rolling door"],
                   "window": ["window", "bay window", "blind window"], "toilet": ["toilet", "squat toilet"], "sink": ["sink"],
                   "staircase": ["stairs"], "elevator": ["elevator"], "urinal": ["urinal"], "stove": ["gas stove"],
                   "bathtub": ["bath", "bath tub"]}
CLASSES = {"door": "doors", "window": "windows", "toilet": "toilets", "sink": "sinks", "staircase": "staircases",
           "elevator": "elevators", "urinal": "urinals", "stove": "stoves", "bathtub": "bathtubs"}

MODELS = {
    "glm": {"label": "GLM 5.3 Flash", "model": "accounts/fireworks/models/glm-5p3-flash", "host": "fireworks", "price": (0.15, 0.50)},
    "kimi": {"label": "Kimi K3", "model": "accounts/fireworks/models/kimi-k3", "host": "fireworks", "price": (3.00, 15.00)},
    "muse": {"label": "Muse Spark", "model": "accounts/fireworks/models/muse-spark-open", "host": "fireworks", "price": (1.25, 4.25)},
    "gemini4": {"label": "Gemini 4 Argon", "model": "google/gemini-4-argon", "host": "openrouter", "price": (2.00, 10.00)},
    "opus55": {"label": "Claude Opus 5.5", "model": "anthropic/claude-opus-5.5", "host": "openrouter", "price": (4.00, 20.00)},
    "gpt61": {"label": "GPT-6.1 Terra Pro", "model": "openai/gpt-6.1-terra-pro", "host": "openrouter", "price": (1.00, 5.00)},
}

SYSTEM = ("You read architectural CAD floor plans (coloured line drawings on a black background). "
          "Symbols follow standard drafting conventions: doors are arcs showing the swing, windows are thin parallel lines "
          "in walls, staircases are rows of parallel step lines, elevators are boxes with an X. "
          "Put the FINAL ANSWER alone on the last line: a whole number for counting questions, yes or no otherwise.")


def load_samples() -> list[dict]:
    """Voxel51 sample index (one entry per tile image); downloaded once."""
    p = DATA / "samples.json"
    if not p.exists():
        hf_hub_download("Voxel51/FloorPlanCAD", "samples.json", repo_type="dataset", local_dir=str(DATA))
    return json.loads(p.read_text())["samples"]


def build(test_frac: float = 0.2, seed: int = 0, max_count: int = 12, test_cap: int = 1000) -> dict:
    """Building-level split; one count + one presence question per tile. Free and idempotent."""
    if (SPLITS / "test.jsonl").exists():
        return {s: sum(1 for _ in (SPLITS / f"{s}.jsonl").open()) for s in ("train", "test")}
    samples = load_samples()
    official = json.loads((DATA / "official_counts.json").read_text())
    group = {OFFICIAL_NAMES.index(n) + 1: c for c, ns in OFFICIAL_GROUPS.items() for n in ns}
    buildings = sorted({x["filepath"].split("/")[-1].split("-")[0] for x in samples})
    rng = random.Random(seed)
    rng.shuffle(buildings)
    test_b = set(buildings[: int(len(buildings) * test_frac)])
    out = {"train": [], "test": []}
    for x in samples:
        b = x["filepath"].split("/")[-1].split("-")[0]
        split = "test" if b in test_b else "train"
        tile = x["filepath"].split("/")[-1].removesuffix(".png")
        counts = Counter()
        for s, v in official[tile].items():
            if int(s) in group:
                counts[group[int(s)]] += v["n"]
        present = [c for c in counts if 1 <= counts[c] <= max_count]
        if present:
            c = rng.choice(present)
            out[split].append({"id": f"{tile}-count-{c}", "tile": tile, "building": b, "filepath": x["filepath"], "kind": "count",
                               "cls": c, "question": f"How many {CLASSES[c]} are shown in this floor plan?", "answer": counts[c]})
        if not any(counts[k] for k in CLASSES) or all(counts[k] for k in CLASSES):
            continue  # need both a present and an absent class, so yes/no stays 50/50
        want_yes = rng.random() < 0.5
        cands = [k for k in CLASSES if (counts[k] > 0) == want_yes]
        c = rng.choice(cands)
        out[split].append({"id": f"{tile}-has-{c}", "tile": tile, "building": b, "filepath": x["filepath"], "kind": "presence",
                           "cls": c, "question": f"Does this floor plan contain any {CLASSES[c]}?", "answer": "yes" if counts[c] else "no"})
    rng.shuffle(out["test"])
    out["test"] = out["test"][:test_cap]
    SPLITS.mkdir(parents=True, exist_ok=True)
    for s, rows in out.items():
        (SPLITS / f"{s}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {s: len(r) for s, r in out.items()}


def load_split(split: str) -> list[dict]:
    return [json.loads(l) for l in (SPLITS / f"{split}.jsonl").read_text().splitlines() if l.strip()]


def image_path(r: dict) -> Path:
    p = DATA / r["filepath"]
    if not p.exists():
        hf_hub_download("Voxel51/FloorPlanCAD", r["filepath"], repo_type="dataset", local_dir=str(DATA))
    return p


def prefetch(rows: list[dict], workers: int = 16) -> None:
    uniq = list({r["filepath"]: r for r in rows}.values())
    with ThreadPoolExecutor(workers) as ex:
        list(tqdm(ex.map(image_path, uniq), total=len(uniq), desc="images"))


def image_url(r: dict, box: tuple | None = None, size: int = 1000) -> str:
    img = Image.open(image_path(r)).convert("RGB")
    if box:
        W, H = img.size
        img = img.crop((int(box[0] * W), int(box[1] * H), int(box[2] * W), int(box[3] * H))).resize((size, size))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def messages(r: dict) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url(r)}},
                                         {"type": "text", "text": r["question"]}]}]


WORD = {"zero": 0, "none": 0, "no": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12}


def parse(kind: str, text: str):
    t = re.sub(r"<think>[\s\S]*?</think>", "", text or "")
    lines = [l.strip().strip("*").lower() for l in t.strip().splitlines() if l.strip()]
    for line in reversed(lines[-3:]):
        if kind == "count":
            nums = re.findall(r"\b\d+\b", line)
            if nums:
                return int(nums[-1])
            words = [WORD[w] for w in re.findall(r"[a-z]+", line) if w in WORD]
            if words:
                return words[-1]
        else:
            hits = re.findall(r"\b(yes|no)\b", line)
            if hits:
                return hits[-1]
    return None


def grade(r: dict, pred) -> dict:
    if r["kind"] == "count":
        ok = pred == r["answer"]
        near = pred is not None and abs(pred - r["answer"]) <= 1
        return {"score": float(ok), "within1": float(near)}
    return {"score": float(pred == r["answer"]), "within1": float(pred == r["answer"])}


def results_path(tag: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    return RESULTS / f"results_{tag}.jsonl"


def load_results(tag: str) -> dict[str, dict]:
    p = results_path(tag)
    return {r["id"]: r for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())} if p.exists() else {}


def _client(host: str):
    if host == "openrouter":
        return (AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=300),
                {"extra_body": {"reasoning": {"effort": "low"}}})
    return (AsyncOpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"], timeout=300),
            {"reasoning_effort": "low"})


async def eval_model(model: str, tag: str, split: str = "test", *, build_messages=None, n: int = 1, temperature: float = 0.0,
                     concurrency: int = 8, max_tokens: int = 4096, limit: int | None = None, endpoint: str | None = None):
    """Cached; re-running fills missing or errored items. ``endpoint`` overrides the model id (e.g. a deployment)."""
    m = MODELS.get(model, {"label": model, "model": model, "host": "fireworks", "price": (0, 0)})
    rows = load_split(split)
    done = {k for k, r in load_results(tag).items() if not r.get("error")}
    todo = [r for r in rows if r["id"] not in done][:limit]
    print(f"{tag}: {m['label']} on {split} · n={n} · to_run={len(todo)}")
    prefetch(todo)
    client, kw = _client(m["host"])
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    build_fn = build_messages or messages

    async def call(msgs):
        err = None
        async with sem:
            for attempt in range(3):
                try:
                    resp = await client.chat.completions.create(model=endpoint or m["model"], messages=msgs, max_tokens=max_tokens,
                                                                temperature=temperature, **kw)
                    if not resp.choices:
                        raise RuntimeError("empty response")
                    return (resp.choices[0].message.content or "").strip(), None, resp.usage
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(2 * (attempt + 1))
        return "", err, None

    async def one(r):
        t0 = time.perf_counter()
        msgs = build_fn(r)
        outs = await asyncio.gather(*(call(msgs) for _ in range(n)))
        preds = [parse(r["kind"], o[0]) for o in outs]
        valid = [p for p in preds if p is not None]
        pred = Counter(valid).most_common(1)[0][0] if valid else None
        pin, pout = m["price"]
        ptok = sum((o[2].prompt_tokens or 0) for o in outs if o[2])
        ctok = sum((o[2].completion_tokens or 0) for o in outs if o[2])
        rec = {"id": r["id"], "model": tag, "kind": r["kind"], "cls": r["cls"], "building": r["building"], "answer": r["answer"],
               "pred": pred, **grade(r, pred), "text": outs[0][0][-1200:],
               "error": next((o[1] for o in outs if o[1]), None) if not valid and all(o[1] for o in outs) else None,
               "prompt_tokens": ptok, "completion_tokens": ctok, "cost_usd": (ptok * pin + ctok * pout) / 1e6,
               "latency_s": round(time.perf_counter() - t0, 2)}
        async with lock:
            with results_path(tag).open("a") as f:
                f.write(json.dumps(rec) + "\n")

    if todo:
        await tqdm.gather(*(one(r) for r in todo), desc=tag)
    res = load_results(tag)
    acc = sum(r["score"] for r in res.values()) / max(len(res), 1)
    cnt = [r for r in res.values() if r["kind"] == "count"]
    print(f"{tag}: accuracy {acc:.1%} · counting exact {sum(r['score'] for r in cnt) / max(len(cnt), 1):.1%}, "
          f"within ±1 {sum(r['within1'] for r in cnt) / max(len(cnt), 1):.1%} (n={len(res)})")
