"""Engineering drawings (ecado/rfq-engineering-drawings) → JSON: data, prompts, JSON F1, cached async eval."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image
from tqdm.auto import tqdm

HERE = Path(__file__).resolve().parent
load_dotenv(HERE.parents[1] / ".env")
DATA = HERE / "data"
RESULTS = HERE / "results"

MODELS = {
    "glm": {"label": "GLM 5.3 Flash", "model": "accounts/fireworks/models/glm-5p3-flash", "host": "fireworks", "price": (0.15, 0.50)},
    "kimi": {"label": "Kimi K3", "model": "accounts/fireworks/models/kimi-k3", "host": "fireworks", "price": (3.00, 15.00)},
    "muse": {"label": "Muse Spark", "model": "accounts/fireworks/models/muse-spark-open", "host": "fireworks", "price": (1.25, 4.25)},
    "gemini4": {"label": "Gemini 4 Argon", "model": "google/gemini-4-argon", "host": "openrouter", "price": (2.00, 10.00)},
    "opus55": {"label": "Claude Opus 5.5", "model": "anthropic/claude-opus-5.5", "host": "openrouter", "price": (4.00, 20.00)},
    "gpt61": {"label": "GPT-6.1 Terra Pro", "model": "openai/gpt-6.1-terra-pro", "host": "openrouter", "price": (1.00, 5.00)},
}

# The dataset's own instruction, plus the JSON-only line from the CORD cookbook.
INSTRUCTION = "Extract all manufacturing information from this 2D engineering drawing. Return structured JSON with: title_block, dimensions, gdt_callouts, surface_finish, treatment, weld_symbols, and notes. Respond with only the JSON object."


def prepare() -> None:
    """Download the three splits from Hugging Face into data/<split>.jsonl + data/images/. Free and idempotent."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    (DATA / "images").mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        if (DATA / f"{split}.jsonl").exists():
            continue
        df = pd.read_parquet(hf_hub_download("ecado/rfq-engineering-drawings", f"data/{split}-00000-of-00001.parquet",
                                             repo_type="dataset"))
        rows = []
        for i, r in enumerate(df.itertuples()):
            img = f"data/images/{split}-{i:04d}.png"
            (HERE / img).write_bytes(r.image["bytes"])
            rows.append({"id": f"{split}-{i:04d}", "image": img, "gold": json.loads(r.ground_truth)["gt_parse"], "question": r.question})
        (DATA / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def load_split(split: str) -> list[dict]:
    prepare()
    return [json.loads(l) for l in (DATA / f"{split}.jsonl").read_text().splitlines() if l.strip()]


def gold_text(r: dict) -> str:
    return json.dumps(r["gold"], ensure_ascii=False, separators=(",", ":"))


def image_url(r: dict, max_dim: int = 1200, quality: int = 90) -> str:
    img = Image.open(HERE / r["image"]).convert("RGB")
    w, h = img.size
    s = max_dim / max(w, h)
    if s < 1:
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def oneshot_system(example: dict | None = None) -> str:
    """The cookbook's frontier prompt: exact schema from one TRAINING drawing (never a test drawing)."""
    ex = example or load_split("train")[0]
    return ("You are an expert at reading 2D engineering drawings. Extract the drawing as a SINGLE JSON object using EXACTLY "
            "the same schema and key names as the example below. Respond with ONLY the JSON object, no prose.\n\n"
            "Example of the exact output format (the values will differ for each drawing):\n" + gold_text(ex))


def messages(r: dict, system: str | None = None) -> list[dict]:
    user = {"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url(r)}},
                                        {"type": "text", "text": INSTRUCTION}]}
    return ([{"role": "system", "content": system}] if system else []) + [user]


# ---- scoring: identical to the cookbook's _extract_json / _norm / _leaf_pairs / _receipt_f1 ----

def extract_json(text: str):
    t = re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.DOTALL)
    t = t.replace("```json", "").replace("```", "")
    a, b = t.find("{"), t.rfind("}")
    if a == -1 or b == -1 or b < a:
        return None
    try:
        return json.loads(t[a:b + 1])
    except Exception:  # noqa: BLE001
        return None


def _norm(v) -> str:
    return re.sub(r"[,\s$]", "", str(v).lower().strip())


def leaf_pairs(obj, key=None, out=None) -> set:
    out = set() if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            leaf_pairs(v, k, out)
    elif isinstance(obj, list):
        for v in obj:
            leaf_pairs(v, key, out)
    elif obj is not None and str(obj).strip() != "":
        out.add((str(key), _norm(obj)))
    return out


def receipt_f1(pred_text: str, gold: dict) -> dict:
    pred = extract_json(pred_text)
    g = leaf_pairs(gold)
    if pred is None:
        return {"f1": 0.0, "valid_json": False, "hits": [], "n_gold": len(g)}
    p = leaf_pairs(pred)
    inter = g & p
    prec = len(inter) / len(p) if p else 0.0
    rec = len(inter) / len(g) if g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"f1": f1, "valid_json": True, "precision": prec, "recall": rec,
            "hits": sorted(f"{k}={v}" for k, v in inter), "n_gold": len(g)}


def medoid(texts: list[str]) -> str:
    """Self-consistency for JSON: keep the sample whose fields agree most with the other samples."""
    sets = [leaf_pairs(extract_json(t) or {}) for t in texts]
    def jac(a, b):
        return len(a & b) / len(a | b) if a | b else 0.0
    scores = [sum(jac(s, o) for j, o in enumerate(sets) if j != i) for i, s in enumerate(sets)]
    return texts[max(range(len(texts)), key=lambda i: (scores[i], len(sets[i])))] if texts else ""


# ---- cached eval ----

def results_path(tag: str) -> Path:
    RESULTS.mkdir(exist_ok=True)
    return RESULTS / f"results_{tag}.jsonl"


def load_results(tag: str) -> dict[str, dict]:
    p = results_path(tag)
    if not p.exists():
        return {}
    return {r["id"]: r for r in (json.loads(l) for l in p.read_text().splitlines() if l.strip())}


def _client(host: str):
    if host == "openrouter":
        return (AsyncOpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"], timeout=300),
                {"extra_body": {"reasoning": {"effort": "low"}}})
    return (AsyncOpenAI(base_url="https://api.fireworks.ai/inference/v1", api_key=os.environ["FIREWORKS_API_KEY"], timeout=300),
            {"reasoning_effort": "low"})


async def eval_model(model: str, tag: str, split: str = "test", *, system: str | None = None, n: int = 1,
                     temperature: float = 0.0, concurrency: int = 8, max_tokens: int = 4096, limit: int | None = None) -> Path:
    """Re-running only fills in missing or errored drawings. n>1 keeps the medoid sample (self-consistency)."""
    m = MODELS[model]
    rows = load_split(split)
    done = {k for k, r in load_results(tag).items() if not r.get("error")}
    todo = [r for r in rows if r["id"] not in done][:limit]
    print(f"{tag}: {m['label']} on {split} · n={n} · to_run={len(todo)}")
    client, kw = _client(m["host"])
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()

    async def call(msgs):
        err = None
        async with sem:
            for attempt in range(3):
                try:
                    resp = await client.chat.completions.create(model=m["model"], messages=msgs, max_tokens=max_tokens,
                                                                temperature=temperature, **kw)
                    if not resp.choices:
                        raise RuntimeError("empty response")
                    return (resp.choices[0].message.content or "").strip(), None, resp.usage
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(2 * (attempt + 1))
        return "", err, None

    async def one(r):
        msgs = messages(r, system)
        t0 = time.perf_counter()
        outs = await asyncio.gather(*(call(msgs) for _ in range(n)))
        texts = [o[0] for o in outs if o[0]]
        final = medoid(texts) if n > 1 else (texts[0] if texts else "")
        sc = receipt_f1(final, r["gold"])
        pin, pout = m["price"]
        ptok = sum((o[2].prompt_tokens or 0) for o in outs if o[2])
        ctok = sum((o[2].completion_tokens or 0) for o in outs if o[2])
        rec = {"id": r["id"], "model": tag, **sc, "pred": final[:6000],
               "error": next((o[1] for o in outs if o[1]), None) if not texts else None,
               "prompt_tokens": ptok, "completion_tokens": ctok, "cost_usd": (ptok * pin + ctok * pout) / 1e6,
               "latency_s": round(time.perf_counter() - t0, 2)}
        async with lock:
            with results_path(tag).open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if todo:
        await tqdm.gather(*(one(r) for r in todo), desc=tag)
    res = load_results(tag)
    f1 = sum(r["f1"] for r in res.values()) / max(len(res), 1)
    print(f"{tag}: mean JSON F1 {f1:.1%} · valid JSON {sum(r['valid_json'] for r in res.values())}/{len(res)}")
    return results_path(tag)
