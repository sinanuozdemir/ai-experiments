"""Decision Index engines for the zero-shot Qwen3 embedder and reranker on Fireworks serverless.

Same text construction as scion_embedder_baselines.ipynb: the task becomes an instructed query, each option becomes
"key: description". The embedder scores options by cosine similarity to the query; the reranker scores each
(query, option) pair. Scores become probabilities with softmax(score / T), where T was fit on our own dev rows.
Every option gets a score directly, so there is no top-k limit and nothing is estimated.
"""

from __future__ import annotations

import math
import os
import time

import numpy as np
import requests

from decision_index.engines.base import Engine, Unsupported, text

INSTRUCTION = "Given a decision task with a state and a question, retrieve the option that correctly answers it"
BASE = "https://api.fireworks.ai/inference/v1"
CAPACITY_MARKERS = ("context", "too long", "maximum", "exceeds", "token")
EMBED_BATCH = 128


def _option_text(key, desc):
    d = key if desc is None else text(desc)
    return d if d.strip().lower() == key.strip().lower() else f"{key.replace('_', ' ')}: {d}"


class _Serverless(Engine):
    def __init__(self, temperature=1.0, timeout=180, retries=6, **options):
        super().__init__(**options)
        self.temperature = float(temperature)
        self.timeout, self.retries = float(timeout), int(retries)
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {os.environ['FIREWORKS_API_KEY']}",
                                     "Content-Type": "application/json"})

    def _post(self, path, body):
        for attempt in range(self.retries):
            try:
                r = self.session.post(f"{BASE}/{path}", json=body, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout):
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
                continue
            if r.status_code in (400, 413, 422) and any(m in r.text.lower() for m in CAPACITY_MARKERS):
                raise Unsupported(f"refused as too long: {r.text[:200]}")
            if r.status_code in (429, 500, 502, 503, 504) and attempt < self.retries - 1:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json()

    @staticmethod
    def _items(state, q):
        keys = list(q["criteria"]) if q["type"] == "choice" else ["true", "false"]
        crit = q["criteria"] if q["type"] == "choice" else {"true": "Yes", "false": "No"}
        s = text(state) if state not in ("", None, {}, []) else ""
        return keys, crit, text(q.get("instructions", "")), s

    def _scores(self, state, q):
        raise NotImplementedError

    def __call__(self, state, questions):
        answers, usage = {}, 0
        for qk, q in questions.items():
            if q["type"] not in ("choice", "noul"):
                raise Unsupported("unsupported question type " + str(q["type"]))
            keys, s = self._scores(state, q)
            z = np.array(s) / self.temperature
            p = np.exp(z - z.max())
            p /= p.sum()
            probs = dict(zip(keys, p.tolist()))
            if q["type"] == "choice":
                answers[qk] = {"type": "choice", "choice": max(probs, key=probs.get), "probabilities": probs}
            else:
                answers[qk] = {"type": "noul", "noul": probs["true"]}
        return {"model": self.model_name, "answers": answers, "usage": {"input_tokens": usage}}, None


class EmbedderEngine(_Serverless):
    name = "fireworks-embedder"
    latency = "HTTPS round trips to Fireworks serverless embeddings, including network."

    def __init__(self, model="fireworks/qwen3-embedding-8b", **options):
        super().__init__(**options)
        self.model_name = model
        self.provenance = {"kind": "zero-shot embedder", "model": model, "temperature": self.temperature,
                           "policy": "cosine(query, option) / T, softmax over the question's options"}

    def _scores(self, state, q):
        keys, crit, instr, s = self._items(state, q)
        texts = [f"Instruct: {INSTRUCTION}\nQuery:{instr}\n{s}"] + [_option_text(k, crit[k]) for k in keys]
        vecs = []
        for i in range(0, len(texts), EMBED_BATCH):
            data = self._post("embeddings", {"model": self.model_name, "input": texts[i:i + EMBED_BATCH]})["data"]
            vecs.extend(d["embedding"] for d in sorted(data, key=lambda d: d["index"]))
        v = np.array(vecs, dtype=float)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return keys, (v[1:] @ v[0]).tolist()


class RerankerEngine(_Serverless):
    name = "fireworks-reranker"
    latency = "HTTPS round trips to Fireworks serverless rerank, including network."

    def __init__(self, model="fireworks/qwen3-reranker-8b", **options):
        super().__init__(**options)
        self.model_name = model
        self.provenance = {"kind": "zero-shot reranker", "model": model, "temperature": self.temperature,
                           "policy": "logit(relevance(query, option)) / T, softmax over the question's options"}

    def _scores(self, state, q):
        keys, crit, instr, s = self._items(state, q)
        docs = [_option_text(k, crit[k]) for k in keys]
        res = self._post("rerank", {"model": self.model_name, "query": f"{instr}\n{s}", "documents": docs,
                                    "top_n": len(docs)})["data"]
        out = [0.0] * len(docs)
        for d in res:
            p = min(max(d["relevance_score"], 1e-6), 1 - 1e-6)
            out[d["index"]] = math.log(p / (1 - p))
        return keys, out
