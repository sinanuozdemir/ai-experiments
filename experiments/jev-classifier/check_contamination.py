"""Check our SFT training data for overlap with the Decision Index 0.2 suite.

Three checks, strongest first:
  1. exact: every training `state` body (prefix like "Text: " stripped, whitespace/case normalized)
     against every text field in the suite rows.
  2. 13-gram: any 13-word window shared between a training body and a suite row (the standard
     contamination window from the GPT-3 / LLaMA reports). Catches partial copies.
  3. source-level: for the benchmarks built from datasets we also trained on (BANKING77, CLINC150),
     exact utterance match against the raw test files the suite is built from.

Usage: python check_contamination.py --train jev_runs/5f25c2/train_plain.jsonl \
           [--suite di_suite-0.2] [--raw di_work/artifacts/benchmark-suite/raw]
"""

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

N = 13
PREFIX = re.compile(r"^(Text|User message|News item|Review|Tweet|Post|Question context|Science question|"
                    r"Passage|Premise|Hypothesis|Question|Sentence|Sentence 1|Sentence 2|Question 1|"
                    r"Question 2|Claim|Word|App|Policy|Rule|Case|Applicant|Routing rules|Ticket|"
                    r"Marketplace rule|Listing|Customer message):\s*", re.I)


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split())


def bodies(state):
    # A state holds one or two labelled fields ("Premise: ...\nHypothesis: ..."); check each on its own.
    out = []
    for line in str(state).split("\n"):
        b = norm(PREFIX.sub("", line.strip()))
        if len(b.split()) >= 4:
            out.append(b)
    return out


def grams(text):
    w = text.split()
    return {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)}


def strings(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings(v)


def load_train(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        user = json.loads(line)["messages"][1]["content"]
        query = json.loads(user.split("\n\nLet me repeat that:\n\n")[0])
        rows.append(query["state"])
    return rows


def suite_question_texts(suite_dir):
    # Question text of every suite row (state + instructions, per line); option descriptions excluded.
    for path in sorted(Path(suite_dir).glob("*rows.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                raw = list(strings(r["state"])) + [q.get("instructions", "") for q in r["questions"].values()]
                texts = [norm(PREFIX.sub("", l.strip())) for t in raw for l in str(t).split("\n")]
                yield r["_evaluation"].get("dataset"), [t for t in texts if len(t.split()) >= 4]


def overlapping_rows(states, suite_dir):
    """Indices of training states sharing an exact text field or any 13-gram with the suite,
    plus which suite datasets each overlap touched."""
    body_owner, gram_owner = defaultdict(set), defaultdict(set)
    for i, st in enumerate(states):
        for b in bodies(st):
            body_owner[b].add(i)
            for g in grams(b):
                gram_owner[g].add(i)
    bad, touched = set(), Counter()
    for ds, texts in suite_question_texts(suite_dir):
        hit = set()
        for t in texts:
            hit |= body_owner.get(t, set())
            for g in grams(t):
                hit |= gram_owner.get(g, set())
        if hit:
            bad |= hit
            touched[ds] += 1
    return bad, touched


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--suite")
    p.add_argument("--raw")
    a = p.parse_args()

    states = load_train(a.train)
    train_bodies = defaultdict(list)
    gram_owner = {}
    for i, st in enumerate(states):
        for b in bodies(st):
            train_bodies[b].append(i)
            for g in grams(b):
                gram_owner.setdefault(g, i)
    print(f"training rows: {len(states):,}   distinct text fields: {len(train_bodies):,}   13-grams: {len(gram_owner):,}")

    if a.raw:
        raw = Path(a.raw)
        test_texts = {}
        with open(raw / "banking77/test.csv", encoding="utf-8") as f:
            test_texts["BANKING77 test"] = [r["text"] for r in csv.DictReader(f)]
        clinc = json.load(open(raw / "clinc150/data_full.json"))
        test_texts["CLINC150 test+oos_test"] = [t for s in ("test", "oos_test") for t, _ in clinc[s]]
        for name, texts in test_texts.items():
            hits = [t for t in texts if norm(t) in train_bodies]
            print(f"[source] {name}: {len(hits)} of {len(texts)} test utterances appear verbatim in training")
            for t in hits[:10]:
                print("    ", repr(t))

    if a.suite:
        exact, gram_hits = Counter(), Counter()
        examples = defaultdict(list)
        n_rows = 0
        for path in sorted(Path(a.suite).glob("*rows.jsonl.gz")):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    n_rows += 1
                    ds = r["_evaluation"].get("dataset", r.get("family"))
                    # Question text only: state and instructions, split into lines. Option descriptions
                    # are excluded because label names ("do you have pets") collide with short utterances.
                    raw = list(strings(r["state"])) + [q.get("instructions", "") for q in r["questions"].values()]
                    texts = [norm(PREFIX.sub("", l.strip())) for t in raw for l in str(t).split("\n")]
                    texts = [t for t in texts if len(t.split()) >= 4]
                    hit_exact = any(t in train_bodies for t in texts)
                    hit_gram = hit_exact or any(g in gram_owner for t in texts for g in grams(t))
                    if hit_exact:
                        exact[ds] += 1
                    if hit_gram:
                        gram_hits[ds] += 1
                        if len(examples[ds]) < 3:
                            t = next((t for t in texts if t in train_bodies or grams(t) & gram_owner.keys()), "")
                            owner = train_bodies.get(t, [None])[0]
                            if owner is None:
                                owner = next(gram_owner[g] for g in grams(t) if g in gram_owner)
                            examples[ds].append((r["_evaluation"]["run_id"], t[:120], states[owner][:120]))
        print(f"\nsuite rows scanned: {n_rows:,}")
        if not gram_hits:
            print("no suite row shares an exact text field or any 13-word window with the training data")
        for ds in sorted(gram_hits, key=gram_hits.get, reverse=True):
            print(f"[suite] {ds}: exact-field matches {exact[ds]}, rows sharing a 13-gram {gram_hits[ds]}")
            for rid, suite_t, train_t in examples[ds]:
                print(f"    {rid}\n      suite: {suite_t!r}\n      train: {train_t!r}")


if __name__ == "__main__":
    main()
