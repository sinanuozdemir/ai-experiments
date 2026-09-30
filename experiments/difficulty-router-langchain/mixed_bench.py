"""Mixed GSM8K / MATH Level 3 / AIME question set, identical to difficulty_router_mixed.ipynb (same seed, same order)."""

from __future__ import annotations

import random

from datasets import load_dataset

MATH_SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus"]


def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    depth, k = 0, j
    while k < len(s):
        depth += {"{": 1, "}": -1}.get(s[k], 0)
        if depth == 0:
            return s[j + 1:k]
        k += 1
    return None


def to_float(s):
    try:
        return float(str(s).replace(",", "").replace("$", "").replace("\\!", "").strip())
    except (TypeError, ValueError):
        return None


def load_mixed(n_per_bucket: int = 50, seed: int = 0) -> dict[str, dict]:
    """qid -> dict(bucket, question, gold). Must match the notebook's sampling exactly."""
    rng = random.Random(seed)
    questions = {}
    gsm = load_dataset("openai/gsm8k", "main", split="test")
    for i in rng.sample(range(len(gsm)), n_per_bucket):
        questions[f"gsm8k-{i}"] = dict(bucket="easy", question=gsm[i]["question"],
                                       gold=to_float(gsm[i]["answer"].split("####")[-1]))
    pool = []
    for subject in MATH_SUBJECTS:
        for j, ex in enumerate(load_dataset("EleutherAI/hendrycks_math", subject, split="test")):
            gold = to_float(last_boxed(ex["solution"]))
            if ex["level"] == "Level 3" and gold is not None:
                pool.append((f"math-{subject}-{j}", ex["problem"], gold))
    for qid, q, gold in rng.sample(pool, n_per_bucket):
        questions[qid] = dict(bucket="medium", question=q, gold=gold)
    aime = load_dataset("AI-MO/aimo-validation-aime", split="train")
    for i in rng.sample(range(len(aime)), n_per_bucket):
        questions[f"aime-{aime[i]['id']}"] = dict(bucket="hard", question=aime[i]["problem"],
                                                  gold=to_float(aime[i]["answer"]))
    return questions


def load_hard_mix(n_math: int = 50, n_aime: int = 50, seed: int = 0) -> dict[str, dict]:
    """Harder set. Buckets are relative: easy = MATH Level 5, medium = AIME 2022-24, hard = AIME 2025 + HMMT Feb 2025.

    Only questions with a plain numeric gold answer are kept, so the existing '#### <number>' grader applies
    (HMMT keeps its ~14 integer-answer problems; its fraction/radical answers are dropped).
    """
    rng = random.Random(seed)
    questions = {}
    pool = []
    for subject in MATH_SUBJECTS:
        for j, ex in enumerate(load_dataset("EleutherAI/hendrycks_math", subject, split="test")):
            gold = to_float(last_boxed(ex["solution"]))
            if ex["level"] == "Level 5" and gold is not None:
                pool.append((f"math5-{subject}-{j}", ex["problem"], gold))
    for qid, q, gold in rng.sample(pool, n_math):
        questions[qid] = dict(bucket="easy", question=q, gold=gold)
    aime = load_dataset("AI-MO/aimo-validation-aime", split="train")
    for i in rng.sample(range(len(aime)), n_aime):
        questions[f"aime-{aime[i]['id']}"] = dict(bucket="medium", question=aime[i]["problem"],
                                                  gold=to_float(aime[i]["answer"]))
    for ex in load_dataset("MathArena/aime_2025", split="train"):
        questions[f"aime25-{ex['problem_idx']}"] = dict(bucket="hard", question=ex["problem"], gold=float(ex["answer"]))
    for ex in load_dataset("MathArena/hmmt_feb_2025", split="train"):
        gold = to_float(ex["answer"])
        if gold is not None:
            questions[f"hmmt25-{ex['problem_idx']}"] = dict(bucket="hard", question=ex["problem"], gold=gold)
    return questions
