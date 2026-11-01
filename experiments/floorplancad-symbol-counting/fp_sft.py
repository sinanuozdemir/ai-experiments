#!/usr/bin/env python3
"""Serverless vision LoRA SFT for GLM 5.3 Flash on FloorPlanCAD count / presence questions (train buildings only),
then score the test set in-session and promote the adapter."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BASE_MODEL = "accounts/fireworks/models/glm-5p3-flash"
TOKENIZER_MODEL = "zai-org/GLM-5.3-Flash"
RENDERER_NAME = "glm53_flash"


def target(r: dict) -> str:
    from fp_common import CLASSES

    noun = CLASSES[r["cls"]]
    if r["kind"] == "count":
        return f"I count {r['answer']} {noun if r['answer'] != 1 else r['cls']} in this plan.\n\n{r['answer']}"
    return f"{'There is at least one' if r['answer'] == 'yes' else 'There are no'} {noun if r['answer'] == 'no' else r['cls']} in this plan.\n\n{r['answer']}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="glm_sft")
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import fp_common as f
    from fw_utils import ensure_cookbook, sampling_prompt

    ensure_cookbook()
    from fireworks import Fireworks
    from fireworks.training import sdk
    from fireworks.training.sdk import FiretitanServiceClient, FireworksClient
    from training.renderer import get_text_content
    from training.renderer.tokenizer import get_tokenizer
    from training.utils.supervised import build_renderer, render_messages_to_datum

    train = f.load_split("train")
    random.Random(args.seed).shuffle(train)
    train = train[: args.n_train]
    test = f.load_split("test")
    f.prefetch(train + test)
    run_id = time.strftime(f"fp-off-{args.tag.replace('_', '-')}-%Y%m%d-%H%M%S")
    run_dir = HERE / "sft_logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_id={run_id} train={len(train)} test={len(test)}", flush=True)

    tok = get_tokenizer(TOKENIZER_MODEL)
    renderer = build_renderer(tok, TOKENIZER_MODEL, RENDERER_NAME)
    datums = []
    for i, r in enumerate(train):
        d = render_messages_to_datum(f.messages(r) + [{"role": "assistant", "content": target(r)}], renderer=renderer,
                                     max_seq_len=args.max_seq_len)
        if len(d.token_ids) <= args.max_seq_len and any(w > 0 for w in d.token_weights):
            dd = d.datum
            datums.append(sdk.Datum(model_input=dd.model_input.model_dump(),
                                    loss_fn_inputs={k: v.tolist() for k, v in dd.loss_fn_inputs.items()}))
        if (i + 1) % 500 == 0:
            print(f"  rendered {i + 1}/{len(train)}", flush=True)
    print(f"datums={len(datums)}", flush=True)

    service = FiretitanServiceClient(
        api_key=os.environ["FIREWORKS_API_KEY"],
        base_url=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai").rstrip("/") + "/training/v1/serverless",
    )
    metrics = (run_dir / "metrics.jsonl").open("a")
    try:
        for attempt in range(1, 9):
            try:
                tc = service.create_lora_training_client(base_model=BASE_MODEL, rank=args.lora_rank, alpha=32,
                                                         train_mlp=True, train_attn=True, train_unembed=True)
                break
            except Exception as e:  # noqa: BLE001
                status = getattr(e, "status_code", None)
                if status not in (429, 500, 502, 503, 504) or attempt == 8:
                    raise
                time.sleep(min(30 * 2 ** (attempt - 1), 600))
        adam = sdk.AdamParams(learning_rate=args.learning_rate, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, grad_clip_norm=1.0)
        rng, step = random.Random(args.seed), 0
        for epoch in range(args.epochs):
            order = list(range(len(datums)))
            rng.shuffle(order)
            for s in range(0, len(order), args.batch_size):
                fb = tc.forward_backward([datums[i] for i in order[s:s + args.batch_size]], "cross_entropy").result()
                tc.optim_step(adam).result()
                step += 1
                m = dict(getattr(fb, "metrics", None) or {})
                toks = m.get("response_tokens") or m.get("num_loss_tokens") or 1.0
                rec = {"step": step, "epoch": epoch + 1, "loss": round(m.get("loss:sum", 0.0) / max(float(toks), 1.0), 4)}
                metrics.write(json.dumps(rec) + "\n"); metrics.flush()
                if step % 10 == 0 or step == 1:
                    print(json.dumps(rec), flush=True)

        snap = tc.save_weights_for_sampler("fp-tuned").result().path
        sampler = service.create_sampling_client(model_path=snap, tokenizer=tok)
        params = sdk.SamplingParams(max_tokens=1024, temperature=0.0, stop=renderer.get_stop_sequences())
        print(f"scoring test n={len(test)}…", flush=True)
        futs = []
        for r in test:
            sp, imgs, _ = sampling_prompt(renderer, tok, f.messages(r))
            futs.append(sampler.sample(prompt=sp, num_samples=1, sampling_params=params, images=imgs))
        with f.results_path(args.tag).open("w") as out:
            for i, (r, fut) in enumerate(zip(test, futs)):
                text = get_text_content(renderer.parse_response(list(fut.result().sequences[0].tokens))[0])
                pred = f.parse(r["kind"], text)
                out.write(json.dumps({"id": r["id"], "model": args.tag, "kind": r["kind"], "cls": r["cls"], "building": r["building"],
                                      "answer": r["answer"], "pred": pred, **f.grade(r, pred), "text": text[-1200:],
                                      "error": None, "cost_usd": 0.0, "latency_s": 0.0}) + "\n")
                if (i + 1) % 200 == 0:
                    print(f"  scored {i + 1}/{len(test)}", flush=True)
        res = f.load_results(args.tag)
        print(f"{args.tag}: accuracy {sum(x['score'] for x in res.values()) / len(res):.1%}", flush=True)

        fw = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"], base_url=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai"))
        acct = next(iter(Fireworks().accounts.list(page_size=1))).name.split("/")[-1]
        cks = [r for r in fw.list_training_session_checkpoints(service.training_session_name)
               if r.get("promotable") and "fp-tuned" in (r.get("checkpointName") or r.get("name") or "")]
        if cks:
            model_id = run_id[:63].rstrip("-")
            fw.promote_session_checkpoint(name=max(cks, key=lambda r: r.get("createTime", ""))["name"],
                                          output_model_id=model_id, base_model=BASE_MODEL)
            print(f"promoted accounts/{acct}/models/{model_id}", flush=True)
    finally:
        metrics.close()
        service.close()
        print("serverless session closed", flush=True)


if __name__ == "__main__":
    main()
