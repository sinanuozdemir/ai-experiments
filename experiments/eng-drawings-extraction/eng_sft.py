#!/usr/bin/env python3
"""Serverless vision LoRA SFT for GLM 5.3 Flash on engineering drawings, then score the test split in-session.

Training rows match the fw-ai cookbook: user = [receipt image, INSTRUCTION], assistant = compact gold JSON.
"""

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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="glm_sft")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--max-seq-len", type=int, default=8192)
    p.add_argument("--max-eval-tokens", type=int, default=2048)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import eng_common as c
    from fw_utils import ensure_cookbook, sampling_prompt

    ensure_cookbook()
    from fireworks.training import sdk
    from fireworks.training.sdk import FiretitanServiceClient
    from training.renderer import get_text_content
    from training.renderer.tokenizer import get_tokenizer
    from training.utils.supervised import build_renderer, render_messages_to_datum

    train = c.load_split("train")[: args.n_train]
    examples = [c.messages(r) + [{"role": "assistant", "content": c.gold_text(r)}] for r in train]
    random.Random(args.seed).shuffle(examples)
    test = c.load_split("test")
    run_id = time.strftime(f"eng-{args.tag.replace('_', '-')}-%Y%m%d-%H%M%S")
    run_dir = HERE / "sft_logs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"run_id={run_id} train={len(train)} total_examples={len(examples)} test={len(test)}", flush=True)

    tok = get_tokenizer(TOKENIZER_MODEL)
    renderer = build_renderer(tok, TOKENIZER_MODEL, RENDERER_NAME)

    datums, skipped, lengths = [], 0, []
    for i, msgs in enumerate(examples):
        d = render_messages_to_datum(msgs, renderer=renderer, max_seq_len=args.max_seq_len)
        if len(d.token_ids) > args.max_seq_len or not any(w > 0 for w in d.token_weights):
            skipped += 1
            continue
        lengths.append(len(d.token_ids))
        dd = d.datum
        datums.append(sdk.Datum(model_input=dd.model_input.model_dump(),
                                loss_fn_inputs={k: v.tolist() for k, v in dd.loss_fn_inputs.items()}))
        if (i + 1) % 200 == 0:
            print(f"  rendered {i + 1}/{len(examples)}", flush=True)
    print(f"datums={len(datums)} skipped={skipped} max_tokens={max(lengths)} mean_tokens={sum(lengths)//len(lengths)}", flush=True)

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
                wait = min(30 * 2 ** (attempt - 1), 600)
                print(f"create_session got {status}; retry {attempt}/7 in {wait}s", flush=True)
                time.sleep(wait)
        print(f"session={getattr(service, 'training_session_id', None)}", flush=True)
        adam = sdk.AdamParams(learning_rate=args.learning_rate, beta1=0.9, beta2=0.999, eps=1e-8,
                              weight_decay=0.0, grad_clip_norm=1.0)
        rng = random.Random(args.seed)
        step = 0
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
                metrics.write(json.dumps(rec) + "\n")
                metrics.flush()
                if step % 5 == 0 or step == 1:
                    print(json.dumps(rec), flush=True)

        snap = tc.save_weights_for_sampler("eng-tuned").result().path
        sampler = service.create_sampling_client(model_path=snap, tokenizer=tok)
        params = sdk.SamplingParams(max_tokens=args.max_eval_tokens, temperature=0.0, stop=renderer.get_stop_sequences())
        print(f"scoring test n={len(test)}…", flush=True)
        futs = []
        for r in test:
            sp, imgs, _ = sampling_prompt(renderer, tok, c.messages(r))
            futs.append(sampler.sample(prompt=sp, num_samples=1, sampling_params=params, images=imgs))
        out = c.results_path(args.tag)
        with out.open("w") as f:
            for i, (r, fut) in enumerate(zip(test, futs)):
                text = get_text_content(renderer.parse_response(list(fut.result().sequences[0].tokens))[0])
                f.write(json.dumps({"id": r["id"], "model": args.tag, **c.receipt_f1(text, r["gold"]), "pred": text[:6000],
                                    "error": None}, ensure_ascii=False) + "\n")
                if (i + 1) % 25 == 0:
                    print(f"  scored {i + 1}/{len(test)}", flush=True)
        res = c.load_results(args.tag)
        print(f"{args.tag}: mean JSON F1 {sum(x['f1'] for x in res.values()) / len(res):.1%}", flush=True)

        # Promote before the session closes, so the adapter outlives it as a deployable model.
        from fireworks import Fireworks
        from fireworks.training.sdk import FireworksClient

        fw = FireworksClient(api_key=os.environ["FIREWORKS_API_KEY"],
                             base_url=os.environ.get("FIREWORKS_BASE_URL", "https://api.fireworks.ai"))
        acct = next(iter(Fireworks().accounts.list(page_size=1))).name.split("/")[-1]
        cks = [r for r in fw.list_training_session_checkpoints(service.training_session_name)
               if r.get("promotable") and "eng-tuned" in (r.get("checkpointName") or r.get("name") or "")]
        if cks:
            model_id = run_id[:63].rstrip("-")
            fw.promote_session_checkpoint(name=max(cks, key=lambda r: r.get("createTime", ""))["name"],
                                          output_model_id=model_id, base_model=BASE_MODEL)
            print(f"promoted accounts/{acct}/models/{model_id}", flush=True)
            reg = HERE / "promoted_models.json"
            data = json.loads(reg.read_text()) if reg.exists() else {}
            data[args.tag] = f"accounts/{acct}/models/{model_id}"
            reg.write_text(json.dumps(data, indent=2) + "\n")
    finally:
        metrics.close()
        service.close()
        print("serverless session closed", flush=True)


if __name__ == "__main__":
    main()
