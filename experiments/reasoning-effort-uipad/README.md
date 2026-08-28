# Does reasoning effort actually help? (UiPad vision QA)

A **minimal** UiPad eval that asks: does cranking up a model's reasoning-effort dial actually make it more accurate?

The point: **more reasoning effort does not always buy better pass@1 on a task.**

## What it does

Sweeps several models across their reasoning-effort ladders on the [macpaw-research/UiPad](https://huggingface.co/datasets/macpaw-research/UiPad) dataset (macOS screenshot QA: counting, yes/no, string readout, click coordinates), then scores everything and plots it.

Models compared:

- **Kimi K3** (Fireworks, open) - `none`, `low`, `high`, `max`
- **Claude Opus 4.8** (Anthropic) - `low`, `medium`, `high`, `xhigh`, `max`
- **GPT-5.6 Sol** (OpenAI) - `none`, `low`, `medium`, `high`, `xhigh`, `max`
- **GPT-5.5** (OpenAI) - `none`, `low`, `medium`, `high`, `xhigh`

## Pipeline

1. **Sweep** - run each model at each reasoning effort, `N_SAMPLES` draws per question at `temperature > 0`; store the raw model output.
2. **Extract** - a separate LLM (**Muse Glimmer** on Fireworks) pulls the final answer out of each raw response using **structured outputs** (a JSON schema per answer type). The extracted answer is stored. The extractor LLM does not judge; it only normalizes free-text into a typed value.
3. **Judge programmatically** - compare the *extracted* answer to gold with plain deterministic rules (number ±1%, yes/no, string equality, coordinates IoU ≥ 0.5). **No LLM decides correctness.**
4. **Graphs** - pass@1 vs effort, token spend (reasoning vs answer), efficiency frontiers, and a per-task-type breakdown.

The LLM's only job in grading is **extraction** (structured), not judging — so the results aren't a brittle-regex artifact or an opaque LLM-judge.

## Metrics used

**pass@1** — the chance a single response is correct (mean single-sample success over `N_SAMPLES` draws). It's the conservative, standard choice: no best-of-k inflation. The extra samples only tighten the error bars.

## Run it

From the repo root:

```bash
cp .env.example .env    # add FIREWORKS_API_KEY (OpenAI / Anthropic optional)
pip install -r requirements.txt
```

`FIREWORKS_API_KEY` is required (Kimi K3 + the Muse Glimmer extractor). `OPENAI_API_KEY` (GPT-5.6 Sol, GPT-5.5) and `ANTHROPIC_API_KEY` (Claude Opus 4.8) are **used but optional** — if a key is unset, those models are simply skipped and the run continues with whatever keys are present.

Then open `uipad_reasoning_sweep.ipynb` and run top to bottom. The rollout resumes from `uipad_results.jsonl`, so interrupted runs pick up where they left off. The first run downloads ~280MB of screenshots (cached under `data/`).

Both `data/` and `uipad_results.jsonl` are gitignored.
