# Reasoning effort on UiPad vision QA

**Question:** does cranking up a model's reasoning-effort dial actually make it more accurate?

**Short answer:** on this task, no — past the first step off zero reasoning, more effort buys essentially nothing while token spend and cost climb.

## What it does

Sweeps several models across their reasoning-effort ladders on the [macpaw-research/UiPad](https://huggingface.co/datasets/macpaw-research/UiPad) dataset (macOS screenshot QA: counting, yes/no, string readout, click coordinates), then scores everything and plots it.

Models compared:

- **Kimi K3** (Fireworks, open) — `none`, `low`, `high`, `max`
- **Claude Opus 4.8** (Anthropic) — `low`, `medium`, `high`, `xhigh`, `max`
- **GPT-5.6 Sol** (OpenAI) — `none`, `low`, `medium`, `high`, `xhigh`, `max`
- **GPT-5.5** (OpenAI) — `none`, `low`, `medium`, `high`, `xhigh`

## Pipeline

1. **Sweep** — run each model at each reasoning effort, `N_SAMPLES` draws per question at `temperature > 0`; store the raw output.
2. **Extract** — a separate LLM (**Muse Glimmer** on Fireworks) pulls the final answer out of each raw response using **structured outputs** (a JSON schema per answer type). The extracted answer is stored.
3. **Judge programmatically** — pure-Python comparison of the extracted answer to gold: number (±1%), yes/no, string equality, coordinates IoU ≥ 0.5. No LLM decides correctness.
4. **Charts** — pass@1 vs effort, token spend (reasoning vs answer), efficiency frontiers, and a per-task-type breakdown.

The LLM's only job in grading is **extraction** (structured), not judging — so the results aren't a brittle-regex artifact or an opaque LLM-judge.

## Metric

**pass@1** — the chance a single response is correct (mean single-sample success over `N_SAMPLES` draws). The conservative, standard headline metric; the extra samples only tighten the error bars.

## Result (full run: 228 questions)

pass@1 (%) by model x reasoning effort:

| Model | none | low | medium | high | xhigh | max |
|---|---|---|---|---|---|---|
| Kimi K3 | 83.9 | 88.5 | — | 88.8 | — | 87.2 |
| Claude Opus 4.8 | — | 87.7 | 87.5 | 87.7 | 87.2 | 87.2 |
| GPT-5.6 Sol | 87.4 | 88.3 | 88.3 | 88.3 | 88.8 | 88.4 |
| GPT-5.5 | 84.6 | 86.5 | 88.0 | 87.4 | 87.7 | — |

The only meaningful movement is the first step off `none`. Beyond that, higher effort (and far more reasoning tokens and cost) buys ~0 additional accuracy — and in a couple of cases it slightly regresses.

## Run it

From the repo root:

```bash
cp .env.example .env    # add FIREWORKS_API_KEY (OpenAI / Anthropic optional)
pip install -r requirements.txt
```

Then open `uipad_reasoning_sweep.ipynb` and run top to bottom. The rollout resumes from `uipad_results.jsonl`, so interrupted runs pick up where they left off. The first run downloads ~280MB of screenshots (cached under `data/`).

Both `data/` and `uipad_results.jsonl` are gitignored.
