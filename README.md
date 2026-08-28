# ai-experiments

A home for small, self-contained AI/LLM experiments — each one a runnable notebook with a clear question and honest charts.

## Setup

Each experiment reads API keys from a `.env` file at the repo root:

```bash
cp .env.example .env   # then fill in your keys
pip install -r requirements.txt
```

`FIREWORKS_API_KEY` is required; `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` are optional (those models are skipped if unset).

## Experiments

| Experiment | Question | Link |
|---|---|---|
| Reasoning effort on UiPad vision QA | Does cranking up a model's reasoning effort actually improve accuracy? | [experiments/reasoning-effort-uipad](experiments/reasoning-effort-uipad/) |

More to come.
