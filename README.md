# Sinan's Mad Scientist Repo

A single place for the random ideas that come to me on AI/ML stuff. The goal is for each one to be a runnable notebook so anyone can reproduce or just see the pathway + results.

## Setup

Each experiment reads API keys from a `.env` file at the repo root:

```bash
cp .env.example .env   # then fill in your keys
pip install -r requirements.txt
```

`FIREWORKS_API_KEY` is almost always required. Others I'll call out in the notebook.

## Experiments

| Experiment | Question | Link |
|---|---|---|
| Reasoning effort on UiPad vision QA | Does cranking up a model's reasoning effort actually improve accuracy? | [experiments/reasoning-effort-uipad](experiments/reasoning-effort-uipad/) |

Always more to come.
