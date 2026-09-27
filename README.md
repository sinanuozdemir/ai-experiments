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

| Experiment name | Question it answers | Experiment folder |
|---|---|---|
| Reasoning effort on UiPad vision QA | Does cranking up a model's reasoning effort actually improve accuracy? | [experiments/reasoning-effort-uipad](experiments/reasoning-effort-uipad/) |
| Scion: Jev-style decision classifiers | Can small fine-tuned open models match Jev/Tev as calibrated decision engines on the Decision Index? | [experiments/scion](experiments/scion/) |
| Difficulty router as LangChain middleware | Does routing easy tasks to a cheap model and hard ones to a strong model cut cost per successful task without losing accuracy? | [experiments/difficulty-router-langchain](experiments/difficulty-router-langchain/) |

Always more to come.
