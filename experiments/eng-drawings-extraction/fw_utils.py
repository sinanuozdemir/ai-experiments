"""Shared helpers: background training jobs, serverless vision sampling prompts, cookbook setup, and plot styling."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent

# ── Plot styling ──────────────────────────────────────────────────────────────

BG = "#F8F7F5"
INK = "#260015"
MUTED = "#8A6F7B"
GRID = "#E9E2E0"
MAGENTA = "#BE1A55"
CORAL = "#EE7D5D"
ORANGE = "#FFA57C"
PEACH = "#FED2A1"
BLUSH = "#F5E2E6"
SUNSET = LinearSegmentedColormap.from_list("sunset", [BG, PEACH, ORANGE, CORAL, MAGENTA, INK])
FIGS = HERE / "figures"


def apply_theme() -> None:
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "font.family": "sans-serif", "font.size": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.axisbelow": True, "legend.frameon": False,
    })


def _title(fig, title: str, subtitle: str = "") -> None:
    """Title block pinned in inches from the top, then reserve room for it."""
    h = fig.get_figheight()
    fig.text(0.01, 1 - 0.12 / h, title, ha="left", va="top", fontsize=19, family="serif", color=INK)
    if subtitle:
        fig.text(0.01, 1 - 0.52 / h, subtitle, ha="left", va="top", fontsize=10.5, color=MUTED)
    fig.subplots_adjust(top=1 - 1.15 / h)


def show(fig, name: str | None = None) -> None:
    """Display a figure; with ``name``, also save it to figures/<name>.png for the README."""
    if name:
        FIGS.mkdir(exist_ok=True)
        fig.savefig(FIGS / f"{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def loss_curve(metrics_path, title: str, subtitle: str = "", save: str | None = None) -> None:
    """Training loss per optimizer step, with a smoothed trend."""
    rows = [json.loads(l) for l in Path(metrics_path).read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "loss" in r]
    if not rows:
        print("no loss rows yet")
        return
    steps = np.array([r["step"] for r in rows])
    loss = np.array([r["loss"] for r in rows])
    k = max(1, len(loss) // 12)
    smooth = np.convolve(loss, np.ones(k) / k, mode="valid")
    fig, ax = plt.subplots(figsize=(11, 4.4))
    _title(fig, title, subtitle)
    ax.fill_between(steps, loss, color=PEACH, alpha=0.55, lw=0)
    ax.plot(steps, loss, color=ORANGE, lw=1, alpha=0.9)
    ax.plot(steps[k - 1:], smooth, color=MAGENTA, lw=2.8, label=f"{k}-step moving average")
    ax.annotate(f"{loss[0]:.2f}", (steps[0], loss[0]), xytext=(6, 6), textcoords="offset points", fontsize=10, color=INK)
    ax.annotate(f"{smooth[-1]:.2f}", (steps[-1], smooth[-1]), xytext=(6, 6), textcoords="offset points",
                fontsize=11, fontweight="bold", color=MAGENTA)
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("cross-entropy on answer tokens")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right")
    show(fig, save)


def mcnemar_figure(rows: list[tuple[str, str, str, dict]], unit: str, n_items: int, save: str | None = None) -> None:
    """McNemar in one picture: ignore the agreements, then ask if the disagreements split 50 / 50 like a coin flip.

    Each ``rows`` entry is (A name, B name, verdict, compare() result with mcnemar.only_a / only_b).
    """
    fig = plt.figure(figsize=(12, 1.9 + 1.55 * len(rows)))
    _title(fig, "McNemar's test: who wins the disagreements?",
           f"Both systems answer the same {n_items:,} {unit}s. When they agree, it tells us nothing about which is better, "
           f"so only the {unit}s where exactly one of them is right count.")
    ax = fig.add_axes([0.03, 0.06, 0.94, fig.subplotpars.top - 0.1])
    ax.set_xlim(0, 100); ax.set_ylim(len(rows) - 0.4, -0.75); ax.axis("off")
    ax.text(50, -0.55, "a fair coin would split them 50 / 50", ha="center", fontsize=10, color=MUTED)
    for i, (a_name, b_name, verdict, r) in enumerate(rows):
        a, b = r["mcnemar"]["only_a"], r["mcnemar"]["only_b"]
        fa = 100 * a / (a + b)
        ax.barh(i, fa, height=0.42, color=MAGENTA)
        ax.barh(i, 100 - fa, left=fa, height=0.42, color=PEACH)
        ax.text(1.2, i, f"only {a_name} right: {a}", va="center", color="white", fontsize=10.5, fontweight="bold")
        ax.text(98.8, i, f"only {b_name} right: {b}", va="center", ha="right", color="white" if fa > 80 else INK,
                fontsize=10.5, fontweight="bold")
        ax.text(0, i - 0.36, f"{a_name} vs {b_name}: {a + b} disagreements", fontsize=11, color=INK)
        ax.text(100, i - 0.36, verdict + f"  (p = {r['mcnemar']['p']:.2g})", fontsize=11, ha="right",
                color=MAGENTA if r["mcnemar"]["p"] < 0.05 else MUTED, fontweight="bold")
        ax.plot([50, 50], [i - 0.27, i + 0.27], color=INK, lw=1.5, ls=(0, (3, 2)))
    show(fig, save)


def bootstrap_figure(rows: list[tuple[str, dict]], n_clusters: int, cluster: str, metric: str,
                     save: str | None = None, seed: int = 7) -> None:
    """The bootstrap in one picture: resample the test set with replacement, recompute the gap, repeat, keep the middle 95%.

    Each ``rows`` entry is (label, compare() result with boot.delta / lo / hi / samples).
    """
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(12, 4.6 + 1.25 * len(rows)))
    _title(fig, "Bootstrap: how much could the gap move with a different test set?",
           f"We can't collect new {cluster}s, so we fake new test sets by drawing from our own {n_clusters} {cluster}s.")
    show_n = 24
    # step 1 and 2: the original set and one resample (showing 24 for readability)
    for row, (label, picks) in enumerate([("1. Our test set", np.arange(show_n)),
                                          ("2. One fake test set: draw the same number, with replacement",
                                           np.sort(rng.integers(0, show_n, show_n)))]):
        ax = fig.add_axes([0.03, 0.80 - row * 0.16 * 5.85 / (4.6 + 1.25 * len(rows)), 0.94, 0.07 * 5.85 / (4.6 + 1.25 * len(rows))])
        ax.set_xlim(-0.5, show_n + 0.5); ax.set_ylim(-0.6, 1.4); ax.axis("off")
        ax.text(-0.5, 1.15, label, fontsize=11, color=INK)
        counts = np.bincount(picks, minlength=show_n)
        for k in range(show_n):
            c = counts[k] if row else 1
            ax.add_patch(plt.Rectangle((k + 0.08, -0.45), 0.84, 1.1, color=[BLUSH, PEACH, CORAL, MAGENTA, INK][min(c, 4)], lw=0))
            ax.text(k + 0.5, 0.1, (f"×{c}" if row else str(k + 1)) if c else "–", ha="center", va="center", fontsize=8,
                    color="white" if c >= 2 else INK)
        if row:
            ax.text(show_n + 0.4, 0.1, f"some {cluster}s twice, some not at all", va="center", fontsize=9.5, color=MUTED)
        else:
            ax.text(show_n + 0.4, 0.1, f"… {n_clusters} {cluster}s", va="center", fontsize=9.5, color=MUTED)
    # step 3: the 5,000 gaps as dots, middle 95% bracketed
    top = 0.80 - 2 * 0.16 * 5.85 / (4.6 + 1.25 * len(rows))
    ax = fig.add_axes([0.03, 0.08, 0.94, top - 0.04])
    ax.axis("off")
    allv = np.concatenate([np.array(r["boot"]["samples"]) * 100 for _, r in rows] + [[0.0]])
    lo_x, hi_x = allv.min() - 1, allv.max() + 1
    ax.set_xlim(lo_x, hi_x); ax.set_ylim(len(rows) - 0.3, -1.0)
    ax.text(lo_x, -0.75, "3. Score both systems on each fake set and take the gap. Repeat 5,000 times. Keep the middle 95%.",
            fontsize=11, color=INK)
    ax.axvline(0, ymin=0.02, ymax=0.82, color=INK, lw=1.2, ls=(0, (3, 2)))
    ax.text(0, len(rows) - 0.35, "no difference", ha="center", fontsize=9.5, color=INK)
    for i, (label, r) in enumerate(rows):
        v = np.array(r["boot"]["samples"]) * 100
        sub = rng.choice(v, 400, replace=False)
        ax.scatter(sub, i + rng.uniform(-0.13, 0.13, len(sub)), s=7, color=PEACH, lw=0)
        lo, hi = r["boot"]["lo"] * 100, r["boot"]["hi"] * 100
        sig = lo > 0 or hi < 0
        ax.plot([lo, hi], [i + 0.24, i + 0.24], color=MAGENTA if sig else MUTED, lw=3)
        for x in (lo, hi):
            ax.plot([x, x], [i + 0.17, i + 0.31], color=MAGENTA if sig else MUTED, lw=3)
        ax.text((lo + hi) / 2, i + 0.38, f"middle 95%: {lo:+.1f} to {hi:+.1f} points", ha="center", va="top", fontsize=9.5,
                color=MAGENTA if sig else MUTED)
        ax.text(lo_x, i - 0.3, label, fontsize=11, color=INK)
        behind = int((v < 0).sum()) if r["boot"]["delta"] >= 0 else int((v > 0).sum())
        verdict = (f"ahead in all {len(v):,} fake test sets" if behind == 0 else
                   f"flips sign in {behind:,} of {len(v):,} ({behind / len(v):.0%})")
        ax.text(hi_x, i - 0.3, verdict, fontsize=11, ha="right", color=MAGENTA if sig else MUTED, fontweight="bold")
    ax.text(hi_x, len(rows) - 0.35, f"gap in {metric} (points) →", ha="right", fontsize=9.5, color=MUTED)
    show(fig, save)


# ── Fireworks training cookbook ───────────────────────────────────────────────

def ensure_cookbook() -> Path:
    """Clone fw-ai/cookbook into ./cookbook (or use $FW_COOKBOOK) and pip-install its training package once."""
    cook = Path(os.environ.get("FW_COOKBOOK", HERE / "cookbook"))
    spec = importlib.util.find_spec("training")
    if spec and spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).parent
    if not (cook / "training").exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/fw-ai/cookbook.git", str(cook)], check=True)
    # Parallel jobs re-running this editable install race on the same .pth file.
    if importlib.util.find_spec("training") is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(cook / "training")], check=True)
    return cook


def sampling_prompt(renderer, tokenizer, messages: list[dict]):
    """Serverless sampler prompt for a vision chat: returns (sdk_prompt, images, expanded_prompt).

    The sampler takes text tokens only, so each image becomes one placeholder token and the images travel separately.
    """
    from fireworks.training import sdk
    from training.utils.rl.rollout.renderer import build_multimodal_completions_prompt_token_ids
    from training.utils.supervised import normalize_messages

    expanded = renderer.build_generation_prompt(normalize_messages(messages))
    if not any(type(c).__name__ in ("ImageChunk", "ImageAssetPointerChunk") for c in expanded.chunks):
        return sdk.ModelInput.from_ints(list(expanded.to_ints())), None, expanded
    ids, images = build_multimodal_completions_prompt_token_ids(messages, expanded, tokenizer, renderer=renderer)
    return sdk.ModelInput.from_ints(ids), images, expanded


# ── Background training jobs (survive kernel restarts: one pid/log pair per tag) ──

LOGS = HERE / "sft_logs"
RESULTS = HERE / "results"
DONE_MARKER = "serverless session closed"


def _paths(tag: str) -> tuple[Path, Path]:
    LOGS.mkdir(exist_ok=True)
    return LOGS / f"{tag}.pid", LOGS / f"{tag}.log"


def alive(tag: str) -> bool:
    pid_path, _ = _paths(tag)
    try:
        pid = int(pid_path.read_text())
    except (FileNotFoundError, ValueError):
        return False
    # A finished child stays a zombie until reaped, and os.kill(pid, 0) still succeeds on zombies.
    try:
        if os.waitpid(pid, os.WNOHANG)[0] == pid:
            return False
    except ChildProcessError:
        pass
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return bool(state) and not state.startswith("Z")


def start(tag: str, args: list[str]) -> None:
    """Launch ``python -u <args>`` in the background unless it's running or results/results_<tag>.jsonl exists."""
    pid_path, log_path = _paths(tag)
    if alive(tag):
        print(f"{tag} already running, pid {pid_path.read_text()}")
        return
    if (RESULTS / f"results_{tag}.jsonl").exists():
        print(f"{tag} already done: results/results_{tag}.jsonl exists. Delete it to run again.")
        return
    proc = subprocess.Popen([sys.executable, "-u", *args], cwd=HERE, stdout=log_path.open("w"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    pid_path.write_text(str(proc.pid))
    print(f"started {tag}, pid {proc.pid} · log sft_logs/{log_path.name}")


def watch(tag: str, every: int = 20, lines: int = 12) -> None:
    """Tail the log until the job exits. Safe to interrupt and re-run."""
    from IPython.display import clear_output

    _, log_path = _paths(tag)
    while True:
        tail = log_path.read_text().splitlines()[-lines:] if log_path.exists() else []
        clear_output(wait=True)
        print("\n".join(tail))
        if not alive(tag) or DONE_MARKER in "\n".join(tail):
            print(f"\n— {tag} finished —")
            return
        time.sleep(every)
