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
    """McNemar in one picture per comparison: the 2x2 table (agreements ignored), then the disagreement split vs a coin flip.

    Each ``rows`` entry is (A name, B name, verdict, compare() result with mcnemar.only_a / only_b / both_right / both_wrong).
    """
    fig = plt.figure(figsize=(13, 1.6 + 2.7 * len(rows)))
    _title(fig, "McNemar's test: who wins the disagreements?",
           f"Both systems answer the same {n_items:,} {unit}s. Sort them into a 2×2 table; only the two cells where they "
           f"disagree count.")
    top = fig.subplotpars.top
    h = (top - 0.04) / len(rows)
    for i, (a_name, b_name, verdict, r) in enumerate(rows):
        m = r["mcnemar"]
        a, b = m["only_a"], m["only_b"]
        y0 = top - (i + 1) * h
        # 2x2 table
        ax = fig.add_axes([0.03, y0 + 0.02, 0.36, h - 0.07])
        ax.set_xlim(-1.25, 2); ax.set_ylim(2.55, -0.6); ax.axis("off")
        ax.text(1, -0.45, f"{b_name}", ha="center", fontsize=10.5, color=INK, fontweight="bold")
        ax.text(0.5, -0.12, "right", ha="center", fontsize=9.5, color=MUTED)
        ax.text(1.5, -0.12, "wrong", ha="center", fontsize=9.5, color=MUTED)
        ax.text(-1.2, 1, f"{a_name}", va="center", fontsize=10.5, color=INK, fontweight="bold")
        ax.text(-0.08, 0.5, "right", ha="right", va="center", fontsize=9.5, color=MUTED)
        ax.text(-0.08, 1.5, "wrong", ha="right", va="center", fontsize=9.5, color=MUTED)
        cells = [(0, 0, m["both_right"], "both right", BLUSH, MUTED), (1, 0, a, f"only {a_name}", MAGENTA, "white"),
                 (0, 1, b, f"only {b_name}", PEACH, INK), (1, 1, m["both_wrong"], "both wrong", BLUSH, MUTED)]
        for cx, cy, n, lab, col, tc in cells:
            ax.add_patch(plt.Rectangle((cx + 0.03, cy + 0.03), 0.94, 0.94, color=col, lw=0))
            ax.text(cx + 0.5, cy + 0.42, f"{n:,}", ha="center", va="center", fontsize=14, color=tc, fontweight="bold")
            ax.text(cx + 0.5, cy + 0.72, lab + ("  (ignored)" if col == BLUSH else ""), ha="center", va="center",
                    fontsize=8.5, color=tc)
        # disagreement split
        ax = fig.add_axes([0.44, y0 + 0.02, 0.54, h - 0.07])
        ax.set_xlim(0, 100); ax.set_ylim(1.2, -1.0); ax.axis("off")
        fa = 100 * a / (a + b)
        ax.text(0, -0.75, f"Only the {a + b:,} disagreements: a fair coin would split them 50 / 50", fontsize=11, color=INK)
        ax.barh(0, fa, height=0.55, color=MAGENTA)
        ax.barh(0, 100 - fa, left=fa, height=0.55, color=PEACH)
        ax.plot([50, 50], [-0.38, 0.38], color=INK, lw=1.5, ls=(0, (3, 2)))
        ax.text(50, 0.55, "50 / 50", ha="center", va="top", fontsize=9, color=INK)
        ax.text(1.2, 0, f"only {a_name}: {a}", va="center", color="white", fontsize=10.5, fontweight="bold")
        if fa > 72:
            ax.text(100, 0.55, f"only {b_name}: {b}", va="top", ha="right", color=INK, fontsize=10.5, fontweight="bold")
        else:
            ax.text(98.8, 0, f"only {b_name}: {b}", va="center", ha="right", color=INK, fontsize=10.5, fontweight="bold")
        ax.text(0, 0.95, f"{verdict}  (p = {m['p']:.2g})", fontsize=11.5, color=MAGENTA if m["p"] < 0.05 else MUTED,
                fontweight="bold")
    show(fig, save)


def bootstrap_figure(rows: list[tuple[str, str, dict]], n_clusters: int, cluster: str, metric: str,
                     save: str | None = None, seed: int = 7) -> None:
    """The bootstrap in one picture: resample the test set with replacement, recompute the gap, repeat 5,000 times.

    Each ``rows`` entry is (A name, B name, compare() result with boot.delta / samples).
    """
    rng = np.random.default_rng(seed)
    fig = plt.figure(figsize=(12, 8.2))
    _title(fig, "Bootstrap: would the gap survive a different test set?",
           f"We can't collect new {cluster}s, so we make 5,000 fake test sets out of our own {n_clusters} {cluster}s.")
    show_n = 24
    for row, (label, picks) in enumerate([(f"Step 1. Our real test set ({n_clusters} {cluster}s; first {show_n} shown)", np.arange(show_n)),
                                          (f"Step 2. A fake test set: pick {n_clusters} {cluster}s at random from the real ones, allowing repeats",
                                           np.sort(rng.integers(0, show_n, show_n)))]):
        ax = fig.add_axes([0.03, 0.80 - row * 0.115, 0.94, 0.055])
        ax.set_xlim(-0.5, show_n + 0.5); ax.set_ylim(-0.6, 1.5); ax.axis("off")
        ax.text(-0.5, 1.2, label, fontsize=11, color=INK)
        counts = np.bincount(picks, minlength=show_n)
        for k in range(show_n):
            c = counts[k] if row else 1
            ax.add_patch(plt.Rectangle((k + 0.08, -0.45), 0.84, 1.1, color=[BLUSH, PEACH, CORAL, MAGENTA, INK][min(c, 4)], lw=0))
            ax.text(k + 0.5, 0.1, (f"×{c}" if row else str(k + 1)) if c else "–", ha="center", va="center", fontsize=8,
                    color="white" if c >= 2 else INK)
        if row:
            ax.text(show_n + 0.4, 0.1, "×2 = picked twice, – = left out", va="center", fontsize=9.5, color=MUTED)
    fig.text(0.03, 0.585, f"Step 3. Score both systems on the fake test set and note the gap. Do it 5,000 times. "
                          f"Each chart below is one comparison.", fontsize=11, color=INK)
    for i, (a_name, b_name, r) in enumerate(rows):
        ax = fig.add_axes([0.05 + i * 0.49, 0.1, 0.42, 0.38])
        v = np.array(r["boot"]["samples"]) * 100
        bins = np.linspace(min(v.min(), -0.5), max(v.max(), 0.5), 45)
        hist, edges = np.histogram(v, bins=bins)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.bar(centers, hist, width=edges[1] - edges[0], color=np.where(centers >= 0, MAGENTA, PEACH), lw=0)
        ax.axvline(0, color=INK, lw=1.5)
        ax.text(0, hist.max() * 1.08, " 0 = tie", fontsize=9.5, color=INK, va="bottom")
        ahead = int((v > 0).sum())
        ax.set_title(f"Comparison {i + 1}: {a_name} vs {b_name}", loc="left", fontsize=11.5, pad=26)
        ax.text(0, 1.03, f"{a_name} ahead on {ahead:,} of {len(v):,} fake test sets", transform=ax.transAxes, fontsize=10.5,
                color=MAGENTA if ahead >= 0.975 * len(v) or ahead <= 0.025 * len(v) else MUTED, fontweight="bold")
        ax.set_xlabel(f"{metric} of {a_name} minus {b_name} (points)\n← {b_name} ahead · {a_name} ahead →")
        ax.set_ylabel("number of fake test sets")
        ax.set_ylim(0, hist.max() * 1.25)
        ax.grid(axis="y", visible=False)
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
