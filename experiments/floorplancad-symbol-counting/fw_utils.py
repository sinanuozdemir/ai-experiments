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


def significance_panels(rows: list[tuple[str, dict]], unit: str, cluster: str, metric: str,
                        title: str, subtitle: str = "", save: str | None = None) -> None:
    """One row per comparison: exact-McNemar null distribution (left) and cluster-bootstrap gap distribution (right).

    Each ``rows`` entry is (label, compare() result with mcnemar.only_a/only_b and boot.delta/lo/hi/samples).
    """
    from math import comb

    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 3.3 * len(rows) + 1.3), squeeze=False,
                             gridspec_kw={"wspace": 0.15, "hspace": 0.9})
    _title(fig, title, subtitle)
    for (label, r), (ax1, ax2) in zip(rows, axes):
        a, b = r["mcnemar"]["only_a"], r["mcnemar"]["only_b"]
        n = a + b
        ks = np.arange(n + 1)
        pmf = np.array([comb(n, k) for k in ks], float) / 2.0 ** n
        tail = (ks <= min(a, b)) | (ks >= max(a, b))
        ax1.bar(ks, pmf, width=1.0, color=np.where(tail, MAGENTA, PEACH), lw=0)
        ax1.axvline(a, color=INK, lw=2)
        ax1.set_ylim(0, pmf.max() * 1.5)
        ax1.annotate(f"observed: {a}", (a, pmf.max() * 0.6), xytext=(-6, 0), textcoords="offset points", ha="right",
                     fontsize=9.5, color=INK, fontweight="bold")
        p = r["mcnemar"]["p"]
        ax1.set_title(f"{label}\nMcNemar: {a} vs {b} disagreements · p = {p:.2g}", loc="left", fontsize=11)
        ax1.set_xlabel(f"{unit}s only A got right, out of {n} disagreements\n(bars: what a coin flip would give if A and B were equally good)")
        from matplotlib.patches import Patch
        ax1.legend(handles=[Patch(color=PEACH, label="coin-flip outcomes"),
                            Patch(color=MAGENTA, label="as or more extreme than observed (total area = p)")],
                   loc="upper left", fontsize=9)
        ax1.set_yticks([])
        lo_x, hi_x = max(0, min(a, b, n // 2 - 4 * np.sqrt(n) / 2) - 3), min(n, max(a, b, n // 2 + 4 * np.sqrt(n) / 2) + 3)
        ax1.set_xlim(lo_x, hi_x)

        s = np.array(r["boot"]["samples"]) * 100
        lo, hi, d = r["boot"]["lo"] * 100, r["boot"]["hi"] * 100, r["boot"]["delta"] * 100
        counts, _, _ = ax2.hist(s, bins=50, color=PEACH, lw=0)
        ax2.set_ylim(0, counts.max() * 1.6)
        ax2.axvspan(lo, hi, color=CORAL, alpha=0.25, lw=0, label=f"95% CI [{lo:+.1f}, {hi:+.1f}]")
        ax2.axvline(0, color=MUTED, lw=1.5, ls="--", label="no difference")
        ax2.axvline(d, color=INK, lw=2, label=f"observed gap {d:+.1f}")
        sig = lo > 0 or hi < 0
        ax2.set_title(f"Bootstrap over {cluster}s: interval {'excludes' if sig else 'includes'} zero", loc="left", fontsize=11)
        ax2.set_xlabel(f"gap in {metric}, A − B (points)\n(histogram: 5,000 resamples of {cluster}s)")
        ax2.set_yticks([])
        ax2.legend(loc="upper left", fontsize=9)
        left = min(s.min(), 0) - 0.5
        ax2.set_xlim(left, max(s.max(), 0) + 0.5)
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
