"""All figures (PNG) and result tables from runs/ -> figures/.   python plot.py [--runs runs] [--out figures]"""
import argparse
import csv
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from agents.q_learning import epsilon  # noqa: E402
from evaluate import above_screen_steps  # noqa: E402
from train import COMMON, tag_of  # noqa: E402

warnings.filterwarnings("ignore", "All-NaN|Mean of empty|Degrees of freedom")  # unvisited cells / empty bins

# reference categorical palette (validated, light surface); text never wears series colour
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, AXIS, SURFACE, NEUTRAL = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb", "#f0efec"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "figure.dpi": 130,
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-", "axes.axisbelow": True,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left",
    "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 2,
})
STEPS = FuncFormatter(lambda x, _: f"{x / 1e6:g}M" if x else "0")
MAIN = [("q_learning", "baseline", "Q-learning"), ("dqn", "baseline", "DQN"), ("dqn", "double", "Double DQN"),
        ("ppo", "baseline", "PPO")]
V_MAX = 0.124 / (1 - 0.99)  # value of never dying: ~0.124 reward/step (0.1 alive, +1 per ~37-step pipe), gamma 0.99
EXT = ".png"  # figure file extension


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] != "" else np.nan for r in rows]) for k in rows[0]}


def seeds(runs, algo, tag):
    return sorted(p.parent for p in (runs / algo / tag).glob("seed*/summary.json"))


def smooth(y, w=3):
    """Trailing moving average (window w, shorter at the start)."""
    c = np.cumsum(np.insert(y, 0, 0.0))
    n = min(w, len(c))
    return np.concatenate([c[1:n] / np.arange(1, n), (c[w:] - c[:-w]) / w])[:len(y)]


def eval_curves(dirs, key="pipes"):
    evs = [read_csv(d / "eval.csv") for d in dirs]
    n = min(len(e["step"]) for e in evs)
    x = evs[0]["step"][:n]
    return x, np.array([np.interp(x, e["step"], e[key]) for e in evs])


def binned(dirs, key, bins=60, how=np.nanmean):
    """Per-seed average of a train.csv column in equal env-step bins -> (bin centres, seeds x bins)."""
    trs = [read_csv(d / "train.csv") for d in dirs]
    edges = np.linspace(0, min(t["step"][-1] for t in trs), bins + 1)
    Y = []
    for t in trs:
        idx = np.digitize(t["step"], edges) - 1
        Y.append([how(t[key][idx == b]) if (idx == b).any() else np.nan for b in range(bins)])
    return (edges[:-1] + edges[1:]) / 2, np.array(Y)


def band(ax, x, Y, color, label, w=5):
    m, s = smooth(np.nanmean(Y, 0), w), smooth(np.nanstd(Y, 0), w)
    ax.plot(x, m, color=color, label=label)
    ax.fill_between(x, m - s, m + s, color=color, alpha=0.14, lw=0)
    return x[-1], m[-1], label


def end_labels(ax, items, gap=0.045):
    """Direct labels at line ends in secondary ink, nudged apart so they never overlap."""
    lo, hi = ax.get_ylim()
    items = sorted(items, key=lambda t: t[1])
    ys = []
    for _, y, _ in items:
        ys.append(max(y, ys[-1] + gap * (hi - lo)) if ys else y)
    for (x, _, text), y in zip(items, ys):
        ax.annotate(text, (x, y), xytext=(4, 0), textcoords="offset points", va="center", fontsize=7.5, color=INK2,
                    annotation_clip=False)


def finish(ax, ylabel="eval pipes (greedy, 10 episodes)"):
    ax.xaxis.set_major_formatter(STEPS)
    ax.set_xlabel("environment steps")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)


def save(fig, out, name):
    path = out / Path(name).with_suffix(EXT)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def learning_curves(runs, out):
    groups = [(a, t, n) for a, t, n in MAIN if seeds(runs, a, t)]
    fig, axes = plt.subplots(1, len(groups), figsize=(3.4 * len(groups), 3.0), squeeze=False)
    for ax, (algo, tag, name) in zip(axes[0], groups):
        d = seeds(runs, algo, tag)
        x, Y = eval_curves(d)
        items = [band(ax, x, Y, SERIES[0], "greedy eval")]
        bx, bY = binned(d, "pipes")
        items.append(band(ax, bx, bY, SERIES[1], "training episodes", w=2))
        ax.set_title(f"{name}  ({len(d)} seeds)")
        finish(ax, "pipes passed")
        end_labels(ax, items)
    axes[0][0].legend(loc="upper left")
    fig.suptitle("Learning curves: mean ± std across seeds, smoothed", x=0.01, ha="left", fontsize=11, weight="bold")
    fig.tight_layout()
    save(fig, out, "learning_curves.png")


def comparison(runs, out):
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    items = []
    for i, (algo, tag, name) in enumerate(MAIN):
        if d := seeds(runs, algo, tag):
            x, Y = eval_curves(d)
            items.append(band(ax, x, Y, SERIES[i], name))
    finish(ax)
    ax.set_title(f"All algorithms, same axes (greedy eval every {COMMON['eval_every'] // 1000}k steps, capped at {COMMON['eval_cap']} pipes)")
    ax.legend(loc="upper left")
    end_labels(ax, items)
    save(fig, out, "comparison.png")


def sweeps(runs, out):
    for algo in ("q_learning", "dqn", "ppo"):
        cfg = json.loads(Path(f"configs/{algo}.json").read_text())
        for param, values in cfg["sweep"].items():
            fig, ax = plt.subplots(figsize=(5.2, 3.2))
            items = []
            base_v = cfg[param]
            ref = "sweep_baseline" if seeds(runs, algo, "sweep_baseline") else "baseline"
            for i, (tag, label) in enumerate([(ref, f"{param}={fmt(base_v)} (baseline)")] +
                                              [(tag_of(param, v), f"{param}={fmt(v)}") for v in values]):
                if d := seeds(runs, algo, tag):
                    x, Y = eval_curves(d)
                    items.append(band(ax, x, Y, SERIES[i], label))
            if len(items) < 2:
                plt.close(fig)
                continue
            finish(ax)
            ax.set_title(f"{algo}: {param} sweep (mean ± std, 3 seeds)")
            ax.legend(loc="upper left")
            end_labels(ax, items)
            save(fig, out, f"sweep_{algo}_{param}.png")


def fmt(v):
    return "x".join(map(str, v)) if isinstance(v, list) else str(v)


def boxplot(runs, out):
    data, names = [], []
    for algo, tag, name in MAIN:
        evs = [json.loads((d / "final_eval.json").read_text()) for d in seeds(runs, algo, tag) if (d / "final_eval.json").exists()]
        if evs:
            data.append((evs, np.concatenate([e["pipes"] for e in evs])))
            names.append(name)
    if not data:
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    cap = data[0][0][0]["cap"]
    bp = ax.boxplot([p for _, p in data], widths=0.5, patch_artist=True, showfliers=True,
                    medianprops=dict(color=INK, lw=1.5), whiskerprops=dict(color=INK2, lw=1), capprops=dict(color=INK2, lw=1),
                    flierprops=dict(marker="o", ms=3, mfc="none", mec=MUTED))
    for i, (patch, (evs, _)) in enumerate(zip(bp["boxes"], data)):
        patch.set(facecolor=SERIES[MAIN.index(next(m for m in MAIN if m[2] == names[i]))], alpha=0.35, edgecolor=INK2, lw=1)
        means = [e["mean"] for e in evs]
        ax.scatter(np.full(len(means), i + 1.33), means, s=22, color=INK, zorder=3, label="per-seed mean" if i == 0 else None)
    ax.axhline(cap, color=MUTED, lw=0.8)
    ax.annotate(f"cap = {cap} pipes", (0.5, cap), xytext=(2, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    ax.set_xticks(range(1, len(names) + 1), names)
    ax.set_ylabel(f"pipes per episode ({data[0][0][0]['episodes']} episodes x seed)")
    ax.set_title("Final evaluation: greedy / deterministic policy, all seeds pooled")
    ax.legend(loc="upper left", bbox_to_anchor=(0.2, 0.93))
    save(fig, out, "final_boxplot.png")


def q_heatmap(runs, out):
    d = seeds(runs, "q_learning", "baseline")
    if not d:
        return
    cfg = json.loads((d[0] / "config.json").read_text())
    q = np.load(d[0] / "q_table.npy")  # (dx, dy, vy, action)
    (x0, x1), (y0, y1), (v0, v1) = cfg["ranges"]
    vy = np.linspace(v0, v1, cfg["bins"][2] + 1)[:-1]
    groups = [("rising (vy ≤ -4)", vy <= -4), ("apex (|vy| < 4)", (vy > -4) & (vy < 4)), ("falling (vy ≥ 4)", vy >= 4)]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6), sharex=True, sharey=True)
    act_cmap = ListedColormap([SERIES[1], SERIES[0]])
    val_cmap = LinearSegmentedColormap.from_list("div", [SERIES[7], NEUTRAL, SERIES[0]])
    val_cmap.set_bad(AXIS)
    for j, (title, m) in enumerate(groups):
        sub = q[:, :, m]                                   # (dx, dy, k, 2)
        seen = (sub != 0).any(-1)                          # visited (dx, dy, vy) cells
        cnt = seen.sum(2)
        qa = np.where(cnt[..., None] > 0, (sub * seen[..., None]).sum(2) / np.maximum(cnt, 1)[..., None], np.nan)
        greedy = np.where(cnt > 0, qa.argmax(-1), np.nan).T  # rows = dy, cols = dx
        vmax = np.nanmax(qa, -1).T
        ext = [x0, x1, y1, y0]                              # dy grows downwards like screen y
        a = axes[0, j]
        a.imshow(np.ma.masked_invalid(greedy), cmap=act_cmap, vmin=0, vmax=1, extent=ext, aspect="auto", interpolation="nearest")
        a.set_facecolor(NEUTRAL)
        a.set_title(f"action, {title}", fontsize=9)
        im = axes[1, j].imshow(vmax, cmap=val_cmap, norm=TwoSlopeNorm(0, min(np.nanmin(vmax), -0.1), max(np.nanmax(vmax), 0.1)),
                               extent=ext, aspect="auto", interpolation="nearest")
        axes[1, j].set_title(f"max Q, {title}", fontsize=9)
        for ax in (a, axes[1, j]):
            ax.grid(False)
            for x in (-52, 34):  # bird overlaps the pipe horizontally between these dx
                ax.axvline(x, color=INK, lw=0.6)
            for y in (-38, 38):  # bird centre must stay within +-38 px of the gap centre
                ax.axhline(y, color=INK, lw=0.6)
    fig.colorbar(im, ax=axes, shrink=0.45, anchor=(0, 0.05), label="max_a Q(s,a)  (red: death expected)")
    fig.legend(handles=[Patch(color=SERIES[0], label="flap"), Patch(color=SERIES[1], label="do nothing"),
                        Patch(color=NEUTRAL, label="never visited")], loc="upper right", ncol=3, bbox_to_anchor=(0.9, 0.97))
    for ax in axes[1]:
        ax.set_xlabel("dx: pipe left edge - bird x (px)")
    for ax in axes[:, 0]:
        ax.set_ylabel("dy: bird - gap centre (px, + = below)")
    fig.suptitle("Q-learning baseline (seed 0): policy and value over (dx, dy); black lines = pipe & gap edges",
                 x=0.01, ha="left", fontsize=11, weight="bold")
    save(fig, out, "q_heatmap.png")


def overestimation(runs, out):
    fig, ax = plt.subplots(figsize=(6, 3.3))
    items = []
    for i, (tag, name) in enumerate([("baseline", "DQN"), ("double", "Double DQN")]):
        if d := seeds(runs, "dqn", tag):
            x, Y = binned(d, "q_mean")
            items.append(band(ax, x, Y, SERIES[1 + i], name, w=2))
    if not items:
        return plt.close(fig)
    ax.axhline(V_MAX, color=MUTED, lw=0.8)
    ax.annotate(f"value of never dying ≈ {V_MAX:.1f}", (0, V_MAX), xytext=(2, 3), textcoords="offset points", fontsize=7.5, color=INK2)
    finish(ax, "mean Q(s,a) on replay batches")
    ax.set_title("Q-value estimates: DQN vs Double DQN")
    ax.legend(loc="lower right")
    end_labels(ax, items)
    save(fig, out, "dqn_q_values.png")


def crashes(runs, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    ia, ib = [], []
    for i, (algo, tag, name) in enumerate(MAIN):
        if not (d := seeds(runs, algo, tag)):
            continue
        x, Y = binned(d, "length", how=lambda L: 1e4 * np.sum(L < COMMON["max_steps"]) / max(L.sum(), 1))
        ia.append(band(a, x, Y, SERIES[i], name, w=2))
        x, Y = eval_curves(d, "crash_rate")
        ib.append(band(b, x, 100 * Y, SERIES[i], name))
    finish(a, "training crashes per 10k steps")
    a.set_title("During training (with exploration)")
    finish(b, f"eval episodes crashing before {COMMON['eval_cap']} pipes (%)")
    b.set_title("Greedy evaluation")
    a.set_yscale("symlog", linthresh=1)
    a.legend(loc="upper right")
    end_labels(b, ib)
    fig.tight_layout()
    save(fig, out, "crash_rates.png")


def exploration(runs, out):
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.2))
    items = []
    for i, (algo, tag, name) in enumerate([("q_learning", "baseline", "Q-learning"), ("q_learning", "eps_schedule=exp", "Q-learning exp"),
                                           ("q_learning", "eps_start=1.0", "Q-learning eps_start=1"), ("dqn", "baseline", "DQN")]):
        if d := seeds(runs, algo, tag):
            cfg = json.loads((d[0] / "config.json").read_text())
            x = np.linspace(0, cfg["total_steps"], 300)
            a.plot(x, [epsilon(cfg, s) for s in x], color=SERIES[i], label=name)
            items.append((x[-1], epsilon(cfg, x[-1]), name))
    finish(a, "ε")
    a.set_yscale("log")
    a.set_ylim(5e-4, 1.5)
    a.set_title("ε-greedy schedules")
    a.legend(loc="lower left")
    items = []
    pc = json.loads(Path("configs/ppo.json").read_text())
    for i, (tag, name) in enumerate([("sweep_baseline", f"ent_coef={pc['ent_coef']} (baseline)")] +
                                    [(tag_of("ent_coef", v), f"ent_coef={v}") for v in pc["sweep"]["ent_coef"]]):
        if d := seeds(runs, "ppo", tag):
            x, Y = binned(d, "entropy")
            items.append(band(b, x, Y, SERIES[i], name, w=2))
    finish(b, "policy entropy (nats, max ln2 = 0.69)")
    b.set_title("PPO policy entropy during training")
    if items:
        b.legend(loc="upper right")
    fig.tight_layout()
    save(fig, out, "exploration.png")


def obs_norm_ablation(runs, out):
    pairs = [(a, n) for a, n in (("dqn", "DQN"), ("ppo", "PPO")) if seeds(runs, a, "obs_norm=false")]
    if not pairs:
        return
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.6 * len(pairs), 3.2), squeeze=False)
    for ax, (algo, name) in zip(axes[0], pairs):
        items = []
        ref = "sweep_baseline" if seeds(runs, algo, "sweep_baseline") else "baseline"  # same 1M budget
        for i, (tag, label) in enumerate([(ref, "standardised obs"), ("obs_norm=false", "raw obs")]):
            x, Y = eval_curves(seeds(runs, algo, tag))
            items.append(band(ax, x, Y, SERIES[i], label))
        finish(ax)
        ax.set_title(f"{name}: input standardisation ablation")
        end_labels(ax, items)
    axes[0][0].legend(loc="upper left")
    fig.tight_layout()
    save(fig, out, "obs_norm_ablation.png")


def sb3_check(runs, out):
    if not seeds(runs, "sb3_ppo", "baseline"):
        return
    fig, ax = plt.subplots(figsize=(6, 3.3))
    items = [band(ax, *eval_curves(seeds(runs, algo, tag)), SERIES[i], name)
             for i, (algo, tag, name) in enumerate([("ppo", "sweep_baseline", "our PPO"), ("sb3_ppo", "baseline", "SB3 PPO")])
             if seeds(runs, algo, tag)]
    finish(ax)
    ax.set_title("PPO from scratch vs Stable-Baselines3 (same hyperparameters, 1M steps)")
    ax.legend(loc="upper left")
    end_labels(ax, items)
    save(fig, out, "ppo_vs_sb3.png")


def first_step(x, y, thr):
    hit = np.flatnonzero(y >= thr)
    return x[hit[0]] if len(hit) else None


def tables(runs, out):
    """results.md: final eval, steps-to-threshold, wall time, for every algo/tag with finished runs."""
    lines = ["| algo | config | seeds | final eval mean ± std | median | max | crash % | steps to 10 pipes | steps to 50 pipes | "
             "last eval | best eval | above-screen steps /1k (train last 20% / final eval) | wall time / run |", "|" + "---|" * 13]
    for algo_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        for tag_dir in sorted(p for p in algo_dir.iterdir() if p.is_dir()):
            d = seeds(runs, algo_dir.name, tag_dir.name)
            if not d:
                continue
            x, Y = eval_curves(d)
            reach = {t: [first_step(x, y, t) for y in Y] for t in (10, 50)}
            fe = [json.loads((s / "final_eval.json").read_text()) for s in d if (s / "final_eval.json").exists()]
            wall = np.mean([json.loads((s / "summary.json").read_text())["wall_time"] for s in d]) / 60
            above = []
            for s in d:
                t = read_csv(s / "train.csv")
                late = t["step"] >= 0.8 * t["step"][-1]
                k = above_screen_steps(t["ret"][late], t["pipes"][late], t["length"][late], t["length"][late] < COMMON["max_steps"])
                above.append(1000 * k.sum() / t["length"][late].sum())
            ev_above = f"{np.mean([e['above_screen_per_1k'] for e in fe]):.2f}" if fe and "above_screen_per_1k" in fe[0] else "–"
            if fe:
                pooled = np.concatenate([e["pipes"] for e in fe])
                fin = (f"{pooled.mean():.1f} ± {np.std([e['mean'] for e in fe]):.1f} | {np.median(pooled):.0f} | {pooled.max():.0f} | "
                       f"{100 * np.mean([e['crash_rate'] for e in fe]):.0f}")
            else:
                fin = "– | – | – | –"

            def r(t):
                hits = [h for h in reach[t] if h is not None]
                return f"{np.median(hits) / 1e3:.0f}k ({len(hits)}/{len(d)})" if hits else f"never (0/{len(d)})"
            lines.append(f"| {algo_dir.name} | {tag_dir.name} | {len(d)} | {fin} | {r(10)} | {r(50)} | "
                         f"{Y[:, -1].mean():.1f} ± {Y[:, -1].std():.1f} | {Y.max(1).mean():.1f} | {np.mean(above):.2f} / {ev_above} | {wall:.1f} min |")
    (out / "results.md").write_text("Final eval mean ± std: mean over all episodes, std across seed means. "
                                    "Steps to N pipes: median over the seeds that got there (first greedy eval with mean ≥ N).\n\n"
                                    + "\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", out / "results.md")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, default=Path("runs"))
    p.add_argument("--out", type=Path, default=Path("figures"))
    args = p.parse_args()
    args.out.mkdir(exist_ok=True)
    for f in (learning_curves, comparison, sweeps, boxplot, q_heatmap, overestimation, crashes, exploration,
              obs_norm_ablation, sb3_check, tables):
        f(args.runs, args.out)
