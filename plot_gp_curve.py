"""
Plot GP/IP degradation curves across CL stages for a given sequence.

Each method is one line; X = stage index (0 = pre-training baseline if available,
1..N = after each CL task); Y = GP mean, IP mean, or individual benchmark scores.

Runs with data at every stage show a full curve.
Runs with only the final stage show a single point (still useful for comparison).

Usage examples
--------------
  # GP curve for one sequence, all methods
  python plot_gp_curve.py --seq NI-Seq-Opposite-v4

  # GP + IP side by side
  python plot_gp_curve.py --seq NI-Seq-Opposite-v4 --metric gp ip

  # Per-benchmark breakdown (one subplot per benchmark)
  python plot_gp_curve.py --seq NI-Seq-Opposite-v4 --metric per_benchmark

  # Filter to specific methods
  python plot_gp_curve.py --seq NI-Seq-Opposite-v4 --method vanilla slice_var_global_cagrad_c050

  # Save instead of showing
  python plot_gp_curve.py --seq NI-Seq-Opposite-v4 --save plots/
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Roots to scan — same as results_analysis.py
# ---------------------------------------------------------------------------

ROOTS = [
    Path("imported_results"),
    Path("imported_results_pending"),
    Path("/mnt/E-SSD/dev-cl-lora/cl-lora/results"),
    Path("/mnt/D-SSD/cl-lora-user/results"),
    Path("/mnt/E-SSD/user/all_results/opposing_seqs_commit_42175dc"),
    Path("/mnt/E-SSD/cl-baselines/cl-lora/results/NI-Seq-G2/basic_methods"),
    Path("/mnt/B-SSD/user/fix-cl-lora/cl-lora/results"),
]

BENCHMARK_SHORT = {
    "hellaswag":          "HellaSwag",
    "commonsenseqa":      "CommonsenseQA",
    "alpaca":             "Alpaca",
    "bbh_object_counting": "BBH-ObjCount",
    "openbookqa":         "OpenBookQA",
    "lambada":            "Lambada",
}

# Directories inside a root that are grouping dirs, not sequence dirs.
# Entries with these names are scanned for run_config-bearing method subdirs,
# but the dir name itself is never used as a fallback sequence name.
_NON_SEQ_DIRS = {"completed", "vanilla_baseline", "ignore_for_now", "base_model"}

# GP and IP exclude BBH zero-shot since it's broken for existing data
GP_EXCLUDE = {"bbh_object_counting"}


def extract_rank(method: str) -> str:
    """Return 'rN' from a trailing _rN suffix, defaulting to 'r64'."""
    m = re.search(r'_r(\d+)$', method)
    return f"r{m.group(1)}" if m else "r64"


def _strip_common_suffix(names: list[str]) -> list[str]:
    if len(names) <= 1:
        return names
    split = [n.split("_") for n in names]
    common = 0
    for parts in zip(*[reversed(s) for s in split]):
        if len(set(parts)) == 1:
            common += 1
        else:
            break
    if common == 0:
        return names
    return ["_".join(s[:-common]) for s in split]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_results_matrix(run_dir: Path) -> list[dict]:
    path = run_dir / "results_matrix.json"
    if not path.exists():
        return []
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _load_stage_records(run_dir: Path) -> list[dict]:
    """Load all stage_record.json files sorted by stage index."""
    stages_dir = run_dir / "stages"
    if not stages_dir.exists():
        return []
    records = []
    for sr_path in sorted(stages_dir.glob("stage_*/stage_record.json")):
        try:
            with sr_path.open() as f:
                records.append(json.load(f))
        except (json.JSONDecodeError, OSError):
            pass
    records.sort(key=lambda r: r.get("stage", 0))
    return records


def _gp_mean_no_bbh(gp: dict) -> float | None:
    vals = [v for k, v in gp.items() if k not in GP_EXCLUDE and v is not None]
    return sum(vals) / len(vals) if vals else None


def _ip_mean_no_bbh(ip: dict) -> float | None:
    vals = [v for k, v in ip.items() if k not in GP_EXCLUDE and v is not None]
    return sum(vals) / len(vals) if vals else None


def collect_runs_for_seq(seq_name: str | None, method_filter: list[str] | None) -> list[dict]:
    """Return list of run dicts with per-stage benchmark data.

    seq_name=None collects all sequences found across ROOTS.
    """
    seen: dict[tuple[str, str], dict] = {}

    def _consider(run_dir: Path, resolved_seq: str) -> None:
        if seq_name is not None and resolved_seq != seq_name:
            return
        method = run_dir.name
        if method_filter and not any(m in method for m in method_filter):
            return
        if method.startswith("incomplete_"):
            return
        records = _load_stage_records(run_dir)
        matrix  = _load_results_matrix(run_dir)
        if not records and not matrix:
            return
        key = (resolved_seq, method)
        if key not in seen or len(records) > len(seen[key]["records"]):
            seen[key] = {"method": method, "seq": resolved_seq, "records": records,
                         "matrix": matrix, "run_dir": run_dir}

    def _infer_seq(d: Path) -> str | None:
        """Read seq from run_config.json / run_summary.json inside dir d."""
        for cfg_name in ("run_config.json", "run_summary.json"):
            cfg_path = d / cfg_name
            if cfg_path.exists():
                try:
                    with cfg_path.open() as f:
                        cfg = json.load(f)
                    return (cfg.get("orchestrator", {}).get("cli_args", {}).get("sequence")
                            or cfg.get("sequence"))
                except Exception:
                    pass
                break
        return None

    for root in ROOTS:
        if not root.exists():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            if (entry / "stages").exists():
                # Flat layout: entry itself is a run dir
                inferred = _infer_seq(entry)
                if inferred:
                    _consider(entry, inferred)
            else:
                # entry is either a seq dir (root/<seq>/<method>) or a grouping dir
                # (root/completed/<method>, root/vanilla_baseline/<method>).
                # Always infer seq from run_config; fall back to entry.name for
                # seq dirs whose method subdirs lack a config file.
                for method_dir in sorted(entry.iterdir()):
                    if not method_dir.is_dir():
                        continue
                    inferred = _infer_seq(method_dir)
                    if inferred is None and entry.name in _NON_SEQ_DIRS:
                        continue  # skip runs without config inside grouping dirs
                    _consider(method_dir, inferred or entry.name)

    return list(seen.values())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _stage_series(run: dict) -> dict[int, dict]:
    """Return {stage_idx: {gp, ip, gp_per_bench, ip_per_bench}} for a run."""
    series = {}
    for rec in run["records"]:
        stage = rec.get("stage", 0)
        gen = rec.get("general", {})
        gp = gen.get("gp") or {}
        ip = gen.get("ip") or {}
        if not gp and not ip:
            continue
        series[stage] = {
            "gp":      _gp_mean_no_bbh(gp),
            "ip":      _ip_mean_no_bbh(ip),
            "gp_per":  {BENCHMARK_SHORT.get(k, k): v for k, v in gp.items()},
            "ip_per":  {BENCHMARK_SHORT.get(k, k): v for k, v in ip.items()},
        }
    return series


def _task_labels(run: dict) -> dict[int, str]:
    """Return {stage_idx: short_task_name}."""
    labels = {}
    for rec in run["records"]:
        t = rec.get("trained_task", "")
        labels[rec.get("stage", 0)] = t.split("_")[-1] if "_" in t else t
    return labels


def plot_gp_ip(runs: list[dict], metrics: list[str], seq_name: str,
               save_dir: Path | None) -> None:
    short_names = _strip_common_suffix([r["method"] for r in runs])
    name_map = dict(zip([r["method"] for r in runs], short_names))

    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5), squeeze=False)
    axes = axes[0]

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(len(runs), 1)) for i in range(len(runs))]

    # Collect all stage indices for x-axis ticks
    all_stages: set[int] = set()
    for r in runs:
        all_stages.update(_stage_series(r).keys())
    all_stages = sorted(all_stages)

    # Build task label from the run with the most records
    task_labels: dict[int, str] = {}
    best = max(runs, key=lambda r: len(r["records"]), default=None)
    if best:
        task_labels = _task_labels(best)

    for ax, metric in zip(axes, metrics):
        for run, color, short in zip(runs, colors, short_names):
            series = _stage_series(run)
            if not series:
                continue
            stages = sorted(series)
            vals = [series[s][metric] for s in stages]

            # full curve vs single final point
            if len(stages) > 1:
                ax.plot(stages, [v * 100 if v else None for v in vals],
                        marker="o", label=short, color=color, linewidth=1.8)
            else:
                ax.scatter(stages, [v * 100 if v else None for v in vals],
                           marker="D", s=60, label=f"{short} (final only)", color=color)

        ax.set_title(f"{'GP' if metric == 'gp' else 'IP'} across stages — {seq_name}")
        ax.set_xlabel("Stage")
        ax.set_ylabel("Score (%)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        if task_labels:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            tick_stages = [s for s in all_stages if s in task_labels]
            ax2.set_xticks(tick_stages)
            ax2.set_xticklabels([task_labels[s] for s in tick_stages],
                                 rotation=30, ha="left", fontsize=7)

        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_dir / seq_name if save_dir else None, "gp_curve.png")


def plot_per_benchmark(runs: list[dict], seq_name: str, save_dir: Path | None) -> None:
    # Collect all benchmark names
    bench_names: list[str] = []
    for r in runs:
        for s in _stage_series(r).values():
            for k in s["gp_per"]:
                if k not in bench_names and k not in {BENCHMARK_SHORT.get("bbh_object_counting")}:
                    bench_names.append(k)
            break

    if not bench_names:
        print("No per-benchmark data found.")
        return

    n = len(bench_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)
    axes = axes[0]
    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(len(runs), 1)) for i in range(len(runs))]
    short_names = _strip_common_suffix([r["method"] for r in runs])

    task_labels: dict[int, str] = {}
    best = max(runs, key=lambda r: len(r["records"]), default=None)
    if best:
        task_labels = _task_labels(best)

    for ax, bench in zip(axes, bench_names):
        for run, color, short in zip(runs, colors, short_names):
            series = _stage_series(run)
            stages = sorted(series)
            vals = [series[s]["gp_per"].get(bench) for s in stages]
            vals_pct = [v * 100 if v is not None else None for v in vals]
            if len(stages) > 1:
                ax.plot(stages, vals_pct, marker="o", label=short, color=color, linewidth=1.8)
            else:
                ax.scatter(stages, vals_pct, marker="D", s=60,
                           label=f"{short} (final only)", color=color)

        ax.set_title(f"{bench} (zero-shot) — {seq_name}")
        ax.set_xlabel("Stage")
        ax.set_ylabel("Score (%)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        if task_labels:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim())
            tick_stages = sorted(task_labels)
            ax2.set_xticks(tick_stages)
            ax2.set_xticklabels([task_labels[s] for s in tick_stages],
                                 rotation=30, ha="left", fontsize=7)
        ax.legend(fontsize=7, loc="lower left")
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Per-benchmark GP degradation — {seq_name}", fontsize=12)
    plt.tight_layout()
    _save_or_show(fig, save_dir / seq_name if save_dir else None, "gp_per_bench.png")


def plot_heatmap(run: dict, save_dir: Path | None) -> None:
    """3-panel paper-style heatmap for one run:
    (I) trained-task scores, (II) GP per benchmark, (III) IP per benchmark.
    All values are shown as % of their respective baselines.
    """
    series = _stage_series(run)
    matrix = run.get("matrix", [])

    # Use the union of stages present in the results matrix AND stage records.
    # stage_record.json may only exist at the final stage, which would collapse
    # the heatmap to 1 row — we want a row for every training stage.
    matrix_stages = sorted({e.get("stage", 0) for e in matrix if e.get("stage", 0) > 0})
    series_stages = sorted(s for s in series if s > 0)
    train_stages  = sorted(set(matrix_stages) | set(series_stages))
    if not train_stages:
        train_stages = sorted(series.keys())  # fallback: include stage 0 if that's all we have
    if not train_stages:
        print(f"  [skip heatmap] {run['method']}: no stage data in matrix or stage records")
        return

    n_rows = len(train_stages)
    row_labels = [f"M{i + 1}" for i in range(n_rows)]

    # ── Panel I: trained-task score matrix ───────────────────────────────
    tasks_ordered: list[str] = []
    for entry in sorted(matrix, key=lambda e: e.get("stage", 0)):
        t = entry.get("trained_task", "")
        if t and t not in tasks_ordered:
            tasks_ordered.append(t)
    n_tasks = len(tasks_ordered)
    task_short = [t.split("_")[-1] if "_" in t else t for t in tasks_ordered]

    raw_I = np.full((n_rows, max(n_tasks, 1)), np.nan)
    for entry in matrix:
        s = entry.get("stage", 0)
        if s not in train_stages:
            continue
        row = train_stages.index(s)
        for task, score in entry.get("scores", {}).items():
            if task in tasks_ordered:
                raw_I[row, tasks_ordered.index(task)] = score

    # Diagonal baseline: score on task t right after it was trained
    diag_base = np.array([
        raw_I[i, i] if i < n_rows and not np.isnan(raw_I[i, i]) else np.nan
        for i in range(n_tasks)
    ])
    norm_I = np.full_like(raw_I, np.nan)
    for row in range(n_rows):
        for col in range(min(n_tasks, row + 1)):
            b = diag_base[col]
            if not np.isnan(b) and b > 0:
                norm_I[row, col] = raw_I[row, col] / b * 100
            elif not np.isnan(b) and b == 0:
                # Baseline is 0 (init/training failure): show as 0 so the cell
                # renders red instead of blank white.
                norm_I[row, col] = 0.0

    # ── Panels II & III: GP / IP per benchmark ───────────────────────────
    BBH_SHORT = BENCHMARK_SHORT.get("bbh_object_counting", "BBH-ObjCount")
    gp_keys: list[str] = []
    ip_keys: list[str] = []
    for s in train_stages:
        if s in series:
            gp_keys = [k for k in series[s]["gp_per"] if k != BBH_SHORT]
            ip_keys = list(series[s]["ip_per"].keys())
            break

    gp_raw = np.full((n_rows, max(len(gp_keys), 1)), np.nan)
    ip_raw = np.full((n_rows, max(len(ip_keys), 1)), np.nan)
    for row, s in enumerate(train_stages):
        if s not in series:
            continue
        for j, k in enumerate(gp_keys):
            v = series[s]["gp_per"].get(k)
            if v is not None:
                gp_raw[row, j] = v
        for j, k in enumerate(ip_keys):
            v = series[s]["ip_per"].get(k)
            if v is not None:
                ip_raw[row, j] = v

    # Baseline for GP/IP: stage 0 if available (pre-training), else first training stage
    base_s = 0 if 0 in series else train_stages[0]
    gp_base = np.array(
        [series[base_s]["gp_per"].get(k) if base_s in series else np.nan for k in gp_keys],
        dtype=float,
    )
    ip_base = np.array(
        [series[base_s]["ip_per"].get(k) if base_s in series else np.nan for k in ip_keys],
        dtype=float,
    )

    def _norm(raw: np.ndarray, base: np.ndarray) -> np.ndarray:
        out = np.full_like(raw, np.nan)
        for col in range(raw.shape[1]):
            b = base[col] if col < len(base) else np.nan
            if not np.isnan(b) and b > 0:
                out[:, col] = raw[:, col] / b * 100
        return out

    norm_II  = _norm(gp_raw[:, :len(gp_keys)], gp_base)
    norm_III = _norm(ip_raw[:, :len(ip_keys)], ip_base)

    # ── Assemble panels ──────────────────────────────────────────────────
    panels = []
    if n_tasks > 0 and not np.all(np.isnan(norm_I[:, :n_tasks])):
        panels.append({
            "title":    "Trained Task Eval",
            "data":     norm_I[:, :n_tasks],
            "cols":     task_short,
            "baseline": [f"{diag_base[j] * 100:.1f}" if not np.isnan(diag_base[j]) else ""
                         for j in range(n_tasks)],
        })
    if gp_keys and not np.all(np.isnan(norm_II)):
        panels.append({
            "title":    "General Task Eval (GP)",
            "data":     norm_II,
            "cols":     gp_keys,
            "baseline": [f"{gp_base[j] * 100:.1f}" if j < len(gp_base) and not np.isnan(gp_base[j]) else ""
                         for j in range(len(gp_keys))],
        })
    if ip_keys and not np.all(np.isnan(norm_III)):
        panels.append({
            "title":    "In-Context Eval (IP)",
            "data":     norm_III,
            "cols":     ip_keys,
            "baseline": [f"{ip_base[j] * 100:.1f}" if j < len(ip_base) and not np.isnan(ip_base[j]) else ""
                         for j in range(len(ip_keys))],
        })

    if not panels:
        print(f"  [skip heatmap] {run['method']}: insufficient data for any panel")
        return

    widths = [max(len(p["cols"]), 1) for p in panels]
    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(max(sum(widths) * 1.6 + 2, 8), max(n_rows * 0.8 + 2.5, 4)),
        gridspec_kw={"width_ratios": widths, "wspace": 0.5},
    )
    if len(panels) == 1:
        axes = [axes]

    cmap  = plt.get_cmap("RdYlGn")   # red = worse than baseline, green = better
    cnorm = plt.Normalize(vmin=60, vmax=120)
    last_im = None

    for ax, panel in zip(axes, panels):
        data, col_labels = panel["data"], panel["cols"]
        im = ax.imshow(data, aspect="auto", cmap=cmap, norm=cnorm)
        last_im = im

        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_title(panel["title"], fontsize=9, pad=18)

        # Baseline values on a top twin axis (shown in dark red)
        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(range(len(col_labels)))
        ax_top.set_xticklabels(panel["baseline"], fontsize=6.5, color="darkred")
        ax_top.tick_params(top=False)

        # Cell annotations
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    txt_color = "white" if (v < 72 or v > 112) else "black"
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=7, color=txt_color)

    plt.colorbar(last_im, ax=axes[-1], fraction=0.046, pad=0.04, label="% of baseline")
    short = _strip_common_suffix([run["method"]])[0]
    fig.suptitle(f"{run['seq']} — {short}", fontsize=10)
    plt.tight_layout()
    method_dir = save_dir / run["seq"] / run["method"] if save_dir else None
    _save_or_show(fig, method_dir, "heatmap.png")


def _save_or_show(fig: plt.Figure, save_dir: Path | None, filename: str) -> None:
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plot GP/IP degradation curves across CL stages.")
    p.add_argument("--seq", default=None,
                   help="Sequence name (e.g. NI-Seq-Opposite-v4). Omit to process all sequences.")
    p.add_argument("--metric", nargs="+", default=["gp", "ip"],
                   choices=["gp", "ip", "per_benchmark"],
                   help="What to plot. 'per_benchmark' shows one subplot per benchmark.")
    p.add_argument("--method", nargs="+", default=None,
                   help="Filter to methods whose name contains any of these strings")
    p.add_argument("--heatmap", action="store_true",
                   help="Plot paper-style 3-panel heatmap (trained tasks + GP/IP benchmarks)")
    p.add_argument("--rank", default=None,
                   help="Filter to a specific rank (e.g. r64, r128). "
                        "If omitted, all ranks are plotted in separate subdirectories.")
    p.add_argument("--save", type=Path, default=None,
                   help="Directory to save plots as .png (default: show interactively)")
    return p.parse_args()


def main():
    args = parse_args()

    seq_label = f"'{args.seq}'" if args.seq else "all sequences"
    print(f"Collecting runs for {seq_label} ...")
    all_runs = collect_runs_for_seq(args.seq, args.method)

    if not all_runs:
        print(f"No runs with stage_record.json data found for {seq_label}.")
        print("Run eval_standalone with benchmarks enabled at every stage first.")
        return

    # Group all runs by rank
    by_rank: dict[str, list[dict]] = defaultdict(list)
    for r in all_runs:
        by_rank[extract_rank(r["method"])].append(r)

    ranks_to_plot = [args.rank] if args.rank else sorted(by_rank.keys())
    if not args.rank:
        print(f"Detected ranks: {ranks_to_plot}")

    for rank in ranks_to_plot:
        runs = by_rank.get(rank, [])
        if not runs:
            print(f"  [skip] no runs for rank={rank}")
            continue

        save_dir = (args.save / rank) if args.save else None

        full_curve = [r for r in runs if len(_stage_series(r)) > 1]
        final_only = [r for r in runs if len(_stage_series(r)) == 1]
        print(f"\nRank={rank}: {len(full_curve)} run(s) with full curves, "
              f"{len(final_only)} run(s) final-only")

        by_seq: dict[str, list[dict]] = defaultdict(list)
        for r in runs:
            by_seq[r["seq"]].append(r)

        for seq, seq_runs in sorted(by_seq.items()):
            if args.heatmap:
                for run in seq_runs:
                    print(f"  heatmap: {seq}/{run['method']}")
                    plot_heatmap(run, save_dir)

            if "per_benchmark" in args.metric:
                plot_per_benchmark(seq_runs, seq, save_dir)
                other = [m for m in args.metric if m != "per_benchmark"]
                if other:
                    plot_gp_ip(seq_runs, other, seq, save_dir)
            else:
                plot_gp_ip(seq_runs, args.metric, seq, save_dir)


if __name__ == "__main__":
    main()
