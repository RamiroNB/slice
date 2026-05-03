"""
Generate all CL result plots.

Plots produced:
  1. cl_scatter_selected.{png,pdf}     — selected variants only, all sequences as cloud
  2. cl_scatter_all_variants.{png,pdf} — every slice variant, all sequences (apdx)
  3. cl_scatter_<seq>.{png,pdf}        — single sequence highlight
  4. cl_relative_gain.{png,pdf}        — slice mean - baseline mean per sequence
  5. cl_skill_score.{png,pdf}          — skill score normalization
  6. cl_radar.{png,pdf}                — per-sequence AP radar
  7. cl_mean_rank.{png,pdf}            — mean rank by AP and FP
  8. cl_winloss.{png,pdf}              — slice vs baseline win/loss heatmap

Run:    python plot.py [--csv results.csv]
Output: ./figs/*.png and ./figs/*.pdf
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Ellipse, Patch

# ============================================================
# CONFIGURATION
# ============================================================

# Default CSV path; override with --csv at the command line.
CSV_PATH = "results.csv"

# Rename seq_name values from the CSV for short display labels.
# Any seq not listed here is used as-is.
SEQ_RENAME = {
    "NI-Seq-G1":          "G1",
    "NI-Seq-G2":          "G2",
    "NI-Seq-Opposite-v2": "Opp-v2",
    "NI-Seq-Opposite-v3": "Opp-v3",
    "NI-Seq-Opposite-v4": "Opp-v4",
}

# Preferred sequence display order (sequences not listed are appended alphabetically).
SEQ_ORDER = ["G1", "G2", "Opp-v2", "Opp-v3", "Opp-v4", "TRACE"]

# Sequence to use for the single-sequence scatter plot (plot 3).
HIGHLIGHT_SEQ = "G2"

# Selected SLICE variants: key must be a substring of the method name in the CSV.
# Longer keys are tried before shorter ones to avoid ambiguous matches.
SELECTED_SLICE_CFG = {
    "magpreserve":   {"label": "MagPres (global)", "marker": "o"},
    "cagrad_c050":   {"label": "CAGrad c=0.50",    "marker": "s"},
    "cagrad_c075":   {"label": "CAGrad c=0.75",    "marker": "^"},
    "slice_lora_ga": {"label": "LoRA-GA (slice)",  "marker": "D"},
    "slice_top_r":   {"label": "Top-r (slice)",    "marker": "P"},
}

# Baseline methods (same matching rule as above).
BASELINE_CFG = {
    "vanilla":      {"label": "Vanilla LoRA",  "marker": "o"},
    "loram":        {"label": "LoRAM",         "marker": "s"},
    "lora_ga_topr": {"label": "LoRA-GA Top-r", "marker": "P"},
    "lora_ga":      {"label": "LoRA-GA",       "marker": "D"},
}

# Other slice variants shown in the appendix scatter (no individual labels).
OTHER_SLICE_PATTERNS = [
    "magpreserve_local", "global_costau_neg005", "global_cagrad_c025",
    "global_costau_005", "global_costau_000", "global_costau_010",
    "svd_topr_no_sigma", "combo_cagrad_mag_topr", "global_combo_cagrad_mag_topr",
    "gradvac_phi0_b05", "cagrad_c025_local", "cagrad_c050_local",
    "cagrad_c075_local", "costau_neg005_local", "nullspace_r8", "nullspace_r32",
]

# Weak variants excluded from aggregate comparisons but still shown in scatter.
WEAK_METHODS = {"slice_top_r"}

# Preferred per-sequence colours (any sequence not listed gets an auto colour).
SEQ_COLORS_PREFERRED = {
    "G1": "#4c72b0", "G2": "#dd8452", "Opp-v2": "#55a868",
    "Opp-v3": "#c44e52", "Opp-v4": "#8172b3", "TRACE": "#937860",
}

SLICE_COLOR = "#d83838"
SLICE_LIGHT = "#f0a8a8"
BASE_COLOR  = "#666666"

OUT = "./figs"
os.makedirs(OUT, exist_ok=True)

# ============================================================
# Runtime data (populated by load_data())
# ============================================================
selected_slice:  dict = {}
other_slice:     dict = {}
baselines:       dict = {}
labels_selected: dict = {}
labels_baseline: dict = {}
markers_selected: dict = {}
markers_baseline: dict = {}
SEQUENCES: list = []


# ============================================================
# Data loading
# ============================================================

def _match_key(method: str, keys: list[str]) -> str | None:
    """Exact match first; then try substring matches, longest key first."""
    if method in keys:
        return method
    for k in sorted(keys, key=len, reverse=True):
        if k in method:
            return k
    return None


def load_data(csv_path: str) -> None:
    global selected_slice, other_slice, baselines
    global labels_selected, labels_baseline, markers_selected, markers_baseline
    global SEQUENCES

    df = pd.read_csv(csv_path)
    required = {"seq_name", "method", "AP", "FP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}. Run results_analysis.py --export-csv first.")

    df = df.copy()
    df["seq"] = df["seq_name"].map(lambda s: SEQ_RENAME.get(s, s))
    df = df.dropna(subset=["AP", "FP"])

    def _build(cfg: dict) -> dict:
        result = {k: {} for k in cfg}
        keys = list(cfg.keys())
        for row in df.itertuples(index=False):
            key = _match_key(row.method, keys)
            if key is not None:
                result[key][row.seq] = (float(row.AP), float(row.FP))
        return {k: v for k, v in result.items() if v}

    selected_slice = _build(SELECTED_SLICE_CFG)
    baselines      = _build(BASELINE_CFG)

    other_slice = {}
    for row in df.itertuples(index=False):
        key = _match_key(row.method, OTHER_SLICE_PATTERNS)
        if key:
            other_slice.setdefault(key, {})[row.seq] = (float(row.AP), float(row.FP))

    labels_selected  = {k: SELECTED_SLICE_CFG[k]["label"]  for k in selected_slice}
    labels_baseline  = {k: BASELINE_CFG[k]["label"]         for k in baselines}
    markers_selected = {k: SELECTED_SLICE_CFG[k]["marker"]  for k in selected_slice}
    markers_baseline = {k: BASELINE_CFG[k]["marker"]         for k in baselines}

    all_seqs: set[str] = set()
    for d in (selected_slice, baselines, other_slice):
        for pts in d.values():
            all_seqs.update(pts.keys())

    SEQUENCES = [s for s in SEQ_ORDER if s in all_seqs] + \
                sorted(all_seqs - set(SEQ_ORDER))

    print(f"Loaded {len(df)} rows from {csv_path}")
    print(f"  sequences:      {SEQUENCES}")
    print(f"  selected slice: {list(selected_slice.keys())}")
    print(f"  baselines:      {list(baselines.keys())}")
    print(f"  other slice:    {list(other_slice.keys())}")


def _seq_colors() -> dict[str, str]:
    """Return a colour for every sequence, using preferred colours where defined."""
    cmap = plt.get_cmap("tab10")
    result = {}
    auto_idx = 0
    for s in SEQUENCES:
        if s in SEQ_COLORS_PREFERRED:
            result[s] = SEQ_COLORS_PREFERRED[s]
        else:
            result[s] = cmap(auto_idx % 10)
            auto_idx += 1
    return result


# ============================================================
# Helpers
# ============================================================
def points_array(method_dict):
    return np.array(list(method_dict.values()))


def add_diagonal_and_regions(ax, lim=(0.0, 0.45)):
    poly = Polygon([[lim[0], lim[0]], [lim[1], lim[1]], [lim[0], lim[1]]],
                   closed=True, color="green", alpha=0.06, zorder=0)
    ax.add_patch(poly)
    ax.plot(lim, lim, color="gray", linestyle="--",
            linewidth=0.9, alpha=0.6, zorder=1)


# ============================================================
# Plot 1: scatter selected variants, all sequences (cloud)
# ============================================================
def plot_scatter_selected():
    fig, ax = plt.subplots(figsize=(8, 7))
    add_diagonal_and_regions(ax)
    ax.text(0.42, 0.435, "FP = AP", fontsize=8, color="gray",
            rotation=42, va="bottom")
    ax.text(0.05, 0.40, "backward transfer", fontsize=8,
            color="#3a7a3a", style="italic", alpha=0.85)
    ax.text(0.30, 0.025, "forgetting", fontsize=8,
            color="#a03030", style="italic", alpha=0.85)

    for name, pts in baselines.items():
        arr = points_array(pts)
        m = markers_baseline[name]
        ax.scatter(arr[:, 0], arr[:, 1], color=BASE_COLOR, marker=m,
                   s=32, alpha=0.45, linewidths=0, zorder=3)
        mean = arr.mean(axis=0)
        ax.scatter(*mean, color=BASE_COLOR, marker=m, s=130,
                   edgecolors="white", linewidths=1.2, zorder=6,
                   label=labels_baseline[name])

    for name, pts in selected_slice.items():
        arr = points_array(pts)
        m = markers_selected[name]
        ax.scatter(arr[:, 0], arr[:, 1], color=SLICE_COLOR, marker=m,
                   s=42, alpha=0.65, linewidths=0, zorder=4)
        mean = arr.mean(axis=0)
        ax.scatter(*mean, color=SLICE_COLOR, marker=m, s=140,
                   edgecolors="white", linewidths=1.3, zorder=7,
                   label=labels_selected[name])

    ax.set_xlabel("AP (plasticity)", fontsize=11)
    ax.set_ylabel("FP (stability)", fontsize=11)
    ax.set_title("Stability vs Plasticity — selected variants, all sequences",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0.05, 0.45); ax.set_ylim(-0.01, 0.45)
    ax.grid(True, color="gray", linewidth=0.25, alpha=0.4)
    ax.set_aspect("equal")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92,
              edgecolor="lightgray",
              title="(large = mean across sequences)", title_fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/cl_scatter_selected.pdf", bbox_inches="tight")
    plt.savefig(f"{OUT}/cl_scatter_selected.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Plot 2: scatter all variants (incl. appendix-only ones)
# ============================================================
def plot_scatter_all_variants():
    fig, ax = plt.subplots(figsize=(8, 7))
    add_diagonal_and_regions(ax)
    ax.text(0.42, 0.435, "FP = AP", fontsize=8, color="gray",
            rotation=42, va="bottom")
    ax.text(0.05, 0.40, "backward transfer", fontsize=8,
            color="#3a7a3a", style="italic", alpha=0.85)
    ax.text(0.30, 0.025, "forgetting", fontsize=8,
            color="#a03030", style="italic", alpha=0.85)

    for name, pts in other_slice.items():
        arr = points_array(pts)
        ax.scatter(arr[:, 0], arr[:, 1], color=SLICE_LIGHT, marker="o",
                   s=22, alpha=0.55, linewidths=0, zorder=2)
        mean = arr.mean(axis=0)
        ax.scatter(*mean, facecolors="none", edgecolors=SLICE_LIGHT,
                   marker="o", s=55, linewidths=1.0, alpha=0.8, zorder=3)

    for name, pts in baselines.items():
        arr = points_array(pts)
        m = markers_baseline[name]
        ax.scatter(arr[:, 0], arr[:, 1], color=BASE_COLOR, marker=m,
                   s=32, alpha=0.45, linewidths=0, zorder=3)
        mean = arr.mean(axis=0)
        ax.scatter(*mean, color=BASE_COLOR, marker=m, s=130,
                   edgecolors="white", linewidths=1.2, zorder=6)

    for name, pts in selected_slice.items():
        arr = points_array(pts)
        m = markers_selected[name]
        ax.scatter(arr[:, 0], arr[:, 1], color=SLICE_COLOR, marker=m,
                   s=42, alpha=0.65, linewidths=0, zorder=4)
        mean = arr.mean(axis=0)
        ax.scatter(*mean, color=SLICE_COLOR, marker=m, s=140,
                   edgecolors="white", linewidths=1.3, zorder=7)

    ax.set_xlabel("AP (plasticity)", fontsize=11)
    ax.set_ylabel("FP (stability)", fontsize=11)
    ax.set_title("Stability vs Plasticity — all slice variants, all sequences",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0.05, 0.45); ax.set_ylim(-0.01, 0.45)
    ax.grid(True, color="gray", linewidth=0.25, alpha=0.4)
    ax.set_aspect("equal")

    sel_handles = [Line2D([0], [0], marker=markers_selected[k], color="w",
                          markerfacecolor=SLICE_COLOR, markersize=9,
                          label=labels_selected[k]) for k in labels_selected]
    base_handles = [Line2D([0], [0], marker=markers_baseline[k], color="w",
                           markerfacecolor=BASE_COLOR, markersize=9,
                           label=labels_baseline[k]) for k in labels_baseline]
    other_handle = Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=SLICE_LIGHT, markersize=7,
                          alpha=0.7, label="other slice variants")
    ax.legend(handles=sel_handles + base_handles + [other_handle],
              fontsize=8, loc="upper left", framealpha=0.92,
              edgecolor="lightgray", title_fontsize=7.5,
              title="(large = mean across sequences)")
    plt.tight_layout()
    plt.savefig(f"{OUT}/cl_scatter_all_variants.pdf", bbox_inches="tight")
    plt.savefig(f"{OUT}/cl_scatter_all_variants.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Plot 3: single sequence highlight
# ============================================================
def plot_highlight_seq():
    seq = HIGHLIGHT_SEQ
    hl_selected = {k: v[seq] for k, v in selected_slice.items() if seq in v}
    hl_base     = {k: v[seq] for k, v in baselines.items()      if seq in v}

    if not hl_selected and not hl_base:
        print(f"  [skip] highlight seq '{seq}' not found in loaded data")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    add_diagonal_and_regions(ax)
    ax.text(0.42, 0.435, "FP = AP", fontsize=9, color="gray",
            rotation=42, va="bottom")
    ax.text(0.04, 0.40, "backward transfer", fontsize=9,
            color="#3a7a3a", style="italic", alpha=0.85)
    ax.text(0.30, 0.025, "forgetting", fontsize=9,
            color="#a03030", style="italic", alpha=0.85)

    for name, (ap, fp) in hl_base.items():
        ax.scatter(ap, fp, color=BASE_COLOR, marker=markers_baseline[name],
                   s=180, edgecolors="white", linewidths=1.5, zorder=5)
        ax.annotate(labels_baseline[name], (ap, fp), xytext=(8, -2),
                    textcoords="offset points", fontsize=10, color=BASE_COLOR,
                    va="center")

    for name, (ap, fp) in hl_selected.items():
        ax.scatter(ap, fp, color=SLICE_COLOR, marker=markers_selected[name],
                   s=220, edgecolors="white", linewidths=1.5, zorder=6)
        ax.annotate(labels_selected[name], (ap, fp), xytext=(10, 0),
                    textcoords="offset points", fontsize=10, color=SLICE_COLOR,
                    fontweight="bold", va="center")

    if hl_selected:
        sel_pts = np.array(list(hl_selected.values()))
        cx, cy = sel_pts.mean(axis=0)
        ell = Ellipse((cx, cy), width=0.10, height=0.14, angle=15,
                      fill=False, edgecolor=SLICE_COLOR, linewidth=1.0,
                      linestyle=":", alpha=0.5, zorder=2)
        ax.add_patch(ell)

    ax.set_xlabel("AP (plasticity) — higher is better", fontsize=11)
    ax.set_ylabel("FP (stability) — higher is better", fontsize=11)
    ax.set_title(f"Stability vs Plasticity — {seq}", fontsize=13, fontweight="bold")
    ax.set_xlim(0.10, 0.45); ax.set_ylim(0.00, 0.40)
    ax.grid(True, color="gray", linewidth=0.25, alpha=0.4)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SLICE_COLOR,
               markeredgecolor="white", markersize=11, label="SLICE (selected)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=BASE_COLOR,
               markeredgecolor="white", markersize=11, label="baselines"),
    ]
    ax.legend(handles=handles, fontsize=10, loc="lower right",
              framealpha=0.92, edgecolor="lightgray")
    plt.tight_layout()
    safe = seq.replace("/", "-")
    plt.savefig(f"{OUT}/cl_scatter_{safe}.pdf", bbox_inches="tight")
    plt.savefig(f"{OUT}/cl_scatter_{safe}.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Plot 4: relative gain — slice mean vs baseline mean per sequence
# ============================================================
def plot_relative_gain():
    seq_colors = _seq_colors()
    weak = WEAK_METHODS

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    for seq in SEQUENCES:
        sl = [v[seq] for k, v in selected_slice.items()
              if k not in weak and seq in v]
        bs = [v[seq] for v in baselines.values() if seq in v]
        if not sl or not bs:
            continue
        delta = np.array(sl).mean(axis=0) - np.array(bs).mean(axis=0)
        c = seq_colors[seq]
        ax.scatter(*delta, color=c, s=180, edgecolors="white",
                   linewidths=1.5, zorder=5)
        ax.annotate(seq, delta, xytext=(10, 6),
                    textcoords="offset points", fontsize=10,
                    color=c, fontweight="bold")

    ax.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
    ax.axvline(0, color="gray", linewidth=0.8, alpha=0.6)
    xmax, ymax = 0.20, 0.22
    xmin, ymin = -0.06, -0.10
    ax.fill_between([0, xmax], 0, ymax, color="green", alpha=0.07, zorder=0)
    ax.fill_between([xmin, 0], ymin, 0, color="red", alpha=0.07, zorder=0)
    ax.text(xmax * 0.95, ymax * 0.95, "SLICE wins\non both", fontsize=9,
            color="#3a7a3a", ha="right", va="top", style="italic", alpha=0.85)
    ax.text(xmin * 0.95, ymin * 0.95, "baselines win\non both", fontsize=9,
            color="#a03030", ha="left", va="bottom", style="italic", alpha=0.85)
    ax.set_xlabel(r"$\Delta$AP (SLICE mean $-$ baseline mean)", fontsize=11)
    ax.set_ylabel(r"$\Delta$FP (SLICE mean $-$ baseline mean)", fontsize=11)
    ax.set_title("Relative gain per sequence", fontsize=11, fontweight="bold")
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.grid(True, color="gray", linewidth=0.25, alpha=0.4)

    ax2 = axes[1]
    for seq in SEQUENCES:
        bs = [v[seq] for v in baselines.values() if seq in v]
        if not bs:
            continue
        base_mean = np.array(bs).mean(axis=0)
        for m_name, m_data in selected_slice.items():
            if m_name in weak or seq not in m_data:
                continue
            delta = np.array(m_data[seq]) - base_mean
            ax2.scatter(*delta, color=seq_colors[seq],
                        marker=markers_selected[m_name], s=80,
                        alpha=0.75, edgecolors="white", linewidths=0.6,
                        zorder=4)

    ax2.axhline(0, color="gray", linewidth=0.8, alpha=0.6)
    ax2.axvline(0, color="gray", linewidth=0.8, alpha=0.6)
    xmax, ymax = 0.22, 0.27
    xmin, ymin = -0.16, -0.18
    ax2.fill_between([0, xmax], 0, ymax, color="green", alpha=0.07, zorder=0)
    ax2.fill_between([xmin, 0], ymin, 0, color="red", alpha=0.07, zorder=0)
    ax2.set_xlabel(r"$\Delta$AP (variant $-$ baseline mean)", fontsize=11)
    ax2.set_ylabel(r"$\Delta$FP (variant $-$ baseline mean)", fontsize=11)
    ax2.set_title("Per-variant gain (color=seq, marker=method)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlim(xmin, xmax); ax2.set_ylim(ymin, ymax)
    ax2.grid(True, color="gray", linewidth=0.25, alpha=0.4)

    m_handles = [Line2D([0], [0], marker=markers_selected[k], color="w",
                        markerfacecolor="gray", markersize=9,
                        label=labels_selected[k])
                 for k in selected_slice if k not in weak]
    s_handles = [Line2D([0], [0], marker="o", color="w",
                        markerfacecolor=seq_colors[s], markersize=9,
                        label=s) for s in SEQUENCES]
    leg1 = ax2.legend(handles=m_handles, loc="upper left", fontsize=8,
                      title="method", title_fontsize=8,
                      framealpha=0.9, edgecolor="lightgray")
    ax2.add_artist(leg1)
    ax2.legend(handles=s_handles, loc="lower right", fontsize=8,
               title="sequence", title_fontsize=8,
               framealpha=0.9, edgecolor="lightgray")
    plt.tight_layout()
    plt.savefig(f"{OUT}/cl_relative_gain.pdf", bbox_inches="tight")
    plt.savefig(f"{OUT}/cl_relative_gain.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Plot 5: skill score
# ============================================================
def plot_skill_score():
    all_methods = {**selected_slice, **baselines}

    def get_reference(seq):
        if seq in baselines.get("vanilla", {}):
            return np.array(baselines["vanilla"][seq])
        fallbacks = [baselines[m][seq] for m in baselines if seq in baselines[m]]
        return np.array(fallbacks).mean(axis=0) if fallbacks else None

    def best_per_seq(seq):
        vals = [v[seq] for v in all_methods.values() if seq in v]
        return np.array(vals).max(axis=0)

    skill = {m: [] for m in all_methods}
    for seq in SEQUENCES:
        ref = get_reference(seq)
        if ref is None:
            continue
        best = best_per_seq(seq)
        denom = best - ref
        for m_name, m_data in all_methods.items():
            if seq not in m_data:
                continue
            pt = np.array(m_data[seq])
            ss = np.where(denom > 1e-9, (pt - ref) / np.where(denom > 1e-9, denom, 1), np.nan)
            skill[m_name].append((seq, ss[0], ss[1]))

    style_map = {}
    for k in selected_slice:
        style_map[k] = (SLICE_COLOR, markers_selected[k], labels_selected[k])
    for k in baselines:
        style_map[k] = (BASE_COLOR, markers_baseline[k], labels_baseline[k])

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axhline(0, color="gray", linewidth=0.7, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.7, alpha=0.5)
    ax.axhline(1, color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.axvline(1, color="gray", linewidth=0.5, linestyle=":", alpha=0.4)
    ax.scatter([0], [0], marker="x", s=120, color="black", linewidths=2, zorder=10)
    ax.annotate("reference\n(vanilla)", (0, 0), xytext=(8, -16),
                textcoords="offset points", fontsize=8.5, color="black",
                ha="left", style="italic")
    ax.scatter([1], [1], marker="*", s=200, color="goldenrod",
               edgecolors="black", linewidths=0.8, zorder=10)
    ax.annotate("ideal\n(best on both)", (1, 1), xytext=(-8, -8),
                textcoords="offset points", fontsize=8.5, color="black",
                ha="right", va="top", style="italic")

    for m_name, pts in skill.items():
        arr = np.array([(a, f) for _, a, f in pts])
        arr = arr[~np.isnan(arr).any(axis=1)] if arr.size else arr
        if len(arr) == 0:
            continue
        color, marker, label = style_map[m_name]
        ax.scatter(arr[:, 0], arr[:, 1], color=color, marker=marker,
                   s=40, alpha=0.45, linewidths=0, zorder=3)
        mean = arr.mean(axis=0)
        ax.scatter(*mean, color=color, marker=marker, s=160,
                   edgecolors="white", linewidths=1.4, zorder=6, label=label)

    ax.set_xlabel("Skill score — AP", fontsize=11)
    ax.set_ylabel("Skill score — FP", fontsize=11)
    ax.set_title("Skill score across sequences\n(0 = vanilla, 1 = best per sequence)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(-1.5, 1.15); ax.set_ylim(-1.5, 1.15)
    ax.grid(True, color="gray", linewidth=0.25, alpha=0.4)
    ax.set_aspect("equal")
    ax.legend(fontsize=8.5, loc="lower left", framealpha=0.92,
              edgecolor="lightgray",
              title="(large = mean across sequences)", title_fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUT}/cl_skill_score.pdf", bbox_inches="tight")
    plt.savefig(f"{OUT}/cl_skill_score.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Plot 6: radar (AP normalized per sequence)
# ============================================================
def plot_radar():
    # Build method data from loaded dicts
    slice_keys = [k for k in selected_slice if k not in WEAK_METHODS]
    base_keys  = list(baselines.keys())

    def _ap_series(data_dict, key):
        d = data_dict.get(key, {})
        return {s: (d[s][0] if s in d else np.nan) for s in SEQUENCES}

    radar_methods = {}
    for k in slice_keys:
        radar_methods[labels_selected[k]] = _ap_series(selected_slice, k)
    for k in base_keys:
        radar_methods[labels_baseline[k]] = _ap_series(baselines, k)

    seq_max = {}
    for s in SEQUENCES:
        vals = [v[s] for v in radar_methods.values()
                if s in v and not np.isnan(v[s])]
        seq_max[s] = max(vals) if vals else np.nan

    N = len(SEQUENCES)
    if N < 3:
        print("  [skip radar] need at least 3 sequences")
        return
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]

    slice_labels = [labels_selected[k] for k in slice_keys]
    base_labels  = [labels_baseline[k]  for k in base_keys]

    n_sl = len(slice_labels)
    n_bl = len(base_labels)
    slice_colors = plt.get_cmap("Reds")(np.linspace(0.45, 0.9, max(n_sl, 1))).tolist()
    base_colors  = plt.get_cmap("Greys")(np.linspace(0.3, 0.65, max(n_bl, 1))).tolist()

    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             subplot_kw={"projection": "polar"})

    for ax, title, hl_labels, hl_colors, dim_labels in [
        (axes[0], "SLICE methods (vs faded baselines)",
         slice_labels, slice_colors, base_labels),
        (axes[1], "Baselines (vs faded SLICE)",
         base_labels, base_colors, slice_labels),
    ]:
        for m in dim_labels:
            vals = [radar_methods[m].get(s, np.nan) /
                    (seq_max[s] if seq_max.get(s) else 1) for s in SEQUENCES]
            ax.plot(angles, vals + vals[:1], color="lightgray",
                    linewidth=1.0, alpha=0.6)
            ax.fill(angles, vals + vals[:1], color="lightgray", alpha=0.05)

        for m, c in zip(hl_labels, hl_colors):
            vals = [radar_methods[m].get(s, np.nan) /
                    (seq_max[s] if seq_max.get(s) else 1) for s in SEQUENCES]
            ax.plot(angles, vals + vals[:1], color=c, linewidth=2.0, label=m)
            ax.fill(angles, vals + vals[:1], color=c, alpha=0.10)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(SEQUENCES, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.0"],
                           fontsize=8, color="gray")
        ax.grid(color="gray", alpha=0.3, linewidth=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold", pad=20)
        ax.legend(loc="lower right", bbox_to_anchor=(1.25, -0.05),
                  fontsize=8.5, framealpha=0.9, edgecolor="lightgray")

    fig.suptitle("Radar: AP normalized per sequence (1.0 = best on that sequence)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT}/cl_radar.pdf", bbox_inches="tight")
    plt.savefig(f"{OUT}/cl_radar.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Plot 7: mean rank (AP and FP)
# ============================================================
def plot_mean_rank():
    # Build from loaded data: display label → (category, key)
    method_to_data = {}
    for k in selected_slice:
        method_to_data[labels_selected[k]] = ("selected", k)
    for k in baselines:
        method_to_data[labels_baseline[k]] = ("base", k)

    methods = list(method_to_data.keys())
    if not methods:
        return

    def get_pt(label, seq, axis):
        kind, key = method_to_data[label]
        d = selected_slice[key] if kind == "selected" else baselines[key]
        return d[seq][axis] if seq in d else None

    def compute_ranks(axis):
        ranks = {m: {} for m in methods}
        for s in SEQUENCES:
            pairs = [(m, get_pt(m, s, axis)) for m in methods]
            pairs = [(m, v) for m, v in pairs if v is not None]
            pairs.sort(key=lambda x: x[1], reverse=True)
            for r, (m, _) in enumerate(pairs, 1):
                ranks[m][s] = r
        return ranks

    ranks_ap = compute_ranks(0)
    ranks_fp = compute_ranks(1)

    def stats(d):
        return {m: (np.mean(list(v.values())) if v else np.nan, len(v))
                for m, v in d.items()}

    mr_ap = stats(ranks_ap)
    mr_fp = stats(ranks_fp)
    sorted_methods = sorted(methods,
                            key=lambda m: (mr_ap[m][0] + mr_fp[m][0]) / 2)

    slice_labels = set(labels_selected.values())
    bar_h = 0.35
    fig, ax = plt.subplots(figsize=(9, max(4, len(methods) * 0.7 + 1)))
    y_positions = np.arange(len(sorted_methods))

    for i, m in enumerate(sorted_methods):
        color = SLICE_COLOR if m in slice_labels else BASE_COLOR
        ap_mean, ap_n = mr_ap[m]
        fp_mean, fp_n = mr_fp[m]
        ax.barh(i + bar_h / 2, ap_mean, height=bar_h,
                color=color, alpha=0.55, edgecolor="white", linewidth=0.6)
        ax.text(ap_mean + 0.05, i + bar_h / 2,
                f"{ap_mean:.2f} (n={ap_n})", va="center",
                fontsize=8, color=color)
        ax.barh(i - bar_h / 2, fp_mean, height=bar_h,
                color=color, alpha=0.95, edgecolor="white", linewidth=0.6)
        ax.text(fp_mean + 0.05, i - bar_h / 2,
                f"{fp_mean:.2f} (n={fp_n})", va="center",
                fontsize=8, color=color)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(sorted_methods, fontsize=10)
    ax.set_xlabel("Mean rank across sequences (1 = best, lower is better)",
                  fontsize=11)
    ax.set_title("Mean rank by AP and FP\n(top bar = AP, bottom bar = FP per method)",
                 fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    max_rank = max((mr_ap[m][0] for m in methods if not np.isnan(mr_ap[m][0])),
                   default=len(methods))
    ax.set_xlim(0, max_rank + 1.5)
    ax.grid(True, axis="x", color="gray", linewidth=0.25, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(handles=[
        Patch(facecolor=SLICE_COLOR, alpha=0.55, label="SLICE — AP rank"),
        Patch(facecolor=SLICE_COLOR, alpha=0.95, label="SLICE — FP rank"),
        Patch(facecolor=BASE_COLOR,  alpha=0.55, label="baseline — AP rank"),
        Patch(facecolor=BASE_COLOR,  alpha=0.95, label="baseline — FP rank"),
    ], loc="lower right", fontsize=8.5, framealpha=0.92, edgecolor="lightgray")
    plt.tight_layout()
    plt.savefig(f"{OUT}/cl_mean_rank.pdf", bbox_inches="tight")
    plt.savefig(f"{OUT}/cl_mean_rank.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Plot 8: win/loss heatmap
# ============================================================
def plot_winloss():
    slice_keys = [k for k in selected_slice if k not in WEAK_METHODS]
    base_keys  = list(baselines.keys())
    if not slice_keys or not base_keys:
        return

    def winloss(axis):
        M = np.zeros((len(slice_keys), len(base_keys), 2), dtype=int)
        for i, sm in enumerate(slice_keys):
            sd = selected_slice[sm]
            for j, bm in enumerate(base_keys):
                bd = baselines[bm]
                common = set(sd.keys()) & set(bd.keys())
                wins = sum(1 for s in common if sd[s][axis] > bd[s][axis])
                M[i, j] = (wins, len(common))
        return M

    W_ap = winloss(0)
    W_fp = winloss(1)

    fig, axes = plt.subplots(1, 2, figsize=(max(8, len(base_keys) * 2.5),
                                            max(4, len(slice_keys) * 1.2 + 1.5)))
    for ax, W, title in [(axes[0], W_ap, "Wins on AP"),
                         (axes[1], W_fp, "Wins on FP")]:
        ratios = np.array([[w[0] / w[1] if w[1] > 0 else np.nan
                            for w in row] for row in W])
        im = ax.imshow(ratios, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        for i in range(len(slice_keys)):
            for j in range(len(base_keys)):
                wins, total = W[i, j]
                ratio = wins / total if total > 0 else 0
                text_color = "white" if (ratio < 0.25 or ratio > 0.75) else "black"
                ax.text(j, i, f"{wins}/{total}", ha="center", va="center",
                        fontsize=13, fontweight="bold", color=text_color)

        ax.set_xticks(range(len(base_keys)))
        ax.set_xticklabels([labels_baseline[k] for k in base_keys], fontsize=10)
        ax.set_yticks(range(len(slice_keys)))
        ax.set_yticklabels([labels_selected[k] for k in slice_keys], fontsize=10)
        ax.set_xlabel("baseline", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("SLICE method", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xticks(np.arange(-0.5, len(base_keys),  1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(slice_keys), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)

    fig.colorbar(im, ax=axes, shrink=0.7, pad=0.02).set_label("win ratio", fontsize=10)
    fig.suptitle("How often does each SLICE method beat each baseline?",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.savefig(f"{OUT}/cl_winloss.pdf", bbox_inches="tight")
    plt.savefig(f"{OUT}/cl_winloss.png", dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================
# Run all
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CL result plots from CSV.")
    parser.add_argument("--csv", default=CSV_PATH,
                        help=f"Path to results CSV (default: {CSV_PATH})")
    parser.add_argument("--out", default=OUT,
                        help=f"Output directory for figures (default: {OUT})")
    args = parser.parse_args()

    OUT = args.out
    os.makedirs(OUT, exist_ok=True)

    load_data(args.csv)

    print("\nGenerating all plots...")
    plot_scatter_selected();    print("  [1/8] cl_scatter_selected")
    plot_scatter_all_variants();print("  [2/8] cl_scatter_all_variants")
    plot_highlight_seq();       print(f"  [3/8] cl_scatter_{HIGHLIGHT_SEQ}")
    plot_relative_gain();       print("  [4/8] cl_relative_gain")
    plot_skill_score();         print("  [5/8] cl_skill_score")
    plot_radar();               print("  [6/8] cl_radar")
    plot_mean_rank();           print("  [7/8] cl_mean_rank")
    plot_winloss();             print("  [8/8] cl_winloss")
    print(f"\nDone. Output in {OUT}/")
