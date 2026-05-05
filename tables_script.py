#!/usr/bin/env python3
"""
Generate LaTeX tables for NeurIPS 2026 paper.
Produces multiple layout variants — set CSV / OUTDIR below before running.

Usage:  python generate_tables_v2.py
"""

import pandas as pd
import numpy as np
import os, re, textwrap, subprocess

# ── EDIT THESE PATHS BEFORE RUNNING ON THE SERVER ────────────────
CSV        = "results.csv"
OUTDIR     = "./tables_4"
BUILD_PDFS = True   # set False to skip compilation
# ─────────────────────────────────────────────────────────────────

os.makedirs(OUTDIR, exist_ok=True)

# ── sequence metadata ────────────────────────────────────────────
# Maps seq_name in CSV → substring used to detect it inside method strings
SEQ_PAT = {
    "NI-Seq-Dummy":       "ni_seq_dummy",
    "NI-Seq-G1":          "ni_seq_g1",
    "NI-Seq-G2":          "ni_seq_g2",
    "NI-Seq-Opposite-v2": "ni_seq_opposite_v2",
    "NI-Seq-Opposite-v3": "ni_seq_opposite_v3",
    "NI-Seq-Opposite-v4": "ni_seq_opposite_v4",
    "TRACE":              "trace",
}

SEQ_SHORT = {
    "NI-Seq-G1":          "G1",
    "NI-Seq-G2":          "G2",
    "TRACE":              "TRACE",
    "NI-Seq-Opposite-v2": "Opp-v2",
    "NI-Seq-Opposite-v3": "Opp-v3",
    "NI-Seq-Opposite-v4": "Opp-v4",
}
STANDARD_SEQS    = ["NI-Seq-G1", "NI-Seq-G2", "TRACE"]
ADVERSARIAL_SEQS = ["NI-Seq-Opposite-v2", "NI-Seq-Opposite-v3", "NI-Seq-Opposite-v4"]
ALL_SEQS         = STANDARD_SEQS + ADVERSARIAL_SEQS

# ── method taxonomy ──────────────────────────────────────────────
# Maps method-prefix (from CSV) → (display_name, sort_key, canonical_key)
# sort_key < 10  → treated as a baseline (gets separated by \midrule)
# sort_key >= 10 → our method variants
#
# Prefixes that share the same canonical_key are deduplicated: only the
# first-seen row (in sort order) is kept, so duplicates from _full_eval /
# _projvariants variants don't create phantom rows.
#
# NOTE: "slice" is the prefix for slice_ni_seq_g1_full_eval which you
# confirmed is the basic SLICE (LoRA-GA SVD) variant.
METHOD_META = {
    # ── baselines ────────────────────────────────────────────────
    "vanilla":                         ("Vanilla LoRA",                              0, "vanilla"),
    "full_eval_vanilla_pdbs32_alpha2": ("Vanilla LoRA",                              0, "vanilla"),
    "debug_vanilla":                   ("Vanilla LoRA",                              0, "vanilla"),
    "loram":                           ("LoRAM",                                     1, "loram"),
    "lora_ga_lora_ga":                 ("LoRA-GA",                                   2, "lora_ga"),
    "fix_lora_ga":                     ("LoRA-GA",                                   2, "lora_ga"),
    # ── our method variants ──────────────────────────────────────
    "slice_var_global_cagrad_c050":    (r"\method{} (CAGrad $c{=}0.50$)",           10, "cagrad_050"),
    "slice_var_global_cagrad_c075":    (r"\method{} (CAGrad $c{=}0.75$)",           11, "cagrad_075"),
    # slice_var_slice_basic_lora_ga  = LoRA-GA SVD selection rule
    "slice_var_slice_basic_lora_ga":   (r"\method{} (LoRA-GA SVD)",                 12, "slice_lora_ga_svd"),
    # slice  =  slice_ni_seq_g1_full_eval = confirmed basic SLICE (LoRA-GA SVD)
    "slice":                           (r"\method{} (LoRA-GA SVD)",                 12, "slice_lora_ga_svd"),
}

# Which canonical keys appear in the MAIN tables
MAIN_KEYS = [
    "vanilla", "loram", "lora_ga",
    "cagrad_050", "cagrad_075", "slice_lora_ga_svd",
]

# ── metric display info ──────────────────────────────────────────
# key → (column header, higher_is_better, display_scale)
METRIC_INFO = {
    "AP":                    (r"AP $\uparrow$",      True,  100),
    "FP":                    (r"FP $\uparrow$",      True,  100),
    "Forget":                (r"Fgt $\downarrow$",   False, 100),
    "GP":                    (r"GP $\uparrow$",      True,  100),
    "IP":                    (r"IP $\uparrow$",      True,  100),
    "gp_hellaswag":          ("Hella.",              True,  100),
    "gp_commonsenseqa":      ("Com.",                True,  100),
    "gp_alpaca":             ("Alpa.",               True,  100),
    "gp_bbh_object_counting":("Ob.",                 True,  100),
    "ip_hellaswag":          ("Hella.",              True,  100),
    "ip_commonsenseqa":      ("Com.",                True,  100),
    "ip_alpaca":             ("Alpa.",               True,  100),
    "ip_bbh_object_counting":("Ob.",                 True,  100),
}


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════

def extract_prefix_rank(method: str, seq_name: str):
    """Strip the seq-name fragment and rank suffix from a method string."""
    pat = SEQ_PAT.get(seq_name, "")
    idx = method.find(pat)
    if idx == -1:
        # Fallback: strip all known trailing suffixes
        prefix = re.sub(
            r"(_full_eval|_projvariants|_alphasweep_\w+|_r\d+)+$", "", method
        )
        rank = 128 if "_r128" in method else 64
        return prefix, rank
    prefix = method[:idx].rstrip("_")
    suffix = method[idx + len(pat):]
    rank = 128 if "r128" in suffix else 64
    return prefix, rank


def _run_priority(method: str) -> int:
    """
    Priority for deduplication when multiple runs exist for the same
    (seq_name, canon, rank).  Lower = preferred (kept).
      0  projvariants  — core r64 runs
      1  bare          — plain _rNN suffix
      2  full_eval     — older full-eval runs
    """
    if "projvariants" in method:
        return 0
    if "full_eval" in method:
        return 2
    return 1


def _canon(prefix):
    m = METHOD_META.get(prefix)
    return m[2] if m else None


def _disp(prefix):
    m = METHOD_META.get(prefix)
    return m[0] if m else prefix


def _sort(prefix):
    m = METHOD_META.get(prefix)
    return m[1] if m else 99


def fmt(val, scale=100, dec=2):
    if pd.isna(val):
        return "--"
    return f"{val * scale:.{dec}f}"


def fmt_bold(val, is_best, scale=100, dec=2):
    s = fmt(val, scale, dec)
    return rf"\textbf{{{s}}}" if (s != "--" and is_best) else s


def _tabular_wrap(col_spec: str, header_lines: str, body_lines: str) -> str:
    """
    Return a complete tabular block sized to prevent overflow in a table*.
    Threshold is based on column count in col_spec (| separators ignored).
    
    <= 10 cols  : small,        tabcolsep 4pt, no resizebox
    11-13 cols  : footnotesize, tabcolsep 3pt, no resizebox
    14-18 cols  : scriptsize,   tabcolsep 2pt, no resizebox
    19+ cols    : scriptsize,   tabcolsep 2pt, + resizebox linewidth
    """
    ncols = len(re.sub(r"[|@{}]", "", col_spec))  # only count actual col letters

    if ncols >= 19:
        font      = r"\\scriptsize"
        sep       = "2pt"
        pre, post = r"\resizebox{\linewidth}{!}{%", "}"
    elif ncols >= 14:
        font      = r"\\scriptsize"
        sep       = "2pt"
        pre, post = "", ""
    elif ncols >= 11:
        font      = r"\footnotesize"
        sep       = "3pt"
        pre, post = "", ""
    else:
        font      = r"\\small"
        sep       = "4pt"
        pre, post = "", ""

    inner = (
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        f"\\toprule\n"
        f"{header_lines}\n"
        f"\\midrule\n"
        f"{body_lines}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}"
    )
    if pre:
        inner = f"{pre}\n{inner}\n{post}"

    return f"{font}\n\\setlength{{\\tabcolsep}}{{{sep}}}\n{inner}"


def _table_env(caption: str, label: str, content: str) -> str:
    """Wrap content in a table* float."""
    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{content}\n"
        "\\end{table*}\n"
    )


# ════════════════════════════════════════════════════════════════
# Data loading
# ════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV)

    info          = df.apply(lambda r: extract_prefix_rank(r["method"], r["seq_name"]), axis=1)
    df["prefix"]  = [i[0] for i in info]
    df["rank"]    = [i[1] for i in info]
    df["canon"]   = df["prefix"].map(_canon)
    df["disp"]    = df["prefix"].map(_disp)
    df["sort_key"]= df["prefix"].map(_sort)

    # ── deduplicate: if multiple rows share (seq_name, canon, rank),
    # keep the highest-priority run.  Priority: projvariants > bare > full_eval > alphasweep.
    # Drop alphasweep variants entirely — not reported in main tables
    n_before = len(df)
    df = df[~df["method"].str.contains("alphasweep", na=False)].copy()
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"  [info] dropped {n_dropped} alphasweep rows (excluded from all tables)")

    # ── deduplicate: if multiple rows share (seq_name, canon, rank),
    # keep the highest-priority run.  Priority: projvariants > bare > full_eval.
    df["_run_priority"] = df["method"].map(_run_priority)
    df = (df.sort_values("_run_priority")
            .drop_duplicates(subset=["seq_name", "canon", "rank"])
            .drop(columns=["_run_priority"])
            .sort_values(["seq_name", "sort_key", "rank"])
            .reset_index(drop=True))

    n_unmapped = df["canon"].isna().sum()
    if n_unmapped:
        unmapped = df[df["canon"].isna()]["prefix"].unique()
        print(f"  [info] {n_unmapped} rows with no canonical mapping (excluded from main tables):")
        for u in sorted(unmapped):
            print(f"         {u}")

    return df


# ════════════════════════════════════════════════════════════════
# Shared pivot + best-value logic
# ════════════════════════════════════════════════════════════════

def _pivot(df, seqs, metrics, rank):
    """
    Returns (rows_data, sorted_keys) where rows_data[canon] = {
        "disp": str, "sort": int, (seq, metric): float, ...
    }
    """
    sub = df[
        (df["rank"] == rank) &
        (df["canon"].isin(MAIN_KEYS)) &
        (df["seq_name"].isin(seqs))
    ].copy()

    rows_data = {}
    for _, r in sub.iterrows():
        k = r["canon"]
        if k not in rows_data:
            rows_data[k] = {"disp": r["disp"], "sort": r["sort_key"]}
        for m in metrics:
            if m in r.index:
                rows_data[k][(r["seq_name"], m)] = r[m]

    sorted_keys = sorted(rows_data, key=lambda k: rows_data[k]["sort"])
    return rows_data, sorted_keys


def _bests(rows_data, sorted_keys, seqs, metrics):
    """Return dict[(seq, metric)] → set of canon keys that are best."""
    bests = {}
    for seq in seqs:
        for m in metrics:
            hi  = METRIC_INFO[m][1]
            vals = {k: rows_data[k].get((seq, m), np.nan) for k in sorted_keys}
            valid = {k: v for k, v in vals.items() if not pd.isna(v)}
            if valid:
                best_v = max(valid.values()) if hi else min(valid.values())
                bests[(seq, m)] = {k for k, v in valid.items() if v == best_v}
            else:
                bests[(seq, m)] = set()
    return bests


def _midrule_body(rows_data, sorted_keys, row_fn):
    """
    Call row_fn(k, rd) for each key; insert \\midrule between baselines
    (sort_key < 10) and our methods (sort_key >= 10).
    Returns list of line strings.
    """
    lines = []
    prev_bl = None
    for k in sorted_keys:
        rd = rows_data[k]
        cur_bl = rd["sort"] < 10
        if prev_bl is not None and prev_bl and not cur_bl:
            lines.append(r"\midrule")
        prev_bl = cur_bl
        line = row_fn(k, rd)
        if line is not None:
            lines.append(line)
    return lines


# ════════════════════════════════════════════════════════════════
# Table generators
# ════════════════════════════════════════════════════════════════

# ── A / B: CL metrics (AP / FP / Forget) grouped by sequence ────

def table_cl_across_seqs(df, seqs, metrics, rank, label, caption):
    rows_data, sorted_keys = _pivot(df, seqs, metrics, rank)
    if not rows_data:
        return ""
    bests = _bests(rows_data, sorted_keys, seqs, metrics)

    n_m   = len(metrics)
    col_spec = "l" + "".join(f"|{'c' * n_m}" for _ in seqs)

    h1_parts = [""]
    h2_parts = ["Method"]
    for seq in seqs:
        h1_parts.append(rf"\multicolumn{{{n_m}}}{{c}}{{{SEQ_SHORT[seq]}}}")
        for m in metrics:
            h2_parts.append(METRIC_INFO[m][0])
    h1 = " & ".join(h1_parts) + r" \\ \cmidrule(l){2-" + str(1 + n_m * len(seqs)) + "}"
    h2 = " & ".join(h2_parts) + r" \\"

    def row_fn(k, rd):
        parts = [rd["disp"]]
        for seq in seqs:
            for m in metrics:
                parts.append(fmt_bold(rd.get((seq, m), np.nan),
                                      k in bests.get((seq, m), set())))
        return " & ".join(parts) + r" \\"

    body  = "\n".join(_midrule_body(rows_data, sorted_keys, row_fn))
    inner = _tabular_wrap(col_spec, f"{h1}\n{h2}", body)
    return _table_env(caption, label, inner)


# ── C: GP + IP + CL aggregates grouped by sequence ──────────────

def table_gp_ip_cl_across_seqs(df, seqs, rank, label, caption):
    return table_cl_across_seqs(df, seqs, ["GP", "IP", "AP", "FP", "Forget"],
                                 rank, label, caption)


# ── D: Full benchmark breakdown, one sub-table per sequence ──────

def table_full_breakdown(df, seqs, rank, label, caption):
    gp_cols  = ["gp_hellaswag", "gp_commonsenseqa", "gp_alpaca", "gp_bbh_object_counting"]
    ip_cols  = ["ip_hellaswag", "ip_commonsenseqa", "ip_alpaca", "ip_bbh_object_counting"]
    cl_cols  = ["AP", "FP", "Forget"]
    all_cols = gp_cols + ["GP"] + ip_cols + ["IP"] + cl_cols

    # col_spec: method | 4 GP + GP_agg | 4 IP + IP_agg | AP FP Forget
    col_spec = "l|cccc|c|cccc|c|ccc"
    h1 = (r" & \multicolumn{5}{c|}{Zero-shot GP}"
          r" & \multicolumn{5}{c|}{In-context IP}"
          r" & \multicolumn{3}{c}{CL} \\")
    gp_hdrs = [METRIC_INFO[c][0] for c in gp_cols] + [METRIC_INFO["GP"][0]]
    ip_hdrs = [METRIC_INFO[c][0] for c in ip_cols] + [METRIC_INFO["IP"][0]]
    cl_hdrs = [METRIC_INFO[c][0] for c in cl_cols]
    h2 = "Method & " + " & ".join(gp_hdrs + ip_hdrs + cl_hdrs) + r" \\"

    tables = []
    for seq in seqs:
        sub = df[
            (df["rank"] == rank) &
            (df["canon"].isin(MAIN_KEYS)) &
            (df["seq_name"] == seq)
        ].copy().sort_values("sort_key")
        if sub.empty:
            continue

        # bests per column
        bests = {}
        for c in all_cols:
            valid = sub[c].dropna()
            if valid.empty:
                bests[c] = set()
                continue
            best_v = valid.max() if METRIC_INFO[c][1] else valid.min()
            bests[c] = set(valid[valid == best_v].index)

        lines = []
        prev_bl = None
        for _, row in sub.iterrows():
            cur_bl = row["sort_key"] < 10
            if prev_bl is not None and prev_bl and not cur_bl:
                lines.append(r"\midrule")
            prev_bl = cur_bl
            parts = [row["disp"]] + [fmt_bold(row[c], row.name in bests.get(c, set()))
                                      for c in all_cols]
            lines.append(" & ".join(parts) + r" \\")

        body  = "\n".join(lines)
        inner = _tabular_wrap(col_spec, f"{h1}\n{h2}", body)
        seq_label   = f"{label}_{SEQ_SHORT[seq].lower().replace('-', '')}"
        seq_caption = f"{caption} — {SEQ_SHORT[seq]}"
        tables.append(_table_env(seq_caption, seq_label, inner))

    return "\n\n".join(tables)


# ── E: Compact AP + Forget with averages ────────────────────────

def table_compact_ap_forget(df, seqs, rank, label, caption):
    rows_data, sorted_keys = _pivot(df, seqs, ["AP", "Forget"], rank)
    if not rows_data:
        return ""

    # append computed averages column
    display_seqs = seqs + ["__avg__"]
    for k, rd in rows_data.items():
        for m in ["AP", "Forget"]:
            vals = [rd.get((s, m), np.nan) for s in seqs]
            valid = [v for v in vals if not pd.isna(v)]
            rd[("__avg__", m)] = float(np.mean(valid)) if valid else np.nan

    bests = _bests(rows_data, sorted_keys, display_seqs, ["AP", "Forget"])

    col_spec = "l" + "|cc" * (len(seqs) + 1)

    h1_parts = [""]
    h2_parts = ["Method"]
    for seq in display_seqs:
        label_str = SEQ_SHORT.get(seq, "Avg")
        h1_parts.append(rf"\multicolumn{{2}}{{c}}{{{label_str}}}")
        h2_parts += [METRIC_INFO["AP"][0], METRIC_INFO["Forget"][0]]
    h1 = " & ".join(h1_parts) + r" \\"
    h2 = " & ".join(h2_parts) + r" \\"

    def row_fn(k, rd):
        parts = [rd["disp"]]
        for seq in display_seqs:
            for m in ["AP", "Forget"]:
                parts.append(fmt_bold(rd.get((seq, m), np.nan),
                                      k in bests.get((seq, m), set())))
        return " & ".join(parts) + r" \\"

    body  = "\n".join(_midrule_body(rows_data, sorted_keys, row_fn))
    inner = _tabular_wrap(col_spec, f"{h1}\n{h2}", body)
    return _table_env(caption, label, inner)


# ── F: ICLR-style — methods as blocks, sequences as rows ────────

def table_iclr_style(df, seqs, rank, label, caption):
    gp_cols  = ["gp_hellaswag", "gp_commonsenseqa", "gp_alpaca", "gp_bbh_object_counting"]
    ip_cols  = ["ip_hellaswag", "ip_commonsenseqa", "ip_alpaca", "ip_bbh_object_counting"]
    cl_cols  = ["AP", "FP", "Forget"]
    all_cols = gp_cols + ["GP"] + ip_cols + ["IP"] + cl_cols

    col_spec = "l|cccc|c|cccc|c|ccc"
    ncols    = 1 + len(all_cols)

    h1 = (r" & \multicolumn{5}{c|}{Zero-shot GP}"
          r" & \multicolumn{5}{c|}{In-context IP}"
          r" & \multicolumn{3}{c}{CL} \\")
    gp_hdrs = [METRIC_INFO[c][0] for c in gp_cols] + [METRIC_INFO["GP"][0]]
    ip_hdrs = [METRIC_INFO[c][0] for c in ip_cols] + [METRIC_INFO["IP"][0]]
    cl_hdrs = [METRIC_INFO[c][0] for c in cl_cols]
    h2 = " & " + " & ".join(gp_hdrs + ip_hdrs + cl_hdrs) + r" \\"

    sub = df[
        (df["rank"] == rank) &
        (df["canon"].isin(MAIN_KEYS)) &
        (df["seq_name"].isin(seqs))
    ].copy()
    canons = (sub.drop_duplicates("canon")
                 .sort_values("sort_key")["canon"].tolist())

    lines = []
    for ci, can in enumerate(canons):
        csub  = sub[sub["canon"] == can]
        dname = csub.iloc[0]["disp"]
        if ci > 0:
            lines.append(r"\midrule")
        lines.append(rf"\multicolumn{{{ncols}}}{{l}}{{\textit{{{dname}}}}} \\")
        for seq in seqs:
            row = csub[csub["seq_name"] == seq]
            parts = [SEQ_SHORT[seq]]
            if row.empty:
                parts += ["--"] * len(all_cols)
            else:
                r_ = row.iloc[0]
                parts += [fmt(r_[c]) for c in all_cols]
            lines.append(" & ".join(parts) + r" \\")

    body  = "\n".join(lines)
    inner = _tabular_wrap(col_spec, f"{h1}\n{h2}", body)
    return _table_env(caption, label, inner)


# ── G: Rank ablation — r64 vs r128 ──────────────────────────────

def table_rank_comparison(df, seqs, label, caption):
    metrics = ["AP", "FP", "Forget"]
    sub = df[
        (df["canon"].isin(MAIN_KEYS)) &
        (df["seq_name"].isin(seqs))
    ].copy().sort_values(["sort_key", "rank"])

    rows = {}
    for _, r in sub.iterrows():
        key = (r["canon"], r["rank"])
        if key not in rows:
            rows[key] = {"disp": r["disp"], "sort": r["sort_key"], "rank": r["rank"]}
        for m in metrics:
            rows[key][(r["seq_name"], m)] = r[m]

    sorted_keys = sorted(rows, key=lambda k: (rows[k]["sort"], rows[k]["rank"]))

    # bests per (seq, metric, rank)
    bests = {}
    for rv in [64, 128]:
        for seq in seqs:
            for m in metrics:
                hi   = METRIC_INFO[m][1]
                vals = {k: rows[k].get((seq, m), np.nan)
                        for k in sorted_keys if k[1] == rv}
                valid = {k: v for k, v in vals.items() if not pd.isna(v)}
                if valid:
                    best_v = max(valid.values()) if hi else min(valid.values())
                    bests[(seq, m, rv)] = {k for k, v in valid.items() if v == best_v}

    n_m      = len(metrics)
    col_spec = "l|c" + "".join(f"|{'c' * n_m}" for _ in seqs)

    h1_parts = ["", ""]
    h2_parts = ["Method", "$r$"]
    for seq in seqs:
        h1_parts.append(rf"\multicolumn{{{n_m}}}{{c}}{{{SEQ_SHORT[seq]}}}")
        for m in metrics:
            h2_parts.append(METRIC_INFO[m][0])
    h1 = " & ".join(h1_parts) + r" \\"
    h2 = " & ".join(h2_parts) + r" \\"

    lines     = []
    prev_canon = None
    for k in sorted_keys:
        rd = rows[k]
        if prev_canon is not None and prev_canon != k[0]:
            lines.append(r"\midrule")
        prev_canon = k[0]
        parts = [rd["disp"], str(rd["rank"])]
        for seq in seqs:
            for m in metrics:
                parts.append(fmt_bold(rd.get((seq, m), np.nan),
                                      k in bests.get((seq, m, rd["rank"]), set())))
        lines.append(" & ".join(parts) + r" \\")

    body  = "\n".join(lines)
    inner = _tabular_wrap(col_spec, f"{h1}\n{h2}", body)
    return _table_env(caption, label, inner)


# ── H: Delta from vanilla ────────────────────────────────────────

def table_delta_from_vanilla(df, seqs, rank, label, caption):
    sub = df[
        (df["rank"] == rank) &
        (df["canon"].isin(MAIN_KEYS)) &
        (df["seq_name"].isin(seqs))
    ].copy()
    if sub.empty:
        return ""

    vanilla_ref = {}
    for seq in seqs:
        v = sub[(sub["canon"] == "vanilla") & (sub["seq_name"] == seq)]
        if not v.empty:
            vanilla_ref[seq] = {"GP": v.iloc[0]["GP"], "IP": v.iloc[0]["IP"]}

    col_spec = "ll|cc|cc|ccc"
    h1 = (r" & & \multicolumn{2}{c|}{General Task}"
          r" & \multicolumn{2}{c|}{In-context}"
          r" & \multicolumn{3}{c}{CL Metrics} \\")
    h2 = (r"Seq. & Method"
          rf" & {METRIC_INFO['GP'][0]} & $\Delta$GP"
          rf" & {METRIC_INFO['IP'][0]} & $\Delta$IP"
          rf" & {METRIC_INFO['AP'][0]} & {METRIC_INFO['FP'][0]}"
          rf" & {METRIC_INFO['Forget'][0]} \\")

    lines = []
    for si, seq in enumerate(seqs):
        if si > 0:
            lines.append(r"\midrule")
        seq_sub  = sub[sub["seq_name"] == seq].sort_values("sort_key")
        first    = True
        prev_bl  = None
        for _, row in seq_sub.iterrows():
            cur_bl = row["sort_key"] < 10
            if prev_bl is not None and prev_bl and not cur_bl:
                lines.append(r"\cmidrule{2-9}")
            prev_bl = cur_bl
            sn = SEQ_SHORT[seq] if first else ""
            first = False
            gp_d = ip_d = ""
            if seq in vanilla_ref and not pd.isna(row["GP"]):
                gp_d = f"{(row['GP'] - vanilla_ref[seq]['GP']) * 100:+.2f}"
            if seq in vanilla_ref and not pd.isna(row["IP"]):
                ip_d = f"{(row['IP'] - vanilla_ref[seq]['IP']) * 100:+.2f}"
            parts = [sn, row["disp"],
                     fmt(row["GP"]), gp_d,
                     fmt(row["IP"]), ip_d,
                     fmt(row["AP"]), fmt(row["FP"]), fmt(row["Forget"])]
            lines.append(" & ".join(parts) + r" \\")

    body  = "\n".join(lines)
    inner = _tabular_wrap(col_spec, f"{h1}\n{h2}", body)
    return _table_env(caption, label, inner)


# ── I: ICLR-style delta table — baseline row + Δ rows per method ─
# Mirrors the FVG paper layout but for AP / FP / Forget.
# Structure:
#   rows: for each baseline, one row showing its raw values, then one
#         "+METHOD" row per SLICE variant showing Δ from that baseline.
#   columns: sequences × {AP, FP, Forget}
# A positive Δ on AP/FP is good; a negative Δ on Forget is good.
# Δ values are coloured via \textcolor if xcolor is loaded (graceful fallback).

def table_iclr_delta(df, seqs, rank, label, caption):
    metrics  = ["AP", "FP", "Forget"]
    # higher_delta_good: positive = improvement for AP/FP; negative = improvement for Forget
    delta_good_pos = {"AP", "FP"}   # positive delta is good
    delta_good_neg = {"Forget"}     # negative delta is good

    sub = df[
        (df["rank"] == rank) &
        (df["canon"].isin(MAIN_KEYS)) &
        (df["seq_name"].isin(seqs))
    ].copy()
    if sub.empty:
        return ""

    baselines_canon = ["vanilla", "loram", "lora_ga"]
    slice_canon     = ["cagrad_050", "cagrad_075", "slice_lora_ga_svd"]

    # Build lookup: (seq, canon) → row
    lookup = {}
    for _, row in sub.iterrows():
        lookup[(row["seq_name"], row["canon"])] = row

    def get_val(seq, canon, metric):
        r = lookup.get((seq, canon))
        return r[metric] if r is not None and not pd.isna(r[metric]) else np.nan

    def get_disp(canon):
        for prefix, meta in METHOD_META.items():
            if meta[2] == canon:
                return meta[0]
        return canon

    # col spec: Method | (AP FP Fgt) per seq
    n_m      = len(metrics)
    col_spec = "l" + "".join(f"|{'c' * n_m}" for _ in seqs)

    h1_parts = [""]
    h2_parts = ["Method"]
    for seq in seqs:
        h1_parts.append(rf"\multicolumn{{{n_m}}}{{c}}{{{SEQ_SHORT[seq]}}}")
        for m in metrics:
            h2_parts.append(METRIC_INFO[m][0])
    h1 = " & ".join(h1_parts) + r" \\ \cmidrule(l){2-" + str(1 + n_m * len(seqs)) + "}"
    h2 = " & ".join(h2_parts) + r" \\"

    lines = []
    for bi, bl_canon in enumerate(baselines_canon):
        if bi > 0:
            lines.append(r"\midrule")

        # baseline raw row
        bl_disp = get_disp(bl_canon)
        parts   = [bl_disp]
        for seq in seqs:
            for m in metrics:
                v = get_val(seq, bl_canon, m)
                parts.append(fmt(v))
        lines.append(" & ".join(parts) + r" \\")

        # one delta row per SLICE variant
        for sl_canon in slice_canon:
            sl_disp = get_disp(sl_canon)
            # shorten display: strip the \method{} prefix for delta rows
            short_disp = re.sub(r"^\\method\{\}", r"\\method{}", sl_disp)
            short_disp = re.sub(r"^\\method\{\} ", r"\\method{} ", short_disp)
            parts = [rf"\quad \textit{{+{sl_disp}}}"]
            for seq in seqs:
                bl_v = get_val(seq, bl_canon, m)   # reuse loop var — fix below
                has_any = any(
                    not np.isnan(get_val(seq, sl_canon, mm)) for mm in metrics
                )
                for m in metrics:
                    bl_v  = get_val(seq, bl_canon, m)
                    sl_v  = get_val(seq, sl_canon, m)
                    if np.isnan(bl_v) or np.isnan(sl_v):
                        parts.append("--")
                        continue
                    delta = (sl_v - bl_v) * 100
                    # positive delta good for AP/FP; negative good for Forget
                    good = (delta > 0 and m in delta_good_pos) or \
                           (delta < 0 and m in delta_good_neg)
                    bad  = (delta < 0 and m in delta_good_pos) or \
                           (delta > 0 and m in delta_good_neg)
                    s = f"{delta:+.2f}"
                    if good:
                        s = rf"\textcolor{{ForestGreen}}{{\textbf{{{s}}}}}"
                    elif bad:
                        s = rf"\textcolor{{BrickRed}}{{{s}}}"
                    parts.append(s)
            lines.append(" & ".join(parts) + r" \\")

    body  = "\n".join(lines)
    inner = _tabular_wrap(col_spec, f"{h1}\n{h2}", body)
    # Remind user to load xcolor in preamble
    note  = "% Requires: \\usepackage[dvipsnames]{xcolor}\n"
    return note + _table_env(caption, label, inner)




def main():
    df = load_data()
    print(f"\n  Sequences found: {sorted(df['seq_name'].unique())}")
    print(f"  Ranks found:     {sorted(df['rank'].unique())}\n")

    generated = {}

    # ── A: CL only, all sequences ────────────────────────────────
    generated["A1_cl_all_seqs_r64.tex"] = table_cl_across_seqs(
        df, ALL_SEQS, ["AP", "FP", "Forget"], rank=64,
        label="tab:cl_all_r64",
        caption="Continual learning metrics across all sequences (rank 64)",
    )

    # ── B: CL only, standard / adversarial split ─────────────────
    generated["B1_cl_standard_r64.tex"] = table_cl_across_seqs(
        df, STANDARD_SEQS, ["AP", "FP", "Forget"], rank=64,
        label="tab:cl_standard_r64",
        caption="CL metrics on standard sequences (rank 64)",
    )
    generated["B2_cl_adversarial_r64.tex"] = table_cl_across_seqs(
        df, ADVERSARIAL_SEQS, ["AP", "FP", "Forget"], rank=64,
        label="tab:cl_adversarial_r64",
        caption=r"CL metrics on adversarial \textsc{NI-Seq-Opposite} sequences (rank 64)",
    )

    # ── C: GP + IP + CL aggregates ───────────────────────────────
    generated["C1_gp_ip_cl_standard_r64.tex"] = table_gp_ip_cl_across_seqs(
        df, STANDARD_SEQS, rank=64,
        label="tab:gp_ip_cl_standard_r64",
        caption="GP, IP, and CL metrics on standard sequences (rank 64)",
    )
    generated["C2_gp_ip_cl_adversarial_r64.tex"] = table_gp_ip_cl_across_seqs(
        df, ADVERSARIAL_SEQS, rank=64,
        label="tab:gp_ip_cl_adversarial_r64",
        caption=r"GP, IP, and CL metrics on adversarial \textsc{NI-Seq-Opposite} sequences (rank 64)",
    )
    generated["C3_gp_ip_cl_all_r64.tex"] = table_gp_ip_cl_across_seqs(
        df, ALL_SEQS, rank=64,
        label="tab:gp_ip_cl_all_r64",
        caption="GP, IP, and CL metrics across all sequences (rank 64)",
    )

    # ── D: Full benchmark breakdown ───────────────────────────────
    generated["D1_full_breakdown_standard_r64.tex"] = table_full_breakdown(
        df, STANDARD_SEQS, rank=64,
        label="tab:breakdown_standard",
        caption="Full benchmark breakdown — standard sequences (rank 64)",
    )
    generated["D2_full_breakdown_adversarial_r64.tex"] = table_full_breakdown(
        df, ADVERSARIAL_SEQS, rank=64,
        label="tab:breakdown_adversarial",
        caption=r"Full benchmark breakdown — adversarial \textsc{NI-Seq-Opposite} sequences (rank 64)",
    )

    # ── E: Compact AP + Forget + averages ────────────────────────
    generated["E1_compact_ap_forget_all_r64.tex"] = table_compact_ap_forget(
        df, ALL_SEQS, rank=64,
        label="tab:compact_all_r64",
        caption="AP and Forget across all sequences with averages (rank 64)",
    )
    generated["E2_compact_ap_forget_adversarial_r64.tex"] = table_compact_ap_forget(
        df, ADVERSARIAL_SEQS, rank=64,
        label="tab:compact_adversarial_r64",
        caption=r"AP and Forget on adversarial \textsc{NI-Seq-Opposite} sequences with averages (rank 64)",
    )

    # ── F: ICLR-style ────────────────────────────────────────────
    generated["F1_iclr_style_all_r64.tex"] = table_iclr_style(
        df, ALL_SEQS, rank=64,
        label="tab:iclr_all_r64",
        caption="Full results in ICLR-style layout — all sequences (rank 64)",
    )
    generated["F2_iclr_style_adversarial_r64.tex"] = table_iclr_style(
        df, ADVERSARIAL_SEQS, rank=64,
        label="tab:iclr_adversarial_r64",
        caption=r"Full results in ICLR-style layout — \textsc{NI-Seq-Opposite} sequences (rank 64)",
    )

    # ── G: Rank ablation ─────────────────────────────────────────
    generated["G1_rank_ablation_all.tex"] = table_rank_comparison(
        df, ALL_SEQS,
        label="tab:rank_ablation_all",
        caption="Rank ablation ($r{=}64$ vs $r{=}128$) across all sequences (CL metrics)",
    )
    generated["G2_rank_ablation_adversarial.tex"] = table_rank_comparison(
        df, ADVERSARIAL_SEQS,
        label="tab:rank_ablation_adversarial",
        caption=r"Rank ablation ($r{=}64$ vs $r{=}128$) on \textsc{NI-Seq-Opposite} sequences",
    )

    # ── H: Delta from vanilla ─────────────────────────────────────
    generated["H1_delta_adversarial_r64.tex"] = table_delta_from_vanilla(
        df, ADVERSARIAL_SEQS, rank=64,
        label="tab:delta_adversarial_r64",
        caption=r"Results on adversarial sequences with $\Delta$ from Vanilla LoRA (rank 64)",
    )
    generated["H2_delta_all_r64.tex"] = table_delta_from_vanilla(
        df, ALL_SEQS, rank=64,
        label="tab:delta_all_r64",
        caption=r"Results on all sequences with $\Delta$ from Vanilla LoRA (rank 64)",
    )

    # ── I: ICLR-style delta table (baseline + Δ rows) ────────────
    generated["I1_iclr_delta_standard_r64.tex"] = table_iclr_delta(
        df, STANDARD_SEQS, rank=64,
        label="tab:iclr_delta_standard_r64",
        caption="CL metrics on standard sequences: baseline values with $\\Delta$ for each \\method{} variant (rank 64)",
    )
    generated["I2_iclr_delta_adversarial_r64.tex"] = table_iclr_delta(
        df, ADVERSARIAL_SEQS, rank=64,
        label="tab:iclr_delta_adversarial_r64",
        caption=r"CL metrics on adversarial \textsc{NI-Seq-Opposite} sequences: baseline values with $\Delta$ for each \method{} variant (rank 64)",
    )
    generated["I3_iclr_delta_all_r64.tex"] = table_iclr_delta(
        df, ALL_SEQS, rank=64,
        label="tab:iclr_delta_all_r64",
        caption="CL metrics across all sequences: baseline values with $\\Delta$ for each \\method{} variant (rank 64)",
    )


    STANDALONE_PREAMBLE = textwrap.dedent(r"""
        \documentclass[11pt]{article}
        \usepackage[margin=0.75in]{geometry}
        \usepackage{booktabs}
        \usepackage{amsmath}
        \usepackage{graphicx}
        \usepackage[dvipsnames]{xcolor}
        \newcommand{\method}{PCGrad}
        \begin{document}
    """).lstrip()

    for fname, tex in generated.items():
        if not tex:
            print(f"  [skip] {fname}  (empty)")
            continue
        path = os.path.join(OUTDIR, fname)
        with open(path, "w") as f:
            f.write(STANDALONE_PREAMBLE + tex + r"\end{document}" + "\n")
        # count columns for a quick sanity check
        cs = re.search(r"\\begin\{tabular\}\{([^}]+)\}", tex)
        ncols = len(re.sub(r"[|@{}]", "", cs.group(1))) if cs else "?"
        print(f"  ✓ {fname}  ({ncols} cols)")

    # ── master preview ────────────────────────────────────────────
    layout_desc = {
        "A": "CL metrics (AP/FP/Forget) across all sequences",
        "B": "CL metrics: standard vs adversarial split",
        "C": "GP + IP + CL aggregates across sequences",
        "D": "Full benchmark breakdown per sequence",
        "E": "Compact: AP + Forget with sequence averages",
        "F": "ICLR-style: sequences as rows, methods as blocks",
        "G": "Rank ablation: r64 vs r128",
        "H": "Delta from Vanilla LoRA baseline",
        "I": "ICLR-style: baseline rows + colour-coded delta rows per SLICE variant",
    }
    master = textwrap.dedent(r"""
    % ============================================================
    % Master preview — compile to see ALL table layouts at once.
    % ============================================================
    \documentclass[11pt]{article}
    \usepackage[margin=0.75in]{geometry}
    \usepackage{booktabs}
    \usepackage{amsmath}
    \usepackage{graphicx}   % needed for \resizebox
    \newcommand{\method}{PCGrad}

    \begin{document}
    \\section*{Table Layouts — NeurIPS 2026}
    """).lstrip()

    for prefix in sorted({f[0] for f in generated if generated.get(f)}):
        master += rf"\\subsection*{{Layout {prefix}: {layout_desc.get(prefix, '')}}}" + "\n\n"
        for fname in sorted(f for f in generated if f.startswith(prefix) and generated.get(f)):
            master += rf"\input{{{fname}}}" + "\n" + r"\clearpage" + "\n\n"
    master += r"\end{document}" + "\n"

    master_path = os.path.join(OUTDIR, "all_tables_preview.tex")
    with open(master_path, "w") as f:
        f.write(master)
    print(f"\n  ★ Master preview → {master_path}")
    print("  Note: add \\usepackage{{graphicx}} to your paper for \\resizebox to work.")

    if BUILD_PDFS:
        _build_pdfs(OUTDIR, [f for f in generated if generated.get(f)] + ["all_tables_preview.tex"])


def _build_pdfs(outdir, fnames):
    junk_exts = {".aux", ".log", ".synctex.gz", ".fls", ".fdb_latexmk", ".out"}
    print()
    for fname in fnames:
        base = fname.replace(".tex", "")
        result = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-quiet", fname],
            cwd=outdir,
            capture_output=True,
        )
        if result.returncode == 0:
            print(f"  ✓ {base}.pdf")
        else:
            print(f"  ✗ FAILED: {fname}")
            print(result.stdout.decode()[-800:])
        for ext in junk_exts:
            junk = os.path.join(outdir, base + ext)
            if os.path.exists(junk):
                os.remove(junk)


if __name__ == "__main__":
    main()
