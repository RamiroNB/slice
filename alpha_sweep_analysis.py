"""Pull AP/FP/GP and ||BA||_F per run for the alpha-sweep figure.

Walks `<results-root>/<sequence>/<run_name>/` and joins:
  * metrics.json                     -> AP, FP, GP, IP, Forget
  * stages/*/stage_record.json       -> train_report.ba_norms (per-stage, averaged)
  * run_config.json                  -> resolved.lora_alpha (authoritative)

Usage:
    python alpha_sweep_analysis.py
    python alpha_sweep_analysis.py --run-glob '*alphasweep*'
    python alpha_sweep_analysis.py --csv alpha_sweep.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


METRIC_KEYS = ("AP", "FP", "GP", "IP", "Forget")


def _safe_load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _infer_method_from_run_name(run_name: str) -> str:
    if "slice_var_global_cagrad_c050" in run_name:
        return "cagrad_c050"
    if "slice_var_global_cagrad_c075" in run_name:
        return "cagrad_c075"
    if "slice_var_global_cagrad_c025" in run_name:
        return "cagrad_c025"
    if "slice_var_magpreserve" in run_name or "slice_var_global_magpreserve" in run_name:
        return "magpreserve"
    if "slice_var" in run_name:
        return "slice_var_other"
    if run_name.startswith("lora_ga_"):
        return "lora_ga"
    if run_name.startswith("loram"):
        return "loram"
    if run_name.startswith("vanilla") or "_vanilla_" in run_name:
        return "vanilla"
    if run_name.startswith("compose_"):
        # compose_<init>_<cl>_<seq>_<suffix>
        parts = run_name.split("_")
        if len(parts) >= 3:
            return f"{parts[1]}+{parts[2]}"
    return "unknown"


def _extract_alpha_from_run_name(run_name: str) -> Optional[int]:
    """Run names from alpha_sweep.sh end in `..._a{N}`. Older runs do not.

    Returns the parsed int if the suffix matches, else None — caller falls
    back to run_config.json's resolved.lora_alpha.
    """
    m = re.search(r"_a(\d+)$", run_name)
    return int(m.group(1)) if m else None


def _stage_ba_means(run_dir: Path) -> Dict[str, Optional[float]]:
    """Average ba_norms (init/final, raw/effective) across all stages of a run.

    Tolerates missing fields — returns None for any aggregate when no stage has
    that field.
    """
    stages_dir = run_dir / "stages"
    if not stages_dir.exists():
        return {k: None for k in ("init_raw", "init_eff", "final_raw", "final_eff")}

    sums = {"init_raw": [], "init_eff": [], "final_raw": [], "final_eff": []}
    for stage_dir in sorted(stages_dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        sr = _safe_load(stage_dir / "stage_record.json")
        if not sr:
            continue
        ba = (sr.get("train_report") or {}).get("ba_norms")
        if not ba:
            continue
        init = ba.get("init") or {}
        final = ba.get("final") or {}
        if isinstance(init.get("raw_mean"), (int, float)):
            sums["init_raw"].append(float(init["raw_mean"]))
        if isinstance(init.get("effective_mean"), (int, float)):
            sums["init_eff"].append(float(init["effective_mean"]))
        if isinstance(final.get("raw_mean"), (int, float)):
            sums["final_raw"].append(float(final["raw_mean"]))
        if isinstance(final.get("effective_mean"), (int, float)):
            sums["final_eff"].append(float(final["effective_mean"]))

    out: Dict[str, Optional[float]] = {}
    for k, vs in sums.items():
        out[k] = (sum(vs) / len(vs)) if vs else None
    out["num_stages_with_ba"] = len(sums["final_raw"])
    return out


def _resolved_alpha(run_dir: Path) -> Optional[int]:
    rc = _safe_load(run_dir / "run_config.json")
    if not rc:
        return None
    resolved = rc.get("resolved") or {}
    val = resolved.get("lora_alpha")
    if isinstance(val, (int, float)):
        return int(val)
    return None


def _resolved_rank(run_dir: Path) -> Optional[int]:
    rc = _safe_load(run_dir / "run_config.json")
    if not rc:
        return None
    resolved = rc.get("resolved") or {}
    val = resolved.get("rank")
    if isinstance(val, (int, float)):
        return int(val)
    return None


def collect(results_root: Path, run_glob: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for seq_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        for run_dir in sorted(seq_dir.glob(run_glob)):
            if not run_dir.is_dir():
                continue
            metrics = _safe_load(run_dir / "metrics.json")
            if not metrics:
                continue
            row: Dict[str, Any] = {
                "sequence": seq_dir.name,
                "run_name": run_dir.name,
                "method": _infer_method_from_run_name(run_dir.name),
            }
            alpha = _extract_alpha_from_run_name(run_dir.name)
            if alpha is None:
                alpha = _resolved_alpha(run_dir)
            row["lora_alpha"] = alpha
            row["rank"] = _resolved_rank(run_dir)
            for k in METRIC_KEYS:
                v = metrics.get(k)
                row[k] = float(v) if isinstance(v, (int, float)) else None
            row.update(_stage_ba_means(run_dir))
            rows.append(row)
    return rows


def _fmt(v: Any, width: int = 8, decimals: int = 4) -> str:
    if v is None:
        return f"{'--':>{width}}"
    if isinstance(v, float):
        return f"{v:>{width}.{decimals}f}"
    return f"{str(v):>{width}}"


def print_per_sequence(rows: List[Dict[str, Any]]) -> None:
    seqs = sorted({r["sequence"] for r in rows})
    for seq in seqs:
        seq_rows = [r for r in rows if r["sequence"] == seq]
        seq_rows.sort(key=lambda r: (r.get("AP") is None, -(r.get("AP") or 0.0)))
        print()
        print("=" * 138)
        print(f"  {seq}    (sorted by AP desc; alpha=rsLoRA alpha;  raw_mean = mean ||B@A||_F over layers, then over stages)")
        print("=" * 138)
        header = (
            f"  {'method':<14} {'alpha':>6} {'rank':>5} "
            f"{'AP':>8} {'FP':>8} {'GP':>8} {'IP':>8} {'Forget':>8}  "
            f"{'init_||BA||':>12} {'final_||BA||':>13}  {'run_name'}"
        )
        print(header)
        print("-" * 138)
        for r in seq_rows:
            print(
                f"  {r['method']:<14} {_fmt(r['lora_alpha'], 6, 0)} {_fmt(r['rank'], 5, 0)} "
                f"{_fmt(r.get('AP'))} {_fmt(r.get('FP'))} {_fmt(r.get('GP'))} {_fmt(r.get('IP'))} {_fmt(r.get('Forget'))}  "
                f"{_fmt(r.get('init_raw'), 12, 4)} {_fmt(r.get('final_raw'), 13, 4)}  {r['run_name']}"
            )


def print_scale_curve(rows: List[Dict[str, Any]]) -> None:
    """One scatter row per (method, sequence) sorted by final ||BA||_F.

    Reads as: 'at this scale, this method's AP / GP / Forget were …'.
    Use this as the source for AP-vs-||BA||_F figure 1.
    """
    keyed: Dict[tuple, List[Dict[str, Any]]] = {}
    for r in rows:
        keyed.setdefault((r["sequence"], r["method"]), []).append(r)

    print()
    print("=" * 130)
    print("  AP / GP / Forget vs final ||BA||_F  (one curve per method × sequence)")
    print("=" * 130)
    for (seq, method) in sorted(keyed.keys()):
        sub = sorted(keyed[(seq, method)], key=lambda r: (r.get("final_raw") is None, r.get("final_raw") or 0.0))
        print()
        print(f"  -- {seq} | {method} --")
        print(f"  {'alpha':>6} {'final_||BA||':>13} {'init_||BA||':>12}  {'AP':>8} {'GP':>8} {'Forget':>8}")
        for r in sub:
            print(
                f"  {_fmt(r['lora_alpha'], 6, 0)} "
                f"{_fmt(r.get('final_raw'), 13, 4)} {_fmt(r.get('init_raw'), 12, 4)}  "
                f"{_fmt(r.get('AP'))} {_fmt(r.get('GP'))} {_fmt(r.get('Forget'))}"
            )


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    if not rows:
        print(f"No rows — skipping {path}", file=sys.stderr)
        return
    cols = [
        "sequence", "method", "lora_alpha", "rank", "run_name",
        "AP", "FP", "GP", "IP", "Forget",
        "init_raw", "init_eff", "final_raw", "final_eff", "num_stages_with_ba",
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})
    print(f"Wrote {len(rows)} rows to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=Path("results"),
        help="Root of the results tree (contains <sequence>/<run_name>/).",
    )
    parser.add_argument(
        "--run-glob", default="*",
        help="Glob filter for run directory names within each sequence dir. "
             "Use e.g. '*alphasweep*' to scope to the sweep.",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Optional CSV output path (for plotting in pandas/matplotlib).",
    )
    parser.add_argument(
        "--no-scale-curve", action="store_true",
        help="Skip the per-method scale-curve printout.",
    )
    args = parser.parse_args()

    if not args.results_root.exists():
        print(f"Results root not found: {args.results_root}", file=sys.stderr)
        sys.exit(2)

    rows = collect(args.results_root, args.run_glob)
    if not rows:
        print(f"No runs matched glob '{args.run_glob}' under {args.results_root}", file=sys.stderr)
        sys.exit(1)

    print_per_sequence(rows)
    if not args.no_scale_curve:
        print_scale_curve(rows)

    if args.csv:
        write_csv(rows, args.csv)


if __name__ == "__main__":
    main()
