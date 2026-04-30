"""
Sync finished CL-LoRA runs from remote machines and analyse results.

SSH aliases "cristian" and "rafa" must be defined in ~/.ssh/config.
Both route through sparta.pucrs.br as a jump host, so two passwords are

Remote layout:  /work/cl-lora/adaptors_eval/<seq_name>/<method>/
Local layout:   ./imported_results/<seq_name>/<method>/

A run is considered finished only when metrics.json is present.
Files synced:   metrics.json  results_matrix.json  run_config.json  run_summary.json

Usage examples
--------------
  # Sync from both machines then analyse
  python results_analysis.py --sync

  # Sync only, no analysis
  python results_analysis.py --sync --no-analyse

  # Analyse only (already synced)
  python results_analysis.py

  # Show bar charts and save them
  python results_analysis.py --plot --save-plots plots/
"""

import json
import os
import subprocess
import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Remote machine config
# ---------------------------------------------------------------------------

# SSH aliases as defined in ~/.ssh/config — no host/user needed here.
REMOTES = ["cristian", "rafa"]

REMOTE_BASE        = "~/work/cl-lora/adaptors_eval"
LOCAL_BASE         = Path("imported_results")
MOTOX_RESULTS      = Path("/mnt/E-SSD/dev-cl-lora/cl-lora/results")
PENDING_EVAL_REMOTES = ["cristian", "rafa"]   # machines running pending_eval
PENDING_EVAL_PATH    = "~/work/cl-lora/pending_eval"
PENDING_EVAL_LOCAL   = Path("imported_results_pending")
SYNC_FILES           = ["metrics.json", "results_matrix.json", "run_config.json", "run_summary.json"]
# stage_record.json files under stages/stage_*/ carry per-benchmark GP/IP breakdowns
SYNC_STAGE_RECORDS   = True

METRICS = ["AP", "FP", "GP", "IP", "Forget"]


# ---------------------------------------------------------------------------
# SSH / rsync helpers
# ---------------------------------------------------------------------------

def load_env(env_file: Path = Path(".env")) -> dict[str, str]:
    if not env_file.exists():
        raise FileNotFoundError(f".env not found at {env_file.resolve()}")
    result = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def _ctrl_socket(alias: str) -> str:
    return f"/tmp/ssh_ctrl_{alias}"


def open_control_master(alias: str, verbose: bool = False) -> bool:
    """
    Open a background SSH ControlMaster connection to alias.
    The user will be prompted for the password here — once — and all
    subsequent SSH/rsync calls reuse the socket without prompting again.
    Returns True if the connection was established.
    """
    ctrl = _ctrl_socket(alias)
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", f"ControlPath={ctrl}",
        "-o", "ControlMaster=yes",
        "-o", "ControlPersist=10m",
        *(["-v"] if verbose else []),
        "-f",   # go to background after auth
        "-N",   # no remote command — just keep the connection open
        alias,
    ]
    print(f"  $ {' '.join(cmd)}")
    print(f"  (type your password for sparta when prompted)")
    result = subprocess.run(cmd)  # stdin/stderr flow to terminal for password prompt
    if result.returncode != 0:
        print(f"  [error] ControlMaster for {alias!r} failed (exit {result.returncode})")
        return False
    print(f"  Connection open — socket: {ctrl}")
    return True


def close_control_master(alias: str) -> None:
    ctrl = _ctrl_socket(alias)
    subprocess.run(
        ["ssh", "-o", f"ControlPath={ctrl}", "-O", "exit", alias],
        capture_output=True,
    )


def list_finished_runs(alias: str, verbose: bool = False) -> list[tuple[str, str]]:
    """Return [(seq_name, method), ...] for finished runs, reusing the ControlMaster socket."""
    ctrl = _ctrl_socket(alias)
    # Use 'ls' first to confirm the base dir exists, then find
    remote_cmd = (
        f"echo '--- ls {REMOTE_BASE} ---' && ls {REMOTE_BASE} 2>&1 && "
        f"echo '--- find ---' && "
        f"find {REMOTE_BASE} -maxdepth 3 -name metrics.json; true"
    )
    cmd = [
        "ssh",
        "-o", "StrictHostKeyChecking=no",
        "-o", f"ControlPath={ctrl}",
        "-o", "ControlMaster=no",
        *(["-v"] if verbose else []),
        alias,
        remote_cmd,
    ]
    print(f"  $ ssh ... {alias} <remote_cmd>")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(f"  exit code: {result.returncode}")
    if result.stderr.strip():
        print(f"  stderr: {result.stderr.strip()}")
    print(f"  stdout:\n    " + "\n    ".join(result.stdout.splitlines()) if result.stdout.strip() else "  stdout: (empty)")

    runs = []
    for line in result.stdout.splitlines():
        if "metrics.json" not in line:
            continue
        parts = line.strip().split("/")
        try:
            idx = parts.index("adaptors_eval")
            runs.append((parts[idx + 1], parts[idx + 2]))
        except (ValueError, IndexError):
            pass
    return runs


def rsync_run(alias: str, seq_name: str, method: str, verbose: bool = False) -> bool:
    """Rsync the result files for one finished run, reusing the ControlMaster socket."""
    remote_dir = f"{REMOTE_BASE}/{seq_name}/{method}/"
    local_dir  = LOCAL_BASE / seq_name / method
    local_dir.mkdir(parents=True, exist_ok=True)

    ctrl = _ctrl_socket(alias)
    include_args = []
    for f in SYNC_FILES:
        include_args += ["--include", f]

    if SYNC_STAGE_RECORDS:
        # Pull stages/stage_**/stage_record.json for per-benchmark GP/IP breakdown
        include_args += [
            "--include", "stages/",
            "--include", "stages/stage_*/",
            "--include", "stages/stage_*/stage_record.json",
        ]

    ssh_opt = (
        f"ssh -o StrictHostKeyChecking=no"
        f" -o ControlPath={ctrl} -o ControlMaster=no"
        + (" -v" if verbose else "")
    )
    cmd = [
        "rsync", "-az", "--progress",
        *(["-v"] if verbose else []),
        "-e", ssh_opt,
        "--include", "*/",
        *include_args,
        "--exclude", "*",
        f"{alias}:{remote_dir}",
        str(local_dir) + "/",
    ]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [error] rsync exited with code {result.returncode}")
        return False
    return True


def _sync_normal_eval(alias: str, verbose: bool = False) -> None:
    """Sync finished runs from adaptors_eval/ on alias. ControlMaster must already be open."""
    print(f"  [normal eval] looking for finished runs in {REMOTE_BASE} ...")
    runs = list_finished_runs(alias, verbose=verbose)
    if not runs:
        print("  No finished runs found.")
        return
    print(f"  Found {len(runs)} finished run(s): {[f'{s}/{m}' for s, m in runs]}")
    for seq_name, method in sorted(runs):
        local_run = LOCAL_BASE / seq_name / method
        local_metrics = local_run / "metrics.json"
        has_stage_records = bool(list((local_run / "stages").glob("stage_*/stage_record.json"))) \
            if (local_run / "stages").exists() else False
        if local_metrics.exists() and has_stage_records:
            print(f"  [already synced] {seq_name}/{method}")
            continue
        if local_metrics.exists():
            print(f"  [re-syncing — missing stage_record.json] {seq_name}/{method}")
        print(f"\n  Syncing {seq_name}/{method} ...")
        rsync_run(alias, seq_name, method, verbose=verbose)


def _sync_pending_eval(alias: str, verbose: bool = False) -> None:
    """Sync pending_eval result JSONs from alias. ControlMaster must already be open."""
    print(f"  [pending eval] syncing result JSONs from {PENDING_EVAL_PATH} ...")
    PENDING_EVAL_LOCAL.mkdir(parents=True, exist_ok=True)
    ctrl = _ctrl_socket(alias)
    ssh_opt = (
        f"ssh -o StrictHostKeyChecking=no"
        f" -o ControlPath={ctrl} -o ControlMaster=no"
        + (" -v" if verbose else "")
    )
    include_args = []
    for f in SYNC_FILES:
        include_args += ["--include", f]

    if SYNC_STAGE_RECORDS:
        include_args += [
            "--include", "stages/stage_*/stage_record.json",
        ]

    cmd = [
        "rsync", "-az",
        "--include", "*/",
        *include_args,
        "--exclude", "*",
        "-e", ssh_opt,
        f"{alias}:{PENDING_EVAL_PATH}/",
        str(PENDING_EVAL_LOCAL) + "/",
    ]
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"  [error] rsync exited with code {result.returncode}")
    else:
        print(f"  Done → {PENDING_EVAL_LOCAL}/")


def sync_all(verbose: bool = False) -> None:
    print("\n=== Syncing from remote machines ===")
    all_aliases = sorted(set(REMOTES) | set(PENDING_EVAL_REMOTES))
    for alias in all_aliases:
        print(f"\n[{alias}] opening connection (you will be prompted for the sparta password) ...")
        if not open_control_master(alias, verbose=verbose):
            continue
        try:
            if alias in REMOTES:
                _sync_normal_eval(alias, verbose=verbose)
            if alias in PENDING_EVAL_REMOTES:
                _sync_pending_eval(alias, verbose=verbose)
        finally:
            close_control_master(alias)


# ---------------------------------------------------------------------------
# Local loading
# ---------------------------------------------------------------------------

def _load_benchmark_scores(run_dir: Path) -> dict:
    """Load per-benchmark GP/IP scores from the final stage_record.json, if present.

    Returns a dict with keys 'gp' and 'ip', each a {benchmark: score} dict,
    or empty dicts if the data isn't available.
    """
    stages_dir = run_dir / "stages"
    if not stages_dir.exists():
        return {"gp": {}, "ip": {}}

    stage_dirs = sorted(stages_dir.glob("stage_*"))
    if not stage_dirs:
        return {"gp": {}, "ip": {}}

    # Use the last stage — that's where the final GP/IP eval runs
    record_path = stage_dirs[-1] / "stage_record.json"
    if not record_path.exists():
        return {"gp": {}, "ip": {}}

    try:
        with record_path.open() as f:
            record = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"gp": {}, "ip": {}}

    general = record.get("general", {})
    return {
        "gp": general.get("gp") or {},
        "ip": general.get("ip") or {},
    }


BBH_GP_EXCLUDE = {"bbh_object_counting"}


def _gp_no_bbh(benchmarks: dict, stored_gp: float | None) -> float | None:
    """Return GP recomputed excluding BBH zero-shot scores.

    Requires per-benchmark data from stage_record.json to compute exactly.
    Falls back to the stored aggregate unchanged when that data isn't available
    (approximating would introduce real error since BBH GP is not always 0).
    """
    gp = benchmarks.get("gp", {})
    valid = {k: v for k, v in gp.items() if k not in BBH_GP_EXCLUDE and v is not None}
    if valid:
        return sum(valid.values()) / len(valid)
    return stored_gp


def load_run(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None

    with metrics_path.open() as f:
        metrics = json.load(f)

    config = {}
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with config_path.open() as f:
            raw = json.load(f)
        config = raw.get("orchestrator", {}).get("cli_args", raw)

    matrix = []
    matrix_path = run_dir / "results_matrix.json"
    if matrix_path.exists():
        with matrix_path.open() as f:
            matrix = json.load(f)

    benchmarks = _load_benchmark_scores(run_dir)
    metrics["GP"] = _gp_no_bbh(benchmarks, metrics.get("GP"))

    return {
        "seq_name": config.get("sequence") or run_dir.parent.name,
        "method": run_dir.name,
        "metrics": metrics,
        "config": config,
        "matrix": matrix,
        "benchmarks": benchmarks,
    }


def _run_completeness(run: dict) -> int:
    """Count non-None metric values — higher is more complete."""
    return sum(1 for m in METRICS if run["metrics"].get(m) is not None)


def collect_runs(*roots: Path) -> list[dict]:
    best: dict[tuple[str, str], dict] = {}

    def _consider(run: dict) -> None:
        key = (run["seq_name"], run["method"])
        if key not in best or _run_completeness(run) > _run_completeness(best[key]):
            best[key] = run

    for root in roots:
        if not root.exists():
            continue
        for seq_dir in sorted(root.iterdir()):
            if not seq_dir.is_dir():
                continue
            if seq_dir.name.startswith("incomplete_"):
                continue
            # Flat layout: seq_dir IS the run directory
            if (seq_dir / "metrics.json").exists():
                run = load_run(seq_dir)
                if run is not None:
                    _consider(run)
                continue
            # Standard 2-level layout: <root>/<seq_name>/<method>/
            for method_dir in sorted(seq_dir.iterdir()):
                if not method_dir.is_dir():
                    continue
                if method_dir.name.startswith("incomplete_"):
                    continue
                run = load_run(method_dir)
                if run is None:
                    print(f"  [skip – not finished] {method_dir.relative_to(root)}")
                    continue
                _consider(run)

    return list(best.values())


# ---------------------------------------------------------------------------
# DataFrame building
# ---------------------------------------------------------------------------

def build_metrics_df(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        row = {"seq_name": r["seq_name"], "method": r["method"]}
        for m in METRICS:
            row[m] = r["metrics"].get(m)
        rows.append(row)
    return pd.DataFrame(rows).set_index(["seq_name", "method"])


def build_matrix_df(runs: list[dict]) -> pd.DataFrame:
    rows = []
    for r in runs:
        for entry in r["matrix"]:
            stage   = entry["stage"]
            trained = entry["trained_task"]
            for task, score in entry["scores"].items():
                rows.append({
                    "seq_name":    r["seq_name"],
                    "method":      r["method"],
                    "stage":       stage,
                    "trained_task": trained,
                    "eval_task":   task,
                    "score":       score,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _strip_common_suffix(names: list[str]) -> list[str]:
    """Remove the longest common suffix shared by all names (split on '_')."""
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


def print_per_sequence_tables(df: pd.DataFrame) -> None:
    metrics = [m for m in METRICS if m in df.columns]
    higher_is_better = {"AP", "FP", "GP", "IP"}   # Forget: lower is better

    for seq, grp in df.groupby(level="seq_name"):
        sub = grp.droplevel("seq_name").copy()
        sub = sub[metrics].astype(float)

        # Split complete vs incomplete (any None metric = incomplete)
        complete_mask = sub.notna().all(axis=1)
        complete   = sub[complete_mask].copy()
        incomplete = sub[~complete_mask].copy()

        # Sort complete by AP descending
        if "AP" in complete.columns and not complete.empty:
            complete = complete.sort_values("AP", ascending=False)
        if "AP" in incomplete.columns and not incomplete.empty:
            incomplete = incomplete.sort_values("AP", ascending=False)

        all_rows = pd.concat([complete, incomplete])
        short_names = _strip_common_suffix(list(all_rows.index))
        name_map = dict(zip(all_rows.index, short_names))

        # Build display strings — stars only from complete runs
        display = all_rows.copy().astype(object)
        for col in metrics:
            col_vals = complete[col].dropna() if not complete.empty else pd.Series(dtype=float)
            best = col_vals.max() if col in higher_is_better else (col_vals.min() if not col_vals.empty else None)
            for idx in all_rows.index:
                v = all_rows.loc[idx, col]
                if pd.isna(v):
                    display.loc[idx, col] = "  --  "
                else:
                    is_best = best is not None and v == best and idx in complete.index
                    display.loc[idx, col] = f"{v:.4f}" + ("*" if is_best else " ")

        display.index = [name_map[i] for i in all_rows.index]

        # Mark incomplete rows with a trailing label
        if not incomplete.empty:
            incomplete_short = {name_map[i] for i in incomplete.index}
            new_index = [
                f"{n}  [incomplete]" if n in incomplete_short else n
                for n in display.index
            ]
            display.index = new_index

        sep = "=" * (max(len(n) for n in display.index) + len(metrics) * 9 + 4)
        print(f"\n{sep}")
        print(f"  {seq}")
        print(f"  (* = best per metric, complete runs only | AP FP GP IP: higher is better | Forget: lower is better)")
        print(sep)
        print(display.to_string())
        print(sep)


# ---------------------------------------------------------------------------
# Benchmark table (per-benchmark GP/IP breakdown, --benchmarks flag)
# ---------------------------------------------------------------------------

BENCHMARK_SHORT = {
    "hellaswag":         "Hella.",
    "commonsenseqa":     "Com.",
    "alpaca":            "Alpa.",
    "bbh_object_counting": "BBH-Ob.",
    "openbookqa":        "OB-QA",
    "lambada":           "Lambada",
}


def print_benchmark_tables(runs: list[dict]) -> None:
    """Print per-benchmark GP/IP tables grouped by sequence, paper-style."""
    from collections import defaultdict

    by_seq: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_seq[r["seq_name"]].append(r)

    for seq, seq_runs in sorted(by_seq.items()):
        # Collect all benchmark keys that appear across any run
        gp_keys: list[str] = []
        ip_keys: list[str] = []
        for r in seq_runs:
            b = r.get("benchmarks", {})
            for k in b.get("gp", {}):
                if k not in gp_keys:
                    gp_keys.append(k)
            for k in b.get("ip", {}):
                if k not in ip_keys:
                    ip_keys.append(k)

        # Prefer the canonical order; append any extras
        canonical = ["hellaswag", "commonsenseqa", "alpaca", "bbh_object_counting",
                     "openbookqa", "lambada"]
        gp_keys = [k for k in canonical if k in gp_keys] + [k for k in gp_keys if k not in canonical]
        ip_keys = [k for k in canonical if k in ip_keys] + [k for k in ip_keys if k not in canonical]

        if not gp_keys and not ip_keys:
            print(f"\n[{seq}] No per-benchmark data — sync stage_record.json files first (run --sync).")
            continue

        # Column headers
        gp_cols = [BENCHMARK_SHORT.get(k, k) for k in gp_keys]
        ip_cols = [BENCHMARK_SHORT.get(k, k) for k in ip_keys]
        cl_cols = ["AP", "FP", "Forget"]

        header_gp = "  ".join(f"{c:>8}" for c in gp_cols)
        header_ip = "  ".join(f"{c:>8}" for c in ip_cols)
        header_cl = "  ".join(f"{c:>8}" for c in cl_cols)

        short_names = _strip_common_suffix([r["method"] for r in seq_runs])
        name_width = max(len(n) for n in short_names)

        sep = "=" * (name_width + 4 + len(header_gp) + 4 + len(header_ip) + 4 + len(header_cl) + 4)
        print(f"\n{sep}")
        print(f"  {seq}  — benchmark breakdown (GP zero-shot | IP in-context | CL metrics)")
        print(f"  (stage_record.json from final stage)")
        print(sep)

        # Header row
        gp_label  = "Zero-shot GP".center(len(header_gp))
        ip_label  = "In-context IP".center(len(header_ip))
        cl_label  = "CL".center(len(header_cl))
        print(f"  {'Method'.ljust(name_width)}  {gp_label}  {ip_label}  {cl_label}")
        print(f"  {' ' * name_width}  {header_gp}  {header_ip}  {header_cl}")
        print("-" * len(sep))

        for r, short in zip(seq_runs, short_names):
            b   = r.get("benchmarks", {})
            gp  = b.get("gp", {})
            ip  = b.get("ip", {})
            m   = r["metrics"]

            def _fmt(v) -> str:
                return f"{v * 100:8.2f}" if v is not None else "      --"

            gp_vals = "  ".join(_fmt(gp.get(k)) for k in gp_keys)
            ip_vals = "  ".join(_fmt(ip.get(k)) for k in ip_keys)

            ap  = m.get("AP")
            fp  = m.get("FP")
            fgt = m.get("Forget")
            cl_vals = "  ".join([_fmt(ap), _fmt(fp), _fmt(fgt)])

            print(f"  {short.ljust(name_width)}  {gp_vals}  {ip_vals}  {cl_vals}")

        print(sep)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_metrics_bar(df: pd.DataFrame, save_dir: Path | None = None) -> None:
    metrics = [m for m in METRICS if m in df.columns]
    for seq in df.index.get_level_values("seq_name").unique():
        sub = df.xs(seq, level="seq_name")[metrics].astype(float)
        ax = sub.plot(kind="bar", figsize=(max(6, len(sub) * 1.2), 4), rot=30)
        ax.set_title(f"CL Metrics — {seq}")
        ax.set_xlabel("Method")
        ax.set_ylabel("Score")
        ax.legend(loc="upper right")
        plt.tight_layout()
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            path = save_dir / f"metrics_{seq}.png"
            plt.savefig(path, dpi=150)
            print(f"  saved {path}")
        else:
            plt.show()
        plt.close()


def plot_results_matrix(matrix_df: pd.DataFrame, seq_name: str, method: str,
                        save_dir: Path | None = None) -> None:
    sub = matrix_df[(matrix_df.seq_name == seq_name) & (matrix_df.method == method)]
    if sub.empty:
        return

    pivot = sub.pivot(index="trained_task", columns="eval_task", values="score").astype(float)
    fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns)), max(4, len(pivot))))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(f"Results Matrix — {seq_name} / {method}")
    ax.set_xlabel("Eval task")
    ax.set_ylabel("After training on")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)
    plt.tight_layout()
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"matrix_{seq_name}_{method}.png"
        plt.savefig(path, dpi=150)
        print(f"  saved {path}")
    else:
        plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Sync and analyse CL-LoRA experiment results.")
    p.add_argument("--sync", action="store_true",
                   help="SSH into remote machines and rsync finished runs first")
    p.add_argument("--no-analyse", action="store_true",
                   help="Skip analysis (useful with --sync to only fetch data)")
    p.add_argument("--roots", nargs="+",
                   default=["imported_results", str(PENDING_EVAL_LOCAL), str(MOTOX_RESULTS)],
                   help="Local folders to scan for results")
    p.add_argument("--seq", nargs="+", default=None,
                   help="Filter to specific sequence names")
    p.add_argument("--method", nargs="+", default=None,
                   help="Filter to specific method names")
    p.add_argument("--plot", action="store_true",
                   help="Show / save bar charts of metrics")
    p.add_argument("--plot-matrix", action="store_true",
                   help="Show / save results-matrix heatmaps")
    p.add_argument("--save-plots", type=Path, default=None,
                   help="Directory to save plots (instead of opening windows)")
    p.add_argument("--export-csv", type=Path, default=None,
                   help="Export metrics DataFrame to CSV")
    p.add_argument("--benchmarks", action="store_true",
                   help="Print per-benchmark GP/IP breakdown table (requires stage_record.json in stages/)")
    p.add_argument("--verbose", action="store_true",
                   help="Pass -v to ssh/rsync for detailed connection logs")
    return p.parse_args()


def main():
    args = parse_args()

    if args.sync:
        sync_all(verbose=args.verbose)

    if args.no_analyse:
        return

    roots = [Path(r) for r in args.roots]
    print(f"\nScanning local roots: {[str(r) for r in roots]}")
    runs = collect_runs(*roots)

    if not runs:
        print("No finished runs found locally.")
        return

    if args.seq:
        runs = [r for r in runs if r["seq_name"] in args.seq]
    if args.method:
        runs = [r for r in runs if r["method"] in args.method]

    metrics_df = build_metrics_df(runs)
    matrix_df  = build_matrix_df(runs)

    print_per_sequence_tables(metrics_df)

    if args.benchmarks:
        print_benchmark_tables(runs)

    if args.export_csv:
        metrics_df.to_csv(args.export_csv)
        print(f"\nExported metrics to {args.export_csv}")

    if args.plot:
        plot_metrics_bar(metrics_df, save_dir=args.save_plots)

    if args.plot_matrix:
        for r in runs:
            plot_results_matrix(matrix_df, r["seq_name"], r["method"],
                                save_dir=args.save_plots)


if __name__ == "__main__":
    main()
