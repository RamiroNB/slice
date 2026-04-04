# recompute_metrics.py
import json
from pathlib import Path


def _extract_primary_metric(task_result):
    preferred = [
        "acc_norm,none",
        "acc,none",
        "exact_match,none",
        "exact_match,get-answer",  # BBH fix
        "f1,none",
        "rougeL,none",
        "bleu,none",
    ]
    for key in preferred:
        if key in task_result:
            return float(task_result[key])

    for key, value in task_result.items():
        if isinstance(value, float) and (
            key.startswith("exact_match,") or key.startswith("acc,")
        ):
            return float(value)

    for key, value in task_result.items():
        if isinstance(value, float) and "stderr" not in key:
            return float(value)

    return None


def _mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


run_dir = Path("results/NI-Seq-Dummy/dummy_dev01")

# Re-extract GP/IP for every stage from raw lm-eval dicts
for stage_dir in sorted((run_dir / "stages").iterdir()):
    record_path = stage_dir / "stage_record.json"
    if not record_path.exists():
        continue

    data = json.loads(record_path.read_text())
    general = data.get("general", {})
    raw = general.get("raw", {})

    if not raw:
        print(f"No raw data in {stage_dir.name}, skipping")
        continue

    # Re-extract GP
    gp_scores = {
        task: _extract_primary_metric(result)
        for task, result in raw.get("gp", {}).items()
    }
    # Re-extract IP
    ip_scores = {
        task: _extract_primary_metric(result)
        for task, result in raw.get("ip", {}).items()
    }

    # Preserve alpaca (already a scalar in gp/ip, not in raw lm-eval dict)
    # since alpaca goes through our own evaluator, not lm-eval
    if "alpaca" in general.get("gp", {}):
        gp_scores["alpaca"] = general["gp"]["alpaca"]
        ip_scores["alpaca"] = general["ip"]["alpaca"]

    # Patch the stage record in memory
    general["gp"] = gp_scores
    general["ip"] = ip_scores
    general["gp_mean"] = _mean(gp_scores.values())
    general["ip_mean"] = _mean(ip_scores.values())

    # Write back
    data["general"] = general
    record_path.write_text(json.dumps(data, indent=2))
    print(f"Patched {stage_dir.name}")
    print(f"  GP: { {k: round(v,4) for k,v in gp_scores.items() if v is not None} }")
    print(f"  IP: { {k: round(v,4) for k,v in ip_scores.items() if v is not None} }")
    print(f"  gp_mean: {general['gp_mean']:.4f}")
    print(f"  ip_mean: {general['ip_mean']:.4f}")

print("\nNow rerunning metrics.py to recompute AP/FP/Forget/GP/IP...")