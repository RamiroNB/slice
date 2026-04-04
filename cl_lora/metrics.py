from __future__ import annotations

from typing import Any, Dict, List


def _mean(values: List[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _stage_seen_scores(stage_record: Dict[str, Any]) -> Dict[str, float | None]:
    seen = stage_record.get("seen_tasks")
    if seen is None:
        seen = stage_record.get("evaluation", {}).get("seen_tasks", {})

    out: Dict[str, float | None] = {}
    for task_name, payload in seen.items():
        if isinstance(payload, dict):
            out[task_name] = payload.get("score")
        elif isinstance(payload, (int, float)):
            out[task_name] = float(payload)
        else:
            out[task_name] = None
    return out


def build_results_matrix(
    stage_records: List[Dict[str, Any]],
    task_order: List[str],
) -> List[Dict[str, Any]]:
    """Build a stage-by-task score matrix for trained tasks."""
    matrix: List[Dict[str, Any]] = []
    for stage_idx, stage in enumerate(stage_records, start=1):
        seen_scores = _stage_seen_scores(stage)
        row_scores = {task: seen_scores.get(task) for task in task_order}
        matrix.append(
            {
                "stage": stage_idx,
                "trained_task": stage.get("trained_task", task_order[stage_idx - 1]),
                "scores": row_scores,
            }
        )
    return matrix


def compute_cl_metrics(
    stage_records: List[Dict[str, Any]],
    task_order: List[str],
) -> Dict[str, Any]:
    """Compute AP, FP, Forget, GP, IP from staged continual-learning results.

    Notes:
        - AP is the mean of diagonal task scores (right after each task is trained).
        - FP is the mean final-stage scores over all trained tasks.
        - Forget is AP - FP.
        - GP/IP are final-stage general means from lm-eval.
    """
    if not stage_records:
        raise ValueError("stage_records is empty.")
    if not task_order:
        raise ValueError("task_order is empty.")

    matrix = build_results_matrix(stage_records=stage_records, task_order=task_order)

    diagonal_scores: Dict[str, float | None] = {}
    diag_values: List[float] = []
    for idx, task_name in enumerate(task_order):
        if idx >= len(matrix):
            break
        score = matrix[idx]["scores"].get(task_name)
        diagonal_scores[task_name] = score
        if score is not None:
            diag_values.append(score)

    final_stage_scores = matrix[-1]["scores"]
    final_values = [v for v in final_stage_scores.values() if v is not None]

    per_task_forgetting = {}
    for task_name in task_order:
        diag = diagonal_scores.get(task_name)
        final = final_stage_scores.get(task_name)
        per_task_forgetting[task_name] = (
            (diag - final) if (diag is not None and final is not None) else None
        )

    final_general = stage_records[-1].get("general")
    if final_general is None:
        final_general = stage_records[-1].get("evaluation", {}).get("general", {})

    metrics = {
        "AP": _mean(diag_values),
        "FP": _mean(final_values),
        "Forget": (
            (_mean(diag_values) - _mean(final_values))
            if (_mean(diag_values) is not None and _mean(final_values) is not None)
            else None
        ),
        "GP": final_general.get("gp_mean"),
        "IP": final_general.get("ip_mean"),
    }

    return {
        "metrics": metrics,
        "task_order": task_order,
        "results_matrix": matrix,
        "diagonal_scores": diagonal_scores,
        "final_scores": final_stage_scores,
        "per_task_forgetting": per_task_forgetting,
    }
