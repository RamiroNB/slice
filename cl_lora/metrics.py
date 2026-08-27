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
    stage_zero_record: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Build a stage-by-task score matrix for trained tasks.

    The stage-0 baseline (untrained base model, no adapters) is **appended**
    last, not prepended, and carries ``"stage": 0`` with an empty
    ``trained_task``. Analysis code across this repo indexes this list
    positionally — ``matrix[st]`` for stage ``st+1``, ``matrix[4]`` for the
    final stage of a 5-task sequence — so putting the baseline first would
    silently shift every one of those reads by one row. Appending keeps
    ``matrix[i] == training stage i+1`` intact; read the baseline by filtering
    on ``stage == 0``, never by position.
    """
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

    if stage_zero_record is not None:
        zero_scores = _stage_seen_scores(stage_zero_record)
        matrix.append(
            {
                "stage": 0,
                # Empty on purpose: consumers build the task axis from
                # trained_task, so a label here would invent a sixth task.
                "trained_task": "",
                "baseline": "base_model_no_adapters",
                "scores": {task: zero_scores.get(task) for task in task_order},
            }
        )

    return matrix


def compute_cl_metrics(
    stage_records: List[Dict[str, Any]],
    task_order: List[str],
    stage_zero_record: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Compute AP, FP, Forget, GP, IP from staged continual-learning results.

    Notes:
        - AP is the mean of diagonal task scores (right after each task is trained).
        - FP is the mean final-stage scores over all trained tasks.
        - Forget is AP - FP.
        - GP/IP are final-stage general means from lm-eval.
        - ``stage_zero_record`` is the untrained base model scored on every task
          in the sequence. It is reported as ZS (and per-task under
          ``zero_shot_scores``) but deliberately kept out of ``stage_records``
          and ``results_matrix``: those are indexed by training stage, so a
          stage-0 row would shift the diagonal AP reads off by one.
    """
    if not stage_records:
        raise ValueError("stage_records is empty.")
    if not task_order:
        raise ValueError("task_order is empty.")

    matrix = build_results_matrix(
        stage_records=stage_records,
        task_order=task_order,
        stage_zero_record=stage_zero_record,
    )
    # AP/FP are defined over training stages only. The stage-0 row lives in the
    # same list, so select by stage rather than by position/`[-1]`.
    train_rows = [row for row in matrix if row.get("stage", 0) > 0]

    diagonal_scores: Dict[str, float | None] = {}
    diag_values: List[float] = []
    for idx, task_name in enumerate(task_order):
        if idx >= len(train_rows):
            break
        score = train_rows[idx]["scores"].get(task_name)
        diagonal_scores[task_name] = score
        if score is not None:
            diag_values.append(score)

    final_stage_scores = train_rows[-1]["scores"]
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

    ap = _mean(diag_values)
    fp = _mean(final_values)
    metrics = {
        "AP": ap,
        "FP": fp,
        "Forget": (ap - fp) if (ap is not None and fp is not None) else None,
        "GP": final_general.get("gp_mean"),
        "IP": final_general.get("ip_mean"),
    }

    out: Dict[str, Any] = {
        "metrics": metrics,
        "task_order": task_order,
        "results_matrix": matrix,
        "diagonal_scores": diagonal_scores,
        "final_scores": final_stage_scores,
        "per_task_forgetting": per_task_forgetting,
    }

    if stage_zero_record is not None:
        zero_seen = _stage_seen_scores(stage_zero_record)
        zero_scores = {task: zero_seen.get(task) for task in task_order}
        zero_values = [v for v in zero_scores.values() if v is not None]
        zero_general = stage_zero_record.get("general") or {}
        metrics["ZS"] = _mean(zero_values)
        metrics["ZS_GP"] = zero_general.get("gp_mean")
        metrics["ZS_IP"] = zero_general.get("ip_mean")
        out["zero_shot_scores"] = zero_scores
        out["gain_over_zero_shot"] = {
            task: (
                (final_stage_scores.get(task) - zero_scores[task])
                if (final_stage_scores.get(task) is not None and zero_scores[task] is not None)
                else None
            )
            for task in task_order
        }

    return out
