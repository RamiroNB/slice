"""
find_conflicting_seq.py

Measure gradient OPPOSITION (not just difference) across CL tasks.

SLICE's projection only fires when dot(g_forget, g_retain) < 0 — i.e. the
gradients point in genuinely opposite directions. Same-domain / same-output
tasks tend to be ALIGNED (cos > 0), and diverse tasks tend to be ORTHOGONAL
(cos ≈ 0); neither case exercises the projection. This module finds which
task pairs in the repo actually produce negative cosines, on the base model.

Registers a few candidate sequences at import time (SEQUENCES is mutated).

Two CLI modes:

    # Analyze one registered sequence pairwise:
    python -m cl_lora.find_conflicting_seq --sequence NI-Seq-Conflict-Rating

    # Search all pairs in a pool of tasks, rank most-opposite first:
    python -m cl_lora.find_conflicting_seq --search
    python -m cl_lora.find_conflicting_seq --search --pool NI363,NI195,NI618,NI589
"""
from __future__ import annotations

import warnings

# Suppress the "found in sys.modules after import of package" warning that
# runpy emits when `python -m cl_lora.find_conflicting_seq` is used and
# __init__.py has already imported this module. Must be at module level so
# it fires before runpy checks sys.modules.
warnings.filterwarnings(
    "ignore",
    message=".*found in sys.modules after import of package.*",
    category=RuntimeWarning,
    module="runpy",
)

import argparse
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

if TYPE_CHECKING:
    import torch

from .task_sequences import (
    SEQUENCES,
    Sequence,
    SuperNITask,
    get_sequence,
    # classification tasks
    NI195,
    NI363,
    NI1292,
    NI1310,
    NI1343,
    # generation tasks
    NI024,
    NI511,
    NI589,
    NI618,
    NI1290,
    NI1357,
)

logger = logging.getLogger("cl_lora.find_conflicting_seq")


# ---------------------------------------------------------------------------
# Candidate sequences (hypotheses, to be validated empirically via --search)
#
# NOTE: "same-domain, same-task" sequences like "3 rating tasks in a row"
# produce ALIGNED gradients (cos > 0) — SLICE's projection does NOTHING on
# aligned gradients. For SLICE to fire you need OPPOSITE gradients (cos < 0),
# which typically requires cross-modality (classification <-> generation) or
# contradictory supervision on similar inputs.
# ---------------------------------------------------------------------------

# Cross-modality: classify vs generate on review-ish text. Hypothesis: the
# output-mode difference forces anti-aligned components on the decoder side.
NI_SEQ_CROSSMODE_REVIEWS = Sequence(
    name="NI-Seq-CrossMode-Reviews",
    task_type="mixed",
    tasks=[NI363, NI618, NI1310, NI589],
    description=(
        "Cross-modality on review text: SST2(classify) -> AmazonRev(generate) -> "
        "AmazonRating(classify) -> AmazonFood(generate). Output mode flips each stage."
    ),
)


# Cross-modality: binary classify -> QA generation.
NI_SEQ_CROSSMODE_QA = Sequence(
    name="NI-Seq-CrossMode-QA",
    task_type="mixed",
    tasks=[NI195, NI024, NI363, NI618],
    description=(
        "Classify -> QA-gen -> classify -> summarize. Forces repeated head swap."
    ),
)


# CONTROL GROUP — same-task-type runs, expected to be ALIGNED (cos > 0).
# SLICE should collapse into LoRA-GA on these. Included so --search can
# confirm the alignment prediction empirically.
NI_SEQ_CONTROL_RATING = Sequence(
    name="NI-Seq-Control-Rating",
    task_type="classification",
    tasks=[NI1343, NI1310, NI1292],
    description="Control: 3 rating tasks, expected cos>0 (SLICE inert).",
)


NI_SEQ_CONTROL_SUMM = Sequence(
    name="NI-Seq-Control-Summ",
    task_type="generation",
    tasks=[NI618, NI589, NI1290, NI511, NI1357],
    description="Control: 5 summarisation tasks, expected cos>0 (SLICE inert).",
)


_NEW_SEQUENCES = [
    NI_SEQ_CROSSMODE_REVIEWS,
    NI_SEQ_CROSSMODE_QA,
    NI_SEQ_CONTROL_RATING,
    NI_SEQ_CONTROL_SUMM,
]

_OUR_SEQUENCE_NAMES = {s.name for s in _NEW_SEQUENCES}
for _seq in _NEW_SEQUENCES:
    existing = SEQUENCES.get(_seq.name)
    if existing is not None and existing.name not in _OUR_SEQUENCE_NAMES:
        # A different module/user already registered this name; refuse to clobber.
        raise RuntimeError(
            f"Sequence name collision: {_seq.name!r} already registered "
            "in task_sequences.SEQUENCES by another module."
        )
    # Idempotent: overwriting our own entry is fine (e.g. double-import via
    # `python -m cl_lora.find_conflicting_seq` which loads the module twice).
    SEQUENCES[_seq.name] = _seq


# Default pool for --search: covers binary classify, rating classify,
# summarisation, and QA generation. Only uses known-good SuperNI IDs.
DEFAULT_SEARCH_POOL: List[SuperNITask] = [
    NI195,   # Sentiment140 - binary classification (tweets)
    NI363,   # SST2 - binary classification (movie reviews)
    NI1310,  # Amazon multi-rating - 5-star classification
    NI618,   # Amazon review - summary generation
    NI589,   # Amazon food review - summary generation
    NI1290,  # XSum - summary generation
    NI024,   # CosmosQA - QA generation
]


# ---------------------------------------------------------------------------
# Gradient conflict measurement
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Disk-cache helpers — store one task's gradient as fp16 on disk so we never
# need to hold more than one (or two, during scoring) gradient dicts in VRAM.
# ---------------------------------------------------------------------------

def _cache_path(cache_dir: str, task_name: str) -> str:
    import os
    safe = task_name.replace("/", "_").replace(" ", "_")
    return os.path.join(cache_dir, f"grad_{safe}.pt")


def _save_grad_cache(task_name: str, grads: Dict[str, Any], cache_dir: str) -> None:
    """Save gradient dict to disk in fp16 (halves disk/read cost vs fp32)."""
    import os
    import torch
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, task_name)
    torch.save({k: v.half().cpu() for k, v in grads.items()}, path)
    logger.info("  cached → %s", path)


def _load_grad_cache(task_name: str, cache_dir: str) -> Dict[str, Any]:
    """Load a cached gradient dict back onto GPU as fp16."""
    import torch
    path = _cache_path(cache_dir, task_name)
    return torch.load(path, map_location="cuda", weights_only=True)


# ---------------------------------------------------------------------------
# CountSketch compression — reduces each 5.6 GB gradient file to ~800 KB
# while preserving dot products (and thus cosine similarity).
#
# How it works: each gradient element g[i] is mapped to a random bucket
# b[i] in {0..k-1} with a random sign s[i] in {±1}. The sketch accumulates
# sketch[b[i]] += s[i] * g[i]. Then E[dot(sketch_A, sketch_B)] = dot(g_A, g_B)
# exactly (unbiased). Std of the estimator ≈ ||g_A|| ||g_B|| / sqrt(k).
# With k=200k, error ≈ 0.2% — plenty for ranking 325 pairs.
#
# The bucket/sign assignments are seeded by module name so they are identical
# across all tasks → dot products are correctly estimated pairwise.
# ---------------------------------------------------------------------------

_SKETCH_K = 200_000  # sketch dimension; 200k × 4 bytes = 800 KB per task


def _sketch_path(sketch_dir: str, task_name: str) -> str:
    import os
    safe = task_name.replace("/", "_").replace(" ", "_")
    return os.path.join(sketch_dir, f"sketch_{safe}.pt")


def _module_seed(module_name: str, base_seed: int = 42) -> int:
    """Deterministic seed from module name (stable across Python runs)."""
    import hashlib
    h = int(hashlib.sha256(module_name.encode()).hexdigest()[:16], 16)
    return (base_seed ^ h) & 0x7FFFFFFF


def _build_global_sketch(
    grads: Dict[str, Any],
    k: int = _SKETCH_K,
    seed: int = 42,
) -> Tuple[Any, float, Dict[str, float]]:
    """CountSketch of the concatenated gradient across all modules.

    Returns (sketch_cpu, global_norm, {module: norm}).
    The sketch is fp32 on CPU (~800 KB for k=200k).
    """
    import torch
    sketch = torch.zeros(k, device="cuda")
    global_norm_sq = 0.0
    module_norms: Dict[str, float] = {}

    for name, g in grads.items():
        g_flat = g.float().view(-1).cuda()
        d_m = g_flat.numel()
        ms = _module_seed(name, seed)

        gen_idx = torch.Generator(device="cuda").manual_seed(ms)
        gen_sgn = torch.Generator(device="cuda").manual_seed(ms ^ 0xDEADBEEF)
        indices = torch.randint(0, k, (d_m,), generator=gen_idx, device="cuda")
        signs = torch.randint(0, 2, (d_m,), generator=gen_sgn, device="cuda").float().mul_(2).sub_(1)

        sketch.scatter_add_(0, indices, g_flat * signs)
        nm = float(g_flat.norm().item())
        module_norms[name] = nm
        global_norm_sq += nm * nm
        del g_flat, indices, signs
        torch.cuda.empty_cache()

    return sketch.cpu(), global_norm_sq ** 0.5, module_norms


def _save_sketch_cache(task_name: str, sketch: Any, global_norm: float,
                        module_norms: Dict[str, float], sketch_dir: str) -> None:
    import os, torch
    os.makedirs(sketch_dir, exist_ok=True)
    torch.save({"sketch": sketch, "global_norm": global_norm,
                "module_norms": module_norms}, _sketch_path(sketch_dir, task_name))


def _load_sketch_cache(task_name: str, sketch_dir: str) -> Dict[str, Any]:
    import torch
    return torch.load(_sketch_path(sketch_dir, task_name), weights_only=True)


def pair_conflict_from_sketch(
    sa: Dict[str, Any],
    sb: Dict[str, Any],
) -> Dict[str, float]:
    """Approximate global cosine from two CountSketches. Fast O(k) dot product."""
    import torch
    dot = float(torch.dot(sa["sketch"].float(), sb["sketch"].float()).item())
    norm_a = sa["global_norm"]
    norm_b = sb["global_norm"]
    return {"global_cosine": dot / (norm_a * norm_b + 1e-12)}


def compress_grad_cache(cache_dir: str, sketch_dir: str,
                         k: int = _SKETCH_K, seed: int = 42) -> List[str]:
    """Load each gradient file once, build a CountSketch, save to sketch_dir.

    This is the one-time compression step. After it runs, pairwise scoring
    reads 800 KB sketches instead of 5.6 GB gradient files → ~10000× less IO.
    No gradient recomputation needed.
    """
    import glob, os, torch
    from tqdm import tqdm

    cache_files = sorted(glob.glob(os.path.join(cache_dir, "grad_*.pt")))
    logger.info(
        "Compressing %d gradient files → CountSketches (k=%d, ~%.0f KB each)",
        len(cache_files), k, k * 4 / 1024,
    )
    task_names = []
    for path in tqdm(cache_files, desc="compressing gradients", unit="task"):
        # Strip "grad_" prefix and ".pt" suffix to get task name.
        task_name = os.path.basename(path)[len("grad_"):-len(".pt")]
        task_names.append(task_name)
        if os.path.exists(_sketch_path(sketch_dir, task_name)):
            logger.info("  [already compressed] %s", task_name)
            continue
        grads = torch.load(path, map_location="cuda", weights_only=True)
        sketch, global_norm, module_norms = _build_global_sketch(grads, k=k, seed=seed)
        del grads
        torch.cuda.empty_cache()
        _save_sketch_cache(task_name, sketch, global_norm, module_norms, sketch_dir)
        logger.info("  compressed → %s", _sketch_path(sketch_dir, task_name))
    return task_names


def _normalize_target_modules(target_modules: Any) -> List[str]:
    """Narrow peft.LoraConfig.target_modules (list[str] | str | None) to list[str]."""
    if target_modules is None:
        raise RuntimeError(
            "LoraConfig.target_modules is None; cannot determine which weights to grad."
        )
    if isinstance(target_modules, str):
        return [target_modules]
    return list(target_modules)


def compute_task_gradient(
    model: Any,
    tokenizer: Any,
    task: Any,
    *,
    max_steps: int = 4,
    batch_size: int = 4,
    max_seq_length: int = 256,
    seed: int = 42,
) -> Dict[str, torch.Tensor]:
    """Average per-step gradient on one task's training data, base model only."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling
    from .lora_config import build_lora_config
    from .load_dataset import load_training_dataset
    from .slice.gradients import accumulate_gradients
    from .slice.utils import (
        model_device,
        target_weight_params,
        tokenize_dataset,
    )

    lora_cfg = build_lora_config()
    target_modules = _normalize_target_modules(lora_cfg.target_modules)
    target_params = target_weight_params(model, target_modules)
    ds, _ = load_training_dataset(task=task, eval_size=1, seed=seed)
    ds = tokenize_dataset(ds, tokenizer=tokenizer, max_length=max_seq_length)
    # num_workers=0: no subprocess spawning — avoids dill/tempfile crash when
    # /tmp is absent on the server. Single-process dataloading is fine here
    # since the bottleneck is the backward pass, not data I/O.
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        num_workers=0,
    )
    grads, steps = accumulate_gradients(
        model=model,
        dataloader=loader,
        target_params=target_params,
        device=model_device(model),
        max_steps=max_steps,
    )
    denom = max(1, steps)
    return {k: (v / float(denom)).detach() for k, v in grads.items()}


def pair_conflict(
    grads_forget: Dict[str, torch.Tensor],
    grads_retain: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """Conflict metrics treating `grads_forget` as new task, `grads_retain` as old.

    Reports what the SLICE projection would see:
      - global cosine similarity (sign indicates conflict)
      - pct_conflicting: % of modules with dot(g_f, g_r) < 0 (projection fires)
      - ratio_mean: mean ||G_tilde||/||G_forget|| across modules
    """
    import torch

    per_module_ratios = []
    per_module_cosines = []
    n_conflicting = 0
    n_modules = 0
    global_dot = 0.0
    global_nf_sq = 0.0
    global_nr_sq = 0.0
    eps = 1e-12

    for name, g_f in grads_forget.items():
        g_r = grads_retain.get(name)
        if g_r is None:
            continue
        n_modules += 1
        f = g_f.detach().float().view(-1).double()
        r = g_r.detach().float().view(-1).double()
        dot = float(torch.dot(f, r).item())
        nf = float(f.norm().item())
        nr = float(r.norm().item())
        cos = dot / (nf * nr + eps)

        if dot < 0:
            n_conflicting += 1
            gamma = -dot / (nr * nr + eps)
            proj_norm = float((f + gamma * r).norm().item())
            ratio = proj_norm / (nf + eps)
        else:
            ratio = 1.0

        per_module_ratios.append(ratio)
        per_module_cosines.append(cos)
        global_dot += dot
        global_nf_sq += nf * nf
        global_nr_sq += nr * nr

    if n_modules == 0:
        return {"n_modules": 0}

    ratios = torch.tensor(per_module_ratios)
    cosines = torch.tensor(per_module_cosines)
    return {
        "n_modules": n_modules,
        "pct_conflicting": n_conflicting / n_modules * 100.0,
        "cosine_mean": float(cosines.mean()),
        "cosine_median": float(cosines.median()),
        "cosine_min": float(cosines.min()),
        "ratio_mean": float(ratios.mean()),
        "ratio_min": float(ratios.min()),
        "ratio_below_0.5_pct": float((ratios < 0.5).float().mean() * 100),
        "global_cosine": global_dot / ((global_nf_sq ** 0.5) * (global_nr_sq ** 0.5) + eps),
    }


def analyze_sequence(
    sequence: Sequence,
    *,
    model_name: str | None = None,
    max_steps: int = 4,
    batch_size: int = 4,
    max_seq_length: int = 256,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:  # keys are "forget|retain"
    """Load base model, compute per-task gradients, report all ordered pairs.

    Returns dict keyed by "forget|retain" with the pair_conflict stats.
    """
    from .train import HF_TOKEN, MODEL_NAME, build_tokenizer, load_base_model

    model_name = model_name or MODEL_NAME
    tokenizer = build_tokenizer(model_name=model_name, hf_token=HF_TOKEN)
    model = load_base_model(model_name=model_name, hf_token=HF_TOKEN)

    logger.info(
        "Computing per-task gradients: n_tasks=%d  max_steps=%d  batch_size=%d",
        len(sequence.tasks), max_steps, batch_size,
    )
    task_grads: Dict[str, Dict[str, torch.Tensor]] = {}
    for task in sequence.tasks:
        name = getattr(task, "name", str(task))
        logger.info("  computing gradient for task=%s", name)
        task_grads[name] = compute_task_gradient(
            model, tokenizer, task,
            max_steps=max_steps, batch_size=batch_size,
            max_seq_length=max_seq_length, seed=seed,
        )

    task_names = list(task_grads)
    report: Dict[str, Dict[str, float]] = {}

    print()
    print("=" * 84)
    print(f"Pairwise gradient conflict — sequence: {sequence.name}")
    print(f"{sequence.description}")
    print(f"base model: {model_name}  (gradients on untrained base)")
    print("=" * 84)
    print(f"{'forget (new)':26s} {'retain (old)':26s} "
          f"{'glob_cos':>9s} {'mean_cos':>9s} {'ratio_mu':>9s} {'conf%':>6s}")
    print("-" * 84)
    for i, f_name in enumerate(task_names):
        for j, r_name in enumerate(task_names):
            if i == j:
                continue
            s = pair_conflict(task_grads[f_name], task_grads[r_name])
            report[f"{f_name}|{r_name}"] = s
            print(
                f"{f_name[:26]:26s} {r_name[:26]:26s} "
                f"{s['global_cosine']:+9.4f} {s['cosine_mean']:+9.4f} "
                f"{s['ratio_mean']:9.4f} {s['pct_conflicting']:5.1f}%"
            )
    print("-" * 84)
    print("glob_cos : global cosine sim (all modules flattened together)")
    print("mean_cos : mean per-module cosine sim")
    print("ratio_mu : mean ||G_tilde|| / ||G_forget|| (1.0 = projection is a no-op)")
    print("conf%    : % of modules where dot(g_forget, g_retain) < 0")
    print()
    return report


def _all_superni_tasks_safe() -> List[SuperNITask]:
    """Like task_sequences.all_superni_tasks but skips TraceTask entries.

    The upstream helper assumes every task in every registered sequence has
    `.ni_id`, which crashes on TRACE-Dummy (contains TraceTask). We filter.
    """
    seen: set = set()
    out: List[SuperNITask] = []
    for seq in SEQUENCES.values():
        for task in seq.tasks:
            nid = getattr(task, "ni_id", None)
            if nid is None or nid in seen:
                continue
            seen.add(nid)
            out.append(task)
    return out


def _resolve_task_pool(pool_arg: str | None) -> List[SuperNITask]:
    """Parse --pool arg into a list of SuperNITask objects.

    Accepts comma-separated NI-IDs ("NI195,NI363") or "all" (the full
    registered SuperNI catalogue) or None (the curated DEFAULT_SEARCH_POOL).
    """
    if pool_arg is None:
        return list(DEFAULT_SEARCH_POOL)
    if pool_arg.strip().lower() == "all":
        return _all_superni_tasks_safe()
    wanted = {tok.strip() for tok in pool_arg.split(",") if tok.strip()}
    by_id = {t.ni_id: t for t in _all_superni_tasks_safe()}
    missing = wanted - set(by_id)
    if missing:
        raise ValueError(
            f"Unknown NI-IDs in --pool: {sorted(missing)}. "
            f"Known: {sorted(by_id)}"
        )
    return [by_id[nid] for nid in wanted]


def search_opposite_pairs(
    tasks: List[Any],
    *,
    model_name: str | None = None,
    max_steps: int = 4,
    batch_size: int = 4,
    max_seq_length: int = 256,
    seed: int = 42,
    top_k: int = 10,
    cache_dir: str | None = None,
) -> List[Tuple[str, str, Dict[str, float]]]:
    """Compute pairwise conflict across `tasks`, rank by most-opposite first.

    When `cache_dir` is given, gradients are stored on disk as fp16 one task at
    a time — the model is never in VRAM alongside more than one gradient dict.
    This lets the full 26-task pool run on 48 GB VRAM that would otherwise OOM
    if all gradients were kept in memory simultaneously.

    Returns all unordered pairs with their stats, sorted by global_cosine
    ascending (most negative = most opposite first).
    """
    import os
    import torch
    from .train import HF_TOKEN, MODEL_NAME, build_tokenizer, load_base_model

    model_name = model_name or MODEL_NAME

    # Determine which tasks still need gradient computation.
    def _needs_compute(task: Any) -> bool:
        if cache_dir is None:
            return True
        name = getattr(task, "name", str(task))
        return not os.path.exists(_cache_path(cache_dir, name))

    tasks_to_compute = [t for t in tasks if _needs_compute(t)]

    logger.info(
        "Search: n_tasks=%d  to_compute=%d  pairs=%d  max_steps=%d  batch_size=%d",
        len(tasks),
        len(tasks_to_compute),
        len(tasks) * (len(tasks) - 1) // 2,
        max_steps,
        batch_size,
    )

    # --- Phase 1: compute gradients ------------------------------------------
    # Keep all in a dict (GPU) when no cache_dir, or stream to disk otherwise.
    task_grads: Dict[str, Dict[str, Any]] = {}  # only populated when cache_dir is None
    task_names: List[str] = []

    if tasks_to_compute:
        tokenizer = build_tokenizer(model_name=model_name, hf_token=HF_TOKEN)
        model = load_base_model(model_name=model_name, hf_token=HF_TOKEN)

        for task in tasks:
            name = getattr(task, "name", str(task))
            if cache_dir and os.path.exists(_cache_path(cache_dir, name)):
                logger.info("  [cached] %s", name)
                task_names.append(name)
                continue
            logger.info("  grad for task=%s", name)
            try:
                grads = compute_task_gradient(
                    model, tokenizer, task,
                    max_steps=max_steps, batch_size=batch_size,
                    max_seq_length=max_seq_length, seed=seed,
                )
                if cache_dir:
                    _save_grad_cache(name, grads, cache_dir)
                    del grads
                    torch.cuda.empty_cache()
                else:
                    task_grads[name] = grads
                task_names.append(name)
            except Exception as exc:  # noqa: BLE001
                logger.warning("  SKIP %s (failed to load/compute): %s", name, exc)

        if cache_dir:
            # Free the model before scoring so Phase 2 only holds 2 grad dicts.
            del model, tokenizer
            torch.cuda.empty_cache()
            logger.info("Model freed. Starting pairwise scoring from cache.")
    else:
        # All already cached — no model needed at all.
        task_names = [getattr(t, "name", str(t)) for t in tasks]

    # --- Phase 2: pairwise scoring -------------------------------------------
    from tqdm import tqdm
    from itertools import combinations

    pairs: List[Tuple[str, str, Dict[str, float]]] = []
    pair_list = list(combinations(range(len(task_names)), 2))
    for i, j in tqdm(pair_list, desc="scoring pairs", unit="pair"):
        a, b = task_names[i], task_names[j]
        if cache_dir:
            ga = _load_grad_cache(a, cache_dir)
            gb = _load_grad_cache(b, cache_dir)
            s = pair_conflict(ga, gb)
            del ga, gb
            torch.cuda.empty_cache()
        else:
            s = pair_conflict(task_grads[a], task_grads[b])
        pairs.append((a, b, s))

    pairs.sort(key=lambda x: x[2].get("global_cosine", 0.0))

    print()
    print("=" * 96)
    print(f"Pair search — {len(names)} tasks, {len(pairs)} unordered pairs")
    print(f"base model: {model_name}  (gradients on untrained base)")
    print("=" * 96)
    header = (
        f"{'task_a':34s} {'task_b':34s} "
        f"{'glob_cos':>9s} {'mean_cos':>9s} {'conf%':>6s} {'ratio_mu':>9s}"
    )
    row = (
        "{:34s} {:34s} {:+9.4f} {:+9.4f} {:5.1f}% {:9.4f}"
    )

    print("\n[MOST OPPOSITE — SLICE projection fires here]")
    print(header)
    print("-" * 96)
    for a, b, s in pairs[:top_k]:
        print(row.format(
            a[:34], b[:34],
            s["global_cosine"], s["cosine_mean"],
            s["pct_conflicting"], s["ratio_mean"],
        ))

    print("\n[MOST ALIGNED — SLICE does nothing here]")
    print(header)
    print("-" * 96)
    for a, b, s in pairs[-top_k:][::-1]:
        print(row.format(
            a[:34], b[:34],
            s["global_cosine"], s["cosine_mean"],
            s["pct_conflicting"], s["ratio_mean"],
        ))
    print()
    return pairs


def find_best_sequences(
    pairs: List[Tuple[str, str, Dict[str, float]]],
    n: int = 5,
    top_k: int = 5,
    metric: str = "global_cosine",
) -> List[Tuple[List[str], float]]:
    """Brute-force search for the n-task subset with minimum mean pairwise cosine.

    C(26, 5) = 65 780 combos × C(5,2) = 10 lookups each → ~660K ops, instant.
    Returns top_k subsets ordered by ascending mean cosine (most opposite first).
    """
    from itertools import combinations

    score_lookup: Dict[Tuple[str, str], float] = {}
    all_names: set = set()
    for a, b, s in pairs:
        v = s.get(metric, 0.0)
        score_lookup[(a, b)] = v
        score_lookup[(b, a)] = v
        all_names.add(a)
        all_names.add(b)

    names = sorted(all_names)
    n_pairs_in_combo = n * (n - 1) // 2
    best: List[Tuple[float, List[str]]] = []

    for combo in combinations(names, n):
        mean_cos = sum(
            score_lookup.get((combo[i], combo[j]), 0.0)
            for i in range(n)
            for j in range(i + 1, n)
        ) / n_pairs_in_combo

        if len(best) < top_k or mean_cos < best[-1][0]:
            best.append((mean_cos, list(combo)))
            best.sort(key=lambda x: x[0])
            if len(best) > top_k:
                best.pop()

    return [(task_list, score) for score, task_list in best]


def _print_best_sequences(
    results: List[Tuple[List[str], float]],
    n: int,
    metric: str = "global_cosine",
) -> None:
    print()
    print("=" * 80)
    print(f"Top-{len(results)} most-opposite {n}-task subsets  (metric: {metric})")
    print("Lower = more opposite = SLICE projection fires more")
    print("=" * 80)
    for rank, (task_list, score) in enumerate(results, 1):
        print(f"\n#{rank}  mean {metric} = {score:+.4f}")
        for i, t in enumerate(task_list, 1):
            print(f"    {i}. {t}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure gradient opposition across CL tasks."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--sequence",
        help="Analyse one registered sequence pairwise (ordered pairs).",
    )
    mode.add_argument(
        "--search", action="store_true",
        help="Search all unordered pairs in a task pool; rank by opposition.",
    )
    mode.add_argument(
        "--compress", action="store_true",
        help=(
            "One-time step: read each gradient in --cache-dir once, build a "
            "CountSketch (~800 KB), save to --sketch-dir. After this, --search "
            "with --sketch-dir scores all pairs in seconds instead of hours."
        ),
    )
    parser.add_argument(
        "--pool", default=None,
        help=(
            "For --search: comma-separated NI-IDs (e.g. NI195,NI363,NI618), "
            "'all' for the full catalogue, or omit for the default pool."
        ),
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--model-name", default=None, help="Override base model.")
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--cache-dir", default=None,
        help="Directory for per-task gradient caches (fp16 .pt files).",
    )
    parser.add_argument(
        "--sketch-dir", default=None,
        help=(
            "Directory for CountSketch summaries (~800 KB per task). "
            "When provided for --search, scores all pairs from sketches "
            "(seconds) instead of loading full 5.6 GB gradient files (hours)."
        ),
    )
    parser.add_argument(
        "--find-sequence", type=int, default=None, metavar="N",
        help=(
            "After pair search, brute-force all C(tasks, N) subsets and print "
            "the top-k with the lowest mean pairwise cosine (most SLICE-friendly)."
        ),
    )
    args = parser.parse_args()

    # Suppress library FutureWarnings we can't control.
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

    lvl = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("peft").setLevel(logging.WARNING)

    if args.compress:
        if not args.cache_dir:
            parser.error("--compress requires --cache-dir")
        sketch_dir = args.sketch_dir or (args.cache_dir + "_sketches")
        compress_grad_cache(args.cache_dir, sketch_dir, seed=args.seed)
        print(f"\nSketches saved to: {sketch_dir}")
        print("Now run --search --sketch-dir to score all pairs in seconds.")

    elif args.search:
        pool = _resolve_task_pool(args.pool)

        if args.sketch_dir:
            # Fast path: score all pairs from tiny sketch files.
            import glob, os
            from itertools import combinations
            from tqdm import tqdm

            sketch_files = sorted(glob.glob(os.path.join(args.sketch_dir, "sketch_*.pt")))
            task_names = [
                os.path.basename(p)[len("sketch_"):-len(".pt")]
                for p in sketch_files
            ]
            logger.info("Scoring %d tasks (%d pairs) from sketches in %s",
                        len(task_names), len(task_names) * (len(task_names) - 1) // 2,
                        args.sketch_dir)

            sketches = {n: _load_sketch_cache(n, args.sketch_dir) for n in task_names}
            pair_list = list(combinations(range(len(task_names)), 2))
            pairs = []
            for i, j in tqdm(pair_list, desc="scoring pairs (sketch)", unit="pair"):
                a, b = task_names[i], task_names[j]
                s = pair_conflict_from_sketch(sketches[a], sketches[b])
                pairs.append((a, b, s))

            pairs.sort(key=lambda x: x[2].get("global_cosine", 0.0))

            header = (f"{'task_a':34s} {'task_b':34s} {'glob_cos':>9s}")
            row = "{:34s} {:34s} {:+9.4f}"
            print()
            print("=" * 80)
            print(f"Pair search (sketch) — {len(task_names)} tasks, {len(pairs)} pairs")
            print("=" * 80)
            print("\n[MOST OPPOSITE — SLICE projection fires here]")
            print(header)
            print("-" * 80)
            for a, b, s in pairs[:args.top_k]:
                print(row.format(a[:34], b[:34], s["global_cosine"]))
            print("\n[MOST ALIGNED — SLICE does nothing here]")
            print(header)
            print("-" * 80)
            for a, b, s in pairs[-args.top_k:][::-1]:
                print(row.format(a[:34], b[:34], s["global_cosine"]))
            print()
        else:
            pairs = search_opposite_pairs(
                pool,
                model_name=args.model_name,
                max_steps=args.max_steps,
                batch_size=args.batch_size,
                max_seq_length=args.max_seq_length,
                seed=args.seed,
                top_k=args.top_k,
                cache_dir=args.cache_dir,
            )

        if args.find_sequence is not None:
            results = find_best_sequences(pairs, n=args.find_sequence, top_k=args.top_k)
            _print_best_sequences(results, n=args.find_sequence)

    else:
        seq = get_sequence(args.sequence)
        analyze_sequence(
            seq,
            model_name=args.model_name,
            max_steps=args.max_steps,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            seed=args.seed,
        )


__all__ = [
    "NI_SEQ_CROSSMODE_REVIEWS",
    "NI_SEQ_CROSSMODE_QA",
    "NI_SEQ_CONTROL_RATING",
    "NI_SEQ_CONTROL_SUMM",
    "DEFAULT_SEARCH_POOL",
    "compute_task_gradient",
    "pair_conflict",
    "analyze_sequence",
    "search_opposite_pairs",
    "find_best_sequences",
]


if __name__ == "__main__":
    main()
