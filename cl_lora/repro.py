from __future__ import annotations

import os
import random


def set_global_seed(
    seed: int,
    *,
    deterministic: bool = True,
    warn_only: bool = True,
    set_env: bool = True,
) -> None:
    """Set common RNG seeds for experiment reproducibility.

    This covers:
      - Python's `random`
      - NumPy (if installed)
      - PyTorch CPU and CUDA RNGs
      - Hugging Face Transformers helpers (if installed)

    Notes:
      - Some environment variables (e.g., `PYTHONHASHSEED`, `CUBLAS_WORKSPACE_CONFIG`)
        are most effective when set before the process starts. We still set them here
        to reduce surprises when users call this late.
      - `deterministic=True` may reduce performance and can surface warnings/errors
        if an operation is inherently non-deterministic.
    """

    if not isinstance(seed, int):
        raise TypeError(f"seed must be int, got {type(seed).__name__}")

    if set_env:
        os.environ["PYTHONHASHSEED"] = str(seed)
        # Improves determinism for some CUDA BLAS kernels when set early.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

    random.seed(seed)

    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=warn_only)  # type: ignore[arg-type]
            except TypeError:
                # Older torch versions don't support warn_only.
                torch.use_deterministic_algorithms(True)
            except Exception:
                # If a platform doesn't support this, keep seeding behavior.
                pass

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    except Exception:
        pass

    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
    except Exception:
        pass
