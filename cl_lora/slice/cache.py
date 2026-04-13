from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import logging

logger = logging.getLogger("cl_lora.slice.cache")


@dataclass
class SliceCacheEntry:
    inits: Dict[str, Dict[str, torch.Tensor]]

    def to(self, device: torch.device) -> "SliceCacheEntry":
        for _, ab in self.inits.items():
            for k, v in ab.items():
                ab[k] = v.to(device)
        return self


def make_cache_key(payload: Dict[str, Any]) -> str:
    def _to_json_safe(obj: Any) -> Any:
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        if isinstance(obj, dict):
            return {str(k): _to_json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [_to_json_safe(v) for v in obj]
        return str(obj)

    safe_payload = _to_json_safe(payload)
    payload_str = json.dumps(safe_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload_str.encode("utf-8")).hexdigest()


def load_slice_cache(
    cache_dir: str,
    cache_key: str,
    device: Optional[torch.device] = None,
) -> Optional[SliceCacheEntry]:
    root_dir = os.path.join(cache_dir, cache_key)
    if not os.path.isdir(root_dir):
        logger.debug("Slice cache root missing: %s", root_dir)
        return None

    inits_dir = os.path.join(root_dir, "inits")
    if not os.path.isdir(inits_dir):
        return None

    inits: Dict[str, Dict[str, torch.Tensor]] = {}
    for fname in os.listdir(inits_dir):
        if not fname.endswith(".pt"):
            continue
        path = os.path.join(inits_dir, fname)
        key = fname[:-3]
        map_loc = device if device is not None else "cpu"
        payload = torch.load(path, map_location=map_loc, weights_only=True)
        if isinstance(payload, dict) and "A" in payload and "B" in payload:
            inits[key] = {"A": payload["A"], "B": payload["B"]}

    if not inits:
        logger.debug("Slice cache at %s contains no inits", root_dir)
        return None

    logger.info("Loaded slice cache from %s with %d modules", root_dir, len(inits))
    return SliceCacheEntry(inits=inits)


def save_slice_cache(
    cache_dir: str,
    cache_key: str,
    entry: SliceCacheEntry,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    root_dir = os.path.join(cache_dir, cache_key)
    inits_dir = os.path.join(root_dir, "inits")
    os.makedirs(inits_dir, exist_ok=True)

    for name, ab in entry.inits.items():
        payload = {"A": ab["A"], "B": ab["B"]}
        torch.save(payload, os.path.join(inits_dir, f"{name}.pt"))

    if meta is not None:
        meta_path = os.path.join(root_dir, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, sort_keys=True, indent=2)
    logger.info("Saved slice cache to %s with %d modules", root_dir, len(entry.inits))
