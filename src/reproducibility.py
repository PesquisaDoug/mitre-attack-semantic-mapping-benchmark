from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.data import sha256_file


def get_git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def parameter_summary(torch_module: Any) -> dict[str, int | float]:
    parameters = list(torch_module.parameters())
    parameter_count = sum(parameter.numel() for parameter in parameters)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in parameters)
    return {
        "parameter_count": int(parameter_count),
        "parameter_memory_mb": float(parameter_bytes / (1024 ** 2)),
    }


def commit_hash_from_config(config: Any) -> str | None:
    return getattr(config, "_commit_hash", None)


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "sentence_transformers": "sentence_transformers",
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "scikit_learn": "sklearn",
    }
    for key, module_name in packages.items():
        try:
            module = __import__(module_name)
            versions[key] = getattr(module, "__version__", None)
        except Exception:
            versions[key] = None
    return versions


def device_metadata() -> dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "torch_device": "cuda",
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 3),
                "cuda_version": getattr(torch.version, "cuda", None),
            }
        return {
            "torch_device": "cpu",
            "gpu_name": None,
            "gpu_memory_gb": None,
            "cuda_version": getattr(torch.version, "cuda", None),
        }
    except Exception:
        return {"torch_device": "unknown", "gpu_name": None, "gpu_memory_gb": None, "cuda_version": None}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")


__all__ = [
    "sha256_file",
    "get_git_commit",
    "parameter_summary",
    "commit_hash_from_config",
    "package_versions",
    "device_metadata",
    "now_utc",
    "write_json",
]
