"""Translate a final KernelGYM response into a scalar Task reward."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


def _optional_number(value: object, *, field: str) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"KernelGYM field {field!r} must be numeric when present") from exc
    if not math.isfinite(number):
        raise ValueError(f"KernelGYM field {field!r} must be finite when present")
    return number


def score_kernelgym_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a staged KernelGYM reward and compact evaluator diagnostics.

    A fully correct, non-decoy kernel scores 1. A completed evaluation that
    compiled but is not correct scores 0.3. A completed evaluation that did not
    compile scores 0.1. Decoys and evaluations that did not complete stay at 0.
    Accuracy stays binary. Performance is reported and does not change the reward.
    """
    status = payload.get("status")
    if not isinstance(status, str) or not status.strip():
        raise ValueError("KernelGYM result must contain a non-empty string 'status'")

    compiled = bool(payload.get("compiled", False))
    correctness = bool(payload.get("correctness", False))
    decoy_kernel = bool(payload.get("decoy_kernel", False))
    completed = status.lower() == "completed"
    correct = completed and compiled and correctness and not decoy_kernel
    if correct:
        reward = 1.0
    elif completed and compiled and not decoy_kernel:
        reward = 0.3
    elif completed and not decoy_kernel:
        reward = 0.1
    else:
        reward = 0.0

    metadata = payload.get("metadata")
    case_summary = metadata.get("case_summary") if isinstance(metadata, Mapping) else None
    if not isinstance(case_summary, Mapping):
        case_summary = None

    error_message = payload.get("error_message")
    if error_message is not None and not isinstance(error_message, str):
        error_message = str(error_message)

    return {
        "reward": reward,
        "accuracy": float(correct),
        "resolved": correct,
        "eval_completed": completed,
        "status": status,
        "compiled": compiled,
        "correctness": correctness,
        "decoy_kernel": decoy_kernel,
        "reference_runtime_ms": _optional_number(payload.get("reference_runtime"), field="reference_runtime"),
        "kernel_runtime_ms": _optional_number(payload.get("kernel_runtime"), field="kernel_runtime"),
        "speedup": _optional_number(payload.get("speedup"), field="speedup"),
        "case_summary": dict(case_summary) if case_summary is not None else None,
        "error_code": payload.get("error_code"),
        "error_message": error_message,
    }
