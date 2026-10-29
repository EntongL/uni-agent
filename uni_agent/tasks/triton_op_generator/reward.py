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


def score_verifier_result(
    payload: Mapping[str, Any],
    *,
    target_speedup: float = 2.0,
) -> dict[str, Any]:
    """Translate verifier-skill verify.py / benchmark.py output into a Task reward.

    Correct kernels start at 0.7. Measured speedup can add up to 0.3, reaching
    1.0 at ``target_speedup``. A compiled but incorrect candidate scores 0.3.
    An evaluation that ran but did not compile scores 0.1.
    """
    if target_speedup <= 0:
        raise ValueError("target_speedup must be positive")
    compile_ok = bool(payload.get("compile_ok") or payload.get("compiled"))
    passed = int(payload.get("passed_cases") or 0)
    total = int(payload.get("total_cases") or 0)
    failed = int(payload.get("failed_cases") or max(total - passed, 0))
    correctness_ok = bool(payload.get("correctness_ok") or payload.get("correctness"))
    if total and passed == total and failed == 0:
        correctness_ok = True
        compile_ok = True
    if "eval_completed" in payload:
        eval_completed = bool(payload.get("eval_completed"))
    else:
        status = payload.get("status")
        eval_completed = not (isinstance(status, str) and status.lower() in {"failed", "timeout", "error"})
    pass_rate = min(1.0, max(0.0, passed / total)) if total else float(correctness_ok)
    speedup = _optional_number(payload.get("speedup"), field="speedup")

    if not eval_completed:
        reward = 0.0
    elif correctness_ok:
        reward = 0.7 + 0.3 * min((speedup or 0.0) / target_speedup, 1.0)
    elif compile_ok:
        reward = 0.3
    else:
        reward = 0.1

    error_message = payload.get("error_message")
    if error_message is not None and not isinstance(error_message, str):
        error_message = str(error_message)

    return {
        "reward": round(reward, 4),
        "accuracy": float(pass_rate),
        "resolved": correctness_ok,
        "eval_completed": eval_completed,
        "status": "completed" if eval_completed else "failed",
        "compiled": compile_ok,
        "correctness": correctness_ok,
        "compile_ok": compile_ok,
        "correctness_ok": correctness_ok,
        "passed_cases": passed,
        "total_cases": total,
        "failed_cases": failed,
        "decoy_kernel": False,
        "speedup": speedup,
        "target_speedup": target_speedup,
        "case_summary": {
            "passed_cases": passed,
            "total_cases": total,
            "failed_cases": failed,
        },
        "error_code": payload.get("error_code") or payload.get("verify_exit_code"),
        "error_message": error_message,
    }


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
