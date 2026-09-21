#!/usr/bin/env python3
"""Submit one final candidate to a KernelGYM service from inside a sandbox.

This helper uses only the standard library so the task can copy it into the
resident Ascend container after the agent has finished. Keeping the final call
outside the agent-owned skill tree prevents a fabricated ``eval_result.json``
from being mistaken for the training reward.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

_TERMINAL_STATUSES = {"completed", "failed", "timeout", "timed_out", "cancelled", "canceled", "error"}
_DEFAULT_REQUEST_TIMEOUT = 30.0
_MAX_RESULT_REQUEST_TIMEOUT = 120.0


def _request_json(
    url: str,
    *,
    payload: dict[str, Any] | None = None,
    timeout: float = _DEFAULT_REQUEST_TIMEOUT,
) -> dict[str, Any]:
    """Send one KernelGYM request and decode its JSON response.

    The timeout is a socket timeout for this individual request, not the
    overall evaluation budget.  In particular, the submit endpoint may take
    longer than the polling requests when KernelGYM starts work synchronously.
    """
    if timeout <= 0:
        raise ValueError("KernelGYM request timeout must be positive")
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - caller configures the trusted local service.
            decoded = response.read().decode("utf-8")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[-1000:]
        raise RuntimeError(f"KernelGYM HTTP {exc.code}: {detail}") from exc
    except TimeoutError as exc:
        raise TimeoutError(f"KernelGYM request timed out after {timeout:g}s: {url}") from exc
    except URLError as exc:
        if isinstance(exc.reason, TimeoutError):
            raise TimeoutError(f"KernelGYM request timed out after {timeout:g}s: {url}") from exc
        raise RuntimeError(f"KernelGYM request failed: {exc.reason}") from exc
    try:
        result = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"KernelGYM returned invalid JSON: {decoded[-1000:]!r}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("KernelGYM response must be a JSON object")
    return result


def _status_value(payload: dict[str, Any]) -> str | None:
    status = payload.get("status")
    if isinstance(status, str):
        return status.lower()
    task = payload.get("task")
    if isinstance(task, dict) and isinstance(task.get("status"), str):
        return task["status"].lower()
    return None


def _task_id(payload: dict[str, Any]) -> str | None:
    for key in ("task_id", "id"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    task = payload.get("task")
    if isinstance(task, dict):
        return _task_id(task)
    return None


def _result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    for key in ("result", "data"):
        nested = payload.get(key)
        if isinstance(nested, dict) and ("status" in nested or "correctness" in nested or "compiled" in nested):
            return nested
    return payload


def evaluate(
    *,
    url: str,
    task_id: str,
    reference_code: str,
    kernel_code: str,
    entry_point: str,
    backend: str,
    toolkit: str,
    correctness_trials: int,
    performance_trials: int,
    enable_profiling: bool,
    enable_triton_detection: bool,
    timeout: float,
    poll_interval: float,
) -> dict[str, Any]:
    base_url = url.rstrip("/")
    submission = _request_json(
        f"{base_url}/evaluate",
        payload={
            "task_id": task_id,
            "reference_code": reference_code,
            "kernel_code": kernel_code,
            "entry_point": entry_point,
            "backend": backend,
            "toolkit": toolkit,
            "enable_profiling": enable_profiling,
            "enable_triton_detection": enable_triton_detection,
            "num_correct_trials": correctness_trials,
            "num_perf_trials": performance_trials,
        },
        # A slow submit must not be cut off by the old 30-second per-request
        # default.  Keep this inside the task runner's timeout, which adds a
        # 60-second cleanup margin around ``evaluation_timeout``.
        timeout=max(_DEFAULT_REQUEST_TIMEOUT, timeout + 30.0),
    )
    submitted_task_id = _task_id(submission) or task_id
    status = _status_value(submission)
    if status in _TERMINAL_STATUSES:
        return _result_payload(submission)

    deadline = time.monotonic() + timeout
    quoted_task_id = quote(submitted_task_id, safe="")
    last_status = submission
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        time.sleep(min(poll_interval, max(0.0, remaining)))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            last_status = _request_json(
                f"{base_url}/status/{quoted_task_id}",
                timeout=min(_DEFAULT_REQUEST_TIMEOUT, remaining),
            )
        except TimeoutError:
            # A status request is only a probe.  KernelGYM may be briefly
            # overloaded while the worker is compiling; retry until the
            # task-level deadline instead of aborting without result.json.
            continue
        status = _status_value(last_status)
        if status not in _TERMINAL_STATUSES:
            continue
        results = _request_json(
            f"{base_url}/results/{quoted_task_id}",
            timeout=max(_DEFAULT_REQUEST_TIMEOUT, min(_MAX_RESULT_REQUEST_TIMEOUT, timeout)),
        )
        return _result_payload(results)

    return {
        "status": "timeout",
        "compiled": False,
        "correctness": False,
        "decoy_kernel": False,
        "error_message": f"KernelGYM task {submitted_task_id!r} exceeded {timeout:g}s",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a final KernelGYM evaluation")
    parser.add_argument("--url", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--entry-point", default="Model")
    parser.add_argument("--backend", default="triton")
    parser.add_argument("--toolkit", default="sandbox_v3")
    parser.add_argument("--correctness-trials", type=int, default=5)
    parser.add_argument("--performance-trials", type=int, default=100)
    parser.add_argument("--enable-profiling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-triton-detection", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--poll-interval", type=float, default=2)
    args = parser.parse_args()

    try:
        result = evaluate(
            url=args.url,
            task_id=args.task_id,
            reference_code=args.reference.read_text(encoding="utf-8"),
            kernel_code=args.kernel.read_text(encoding="utf-8"),
            entry_point=args.entry_point,
            backend=args.backend,
            toolkit=args.toolkit,
            correctness_trials=args.correctness_trials,
            performance_trials=args.performance_trials,
            enable_profiling=args.enable_profiling,
            enable_triton_detection=args.enable_triton_detection,
            timeout=args.timeout,
            poll_interval=args.poll_interval,
        )
    except Exception as exc:
        print(f"KernelGYM evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
