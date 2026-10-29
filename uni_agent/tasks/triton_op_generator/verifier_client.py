#!/usr/bin/env python3
"""Run the AscendKernelBench verifier skill scripts from inside a sandbox.

The agent workflow uses the same verify.py / benchmark.py pair. The task also
calls this helper after the agent finishes so the training reward comes from
those scripts, not from an agent-written file.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _first_json(directory: Path, names: tuple[str, ...]) -> dict[str, Any] | None:
    for name in names:
        payload = _load_json(directory / name)
        if payload is not None:
            return payload
    return None


def _run(command: list[str], *, timeout: float) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def _speedup(payload: dict[str, Any] | None) -> float | None:
    if not payload:
        return None
    containers = [payload]
    perf_data = payload.get("perf_data")
    if isinstance(perf_data, dict):
        containers.append(perf_data)
        implementation = perf_data.get("implementation")
        if isinstance(implementation, dict):
            containers.append(implementation)
    for container in containers:
        for key in ("speedup_vs_torch", "speedup", "geomean_speedup"):
            value = container.get(key)
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number == number:  # not NaN
                return number
    return None


def _brief_error(verify: dict[str, Any] | None, stderr: str) -> str:
    if verify:
        failures = verify.get("failures")
        if isinstance(failures, list) and failures:
            first = failures[0]
            if isinstance(first, dict):
                return str(first.get("error_msg") or first.get("error") or first)[-1200:]
            return str(first)[-1200:]
        for key in ("error_message", "error", "message"):
            value = verify.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[-1200:]
    return (stderr or "")[-1200:]


def evaluate(
    *,
    verify_script: Path,
    benchmark_script: Path | None,
    op_name: str,
    reference: Path,
    kernel: Path,
    verify_dir: Path,
    timeout: float,
    triton_impl_name: str,
    framework: str,
    dsl: str,
    backend: str,
    warmup: int,
    repeats: int,
    extra_args: list[str],
) -> dict[str, Any]:
    if not verify_script.is_file():
        raise FileNotFoundError(f"verifier script not found: {verify_script}")
    verify_dir.mkdir(parents=True, exist_ok=True)
    reference_copy = verify_dir / f"{op_name}_{framework}.py"
    kernel_copy = verify_dir / f"{op_name}_{triton_impl_name}.py"
    shutil.copyfile(reference, reference_copy)
    shutil.copyfile(kernel, kernel_copy)

    verify_output = verify_dir / "verify_result.json"
    verify_cmd = [
        sys.executable,
        str(verify_script),
        "--op_name",
        op_name,
        "--verify_dir",
        str(verify_dir),
        "--triton_impl_name",
        triton_impl_name,
        "--timeout",
        f"{timeout:g}",
        "--output",
        str(verify_output),
        "--dsl",
        dsl,
        "--backend",
        backend,
        "--framework",
        framework,
        *extra_args,
    ]
    try:
        verify_proc = _run(verify_cmd, timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        return {
            "eval_completed": False,
            "compile_ok": False,
            "correctness_ok": False,
            "passed_cases": 0,
            "total_cases": 0,
            "failed_cases": 0,
            "speedup": None,
            "error_message": f"verify.py exceeded {timeout:g}s",
        }

    verify = _load_json(verify_output) or _first_json(
        verify_dir, ("verify_result.json", "result.json")
    )
    total = int((verify or {}).get("total_cases") or 0)
    passed = int((verify or {}).get("passed_cases") or 0)
    failed = int((verify or {}).get("failed_cases") or max(total - passed, 0))
    correctness_ok = bool(total and passed == total and failed == 0)
    compile_ok = correctness_ok or passed > 0
    if verify:
        error_type = str((verify.get("failures") or [{}])[0].get("error_type") or "") if isinstance(verify.get("failures"), list) and verify.get("failures") else ""
        if error_type and "Compilation" in error_type and passed == 0:
            compile_ok = False
        if verify.get("compile_ok") is True:
            compile_ok = True
        if verify.get("compile_ok") is False and not correctness_ok:
            compile_ok = False

    result: dict[str, Any] = {
        "eval_completed": True,
        "compile_ok": compile_ok,
        "correctness_ok": correctness_ok,
        "passed_cases": passed,
        "total_cases": total,
        "failed_cases": failed,
        "speedup": None,
        "error_message": None if correctness_ok else _brief_error(verify, verify_proc.stderr),
        "verify_exit_code": verify_proc.returncode,
        "verify_stdout_tail": (verify_proc.stdout or "")[-2000:],
        "verify_stderr_tail": (verify_proc.stderr or "")[-2000:],
    }
    if not correctness_ok or benchmark_script is None:
        return result
    if not benchmark_script.is_file():
        result["error_message"] = f"benchmark script not found: {benchmark_script}"
        return result

    perf_output = verify_dir / "perf_result.json"
    bench_cmd = [
        sys.executable,
        str(benchmark_script),
        "--op_name",
        op_name,
        "--verify_dir",
        str(verify_dir),
        "--triton_impl_name",
        triton_impl_name,
        "--warmup",
        str(warmup),
        "--repeats",
        str(repeats),
        "--output",
        str(perf_output),
        *extra_args,
    ]
    try:
        bench_proc = _run(bench_cmd, timeout=timeout + 30)
    except subprocess.TimeoutExpired:
        result["error_message"] = f"benchmark.py exceeded {timeout:g}s"
        return result
    perf = _load_json(perf_output) or _first_json(verify_dir, ("perf_result.json",))
    result["speedup"] = _speedup(perf)
    result["benchmark_exit_code"] = bench_proc.returncode
    if result["speedup"] is None and bench_proc.returncode != 0:
        result["error_message"] = _brief_error(perf, bench_proc.stderr) or result["error_message"]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the agent verifier skill scripts")
    parser.add_argument("--verify-script", type=Path, required=True)
    parser.add_argument("--benchmark-script", type=Path)
    parser.add_argument("--op-name", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--kernel", type=Path, required=True)
    parser.add_argument("--verify-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--triton-impl-name", default="triton_ascend_impl")
    parser.add_argument("--framework", default="torch")
    parser.add_argument("--dsl", default="triton_ascend")
    parser.add_argument("--backend", default="ascend")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--extra-arg", action="append", default=[])
    args = parser.parse_args()

    try:
        result = evaluate(
            verify_script=args.verify_script,
            benchmark_script=args.benchmark_script,
            op_name=args.op_name,
            reference=args.reference,
            kernel=args.kernel,
            verify_dir=args.verify_dir,
            timeout=args.timeout,
            triton_impl_name=args.triton_impl_name,
            framework=args.framework,
            dsl=args.dsl,
            backend=args.backend,
            warmup=args.warmup,
            repeats=args.repeats,
            extra_args=list(args.extra_arg),
        )
    except Exception as exc:
        print(f"verifier evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "compile_ok": result.get("compile_ok"),
        "correctness_ok": result.get("correctness_ok"),
        "passed_cases": result.get("passed_cases"),
        "total_cases": result.get("total_cases"),
        "speedup": result.get("speedup"),
        "error": result.get("error_message"),
        "result_file": str(args.output),
    }, ensure_ascii=False))
    return 0 if result.get("correctness_ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
