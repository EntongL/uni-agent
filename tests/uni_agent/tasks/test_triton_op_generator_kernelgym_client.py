from __future__ import annotations

import ast
import json
import sys

import pytest

from uni_agent.tasks.triton_op_generator import kernelgym_client


def _evaluate_kwargs(**overrides):
    values = {
        "url": "http://kernelgym",
        "task_id": "task-1",
        "reference_code": "class Model: pass\n",
        "kernel_code": "class ModelNew: pass\n",
        "entry_point": "Model",
        "backend": "triton",
        "toolkit": "sandbox_v3",
        "correctness_trials": 1,
        "performance_trials": 1,
        "enable_profiling": True,
        "enable_triton_detection": True,
        "timeout": 900.0,
        "poll_interval": 0.0,
    }
    values.update(overrides)
    return values


@pytest.mark.cpu
@pytest.mark.level0
def test_evaluate_does_not_cut_off_a_slow_submit_at_thirty_seconds(monkeypatch):
    calls = []

    def fake_request(url, *, payload=None, timeout):
        calls.append((url, timeout))
        if url.endswith("/evaluate"):
            return {"task_id": "task-1", "status": "queued"}
        if url.endswith("/status/task-1"):
            return {"status": "completed"}
        return {"status": "completed", "compiled": True, "correctness": True}

    monkeypatch.setattr(kernelgym_client, "_request_json", fake_request)

    result = kernelgym_client.evaluate(**_evaluate_kwargs())

    assert result["status"] == "completed"
    assert calls[0] == ("http://kernelgym/evaluate", 930.0)
    assert calls[1][1] == 30.0
    assert calls[2][1] == 120.0


@pytest.mark.cpu
@pytest.mark.level0
def test_status_request_timeout_is_retried_until_task_deadline(monkeypatch):
    def fake_request(url, *, payload=None, timeout):
        if url.endswith("/evaluate"):
            return {"task_id": "task-1", "status": "queued"}
        raise TimeoutError("timed out")

    monkeypatch.setattr(kernelgym_client, "_request_json", fake_request)

    result = kernelgym_client.evaluate(**_evaluate_kwargs(timeout=0.02, poll_interval=0.001))

    assert result["status"] == "timeout"
    assert "task-1" in result["error_message"]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("candidate_class", "entry_point", "expected_error"),
    [
        ("Model", "Model", "Kernel code validation failed"),
        ("ModelNew", "ModelNew", "Reference code validation failed"),
        ("ModelNew", "Model", None),
    ],
)
def test_cli_uses_reference_entry_point_and_separate_candidate_class(
    monkeypatch, tmp_path, capsys, candidate_class, entry_point, expected_error,
):
    reference = tmp_path / "reference.py"
    kernel = tmp_path / "kernel_code.py"
    output = tmp_path / "result.json"
    reference.write_text("class Model: pass\n", encoding="utf-8")
    kernel.write_text(f"class {candidate_class}: pass\n", encoding="utf-8")

    def fake_kernelgym(url, *, payload=None, timeout):
        # Reproduce the service validation seen in the failed episode:
        # entry_point selects the reference; the candidate is always ModelNew.
        assert url == "http://kernelgym/evaluate"
        reference_classes = {
            node.name for node in ast.parse(payload["reference_code"]).body if isinstance(node, ast.ClassDef)
        }
        kernel_classes = {
            node.name for node in ast.parse(payload["kernel_code"]).body if isinstance(node, ast.ClassDef)
        }
        error = None
        if payload["entry_point"] not in reference_classes:
            error = "Reference code validation failed: Code must contain a 'ModelNew' class"
        elif "ModelNew" not in kernel_classes:
            error = "Kernel code validation failed: Code must contain a 'ModelNew' class"
        return {
            "status": "failed" if error else "completed",
            "compiled": error is None,
            "correctness": error is None,
            "error_code": "VALIDATION_ERROR" if error else None,
            "error_message": error,
        }

    monkeypatch.setattr(kernelgym_client, "_request_json", fake_kernelgym)
    monkeypatch.setattr(sys, "argv", [
        "kernelgym_final_client.py", "--url", "http://kernelgym",
        "--reference", str(reference), "--kernel", str(kernel),
        "--output", str(output), "--entry-point", entry_point, "--summary",
    ])

    assert kernelgym_client.main() == (1 if expected_error else 0)
    result = json.loads(output.read_text(encoding="utf-8"))
    summary = json.loads(capsys.readouterr().out)
    assert result["correctness"] is (expected_error is None)
    if expected_error:
        assert expected_error in summary["error"]
    assert reference.read_text(encoding="utf-8") == "class Model: pass\n"
