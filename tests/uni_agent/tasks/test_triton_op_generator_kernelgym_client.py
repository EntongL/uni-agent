from __future__ import annotations

import pytest

from uni_agent.tasks.triton_op_generator import kernelgym_client


def _evaluate_kwargs(**overrides):
    values = {
        "url": "http://kernelgym",
        "task_id": "task-1",
        "reference_code": "class Model: pass\n",
        "kernel_code": "class Model: pass\n",
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
