from __future__ import annotations

import pytest

from uni_agent.tasks.triton_op_generator.reward import score_kernelgym_result


@pytest.mark.cpu
@pytest.mark.level0
def test_score_kernelgym_result_requires_completed_correct_non_decoy_kernel():
    result = score_kernelgym_result(
        {
            "status": "completed",
            "compiled": True,
            "correctness": True,
            "decoy_kernel": False,
            "reference_runtime": 1.2,
            "kernel_runtime": 0.6,
            "speedup": 2.0,
            "metadata": {"case_summary": {"passed_cases": 3, "total_cases": 3}},
        }
    )

    assert result["reward"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["resolved"] is True
    assert result["case_summary"] == {"passed_cases": 3, "total_cases": 3}


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize(
    ("payload", "reward"),
    [
        ({"status": "completed", "compiled": True, "correctness": False}, 0.3),
        ({"status": "completed", "compiled": False, "correctness": False}, 0.1),
        ({"status": "failed", "compiled": True, "correctness": True}, 0.0),
        ({"status": "completed", "compiled": True, "correctness": True, "decoy_kernel": True}, 0.0),
    ],
)
def test_score_kernelgym_result_stages_partial_credit(payload, reward):
    result = score_kernelgym_result(payload)

    assert result["reward"] == reward
    assert result["accuracy"] == 0.0
    assert result["resolved"] is False


@pytest.mark.cpu
@pytest.mark.level0
def test_score_kernelgym_result_rejects_missing_status():
    with pytest.raises(ValueError, match="status"):
        score_kernelgym_result({"compiled": True, "correctness": True})
