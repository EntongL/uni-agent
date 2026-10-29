from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from uni_agent.agents.base import AgentResult
from uni_agent.agents.claude_code.agent import ClaudeCodeAgent, ClaudeCodeConfig
from uni_agent.sandbox.base import ExecResult
from uni_agent.tasks import get_task
from uni_agent.tasks.triton_op_generator.task import TritonOpGeneratorTask, TritonOpGeneratorTaskConfig


@dataclass
class FakeSandbox:
    files: dict[str, bytes] = field(default_factory=dict)
    commands: list[list[str]] = field(default_factory=list)
    final_payload: dict[str, Any] | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return None

    async def exec(self, argv, *, timeout=None, workdir=None, env=None):
        command = [str(arg) for arg in argv]
        self.commands.append(command)
        if command[:2] == ["test", "-f"]:
            path = command[2]
            present = path in self.files or path.startswith("/workspace/AscendKernelBench/")
            return ExecResult(0 if present else 1, "", "")
        if command and command[0] == "python3":
            output = command[command.index("--output") + 1]
            assert self.final_payload is not None
            self.files[output] = json.dumps(self.final_payload).encode("utf-8")
        return ExecResult(0, "evaluator stdout", "")

    async def write_file(self, path, content):
        self.files[path] = content.encode("utf-8") if isinstance(content, str) else content

    async def read_file(self, path):
        return self.files[path]


class FakeAgent:
    def __init__(self):
        self.workdir = None

    async def run(self, *, sandbox, messages, workdir=None):
        self.workdir = workdir
        prompt = messages[0]["content"]
        output_dir = next(line.split("=", 1)[1] for line in prompt.splitlines() if line.startswith("OUTPUT_DIR="))
        reference_path = next(line.split("=", 1)[1] for line in prompt.splitlines() if line.startswith("REFERENCE_PATH="))
        await sandbox.write_file(reference_path, "tampered reference")
        await sandbox.write_file(f"{output_dir}/kernel_code.py", "class ModelNew: pass\n")
        await sandbox.write_file(f"{output_dir}/generated_impl.json", "{}")
        return AgentResult(info={"exit_code": 0}, finished=True)


def _config(
    *,
    cleanup_episode: bool = False,
    missing_kernel_retries: int = 0,
    use_agent_skills: bool = True,
) -> TritonOpGeneratorTaskConfig:
    return TritonOpGeneratorTaskConfig(
        sandbox={"provider": "local"},
        agent={"name": "claude_code"},
        prompt=[{"role": "user", "content": "Generate a kernel."}],
        metadata={"instance_id": "level1/example", "operator_src": "class Model: pass\n"},
        agent_workdir="/workspace/AscendKernelBench/agent",
        episode_root="/tmp/uni-agent-triton-episodes",
        cleanup_episode=cleanup_episode,
        missing_kernel_retries=missing_kernel_retries,
        use_agent_skills=use_agent_skills,
    )


@pytest.mark.cpu
@pytest.mark.level0
def test_registered_triton_op_generator_task_builds_from_serialized_config():
    task = get_task(_config().model_dump(mode="json"))

    assert isinstance(task, TritonOpGeneratorTask)


@pytest.mark.cpu
@pytest.mark.level0
def test_task_restores_reference_and_uses_independent_final_evaluator():
    task = TritonOpGeneratorTask(_config())
    sandbox = FakeSandbox(
        final_payload={
            "eval_completed": True,
            "compile_ok": True,
            "correctness_ok": True,
            "passed_cases": 3,
            "total_cases": 3,
            "failed_cases": 0,
            "speedup": 2.0,
        }
    )
    agent = FakeAgent()
    task.build_sandbox = lambda: sandbox  # type: ignore[method-assign]
    task.build_agent = lambda: agent  # type: ignore[method-assign]

    result = asyncio.run(task.run())

    assert result.reward == 1.0
    assert result.accuracy == 1.0
    assert result.finished is True
    assert result.extra_info["generated_impl_present"] is True
    assert result.extra_info["speedup"] == 2.0
    assert any(command[:1] == ["python3"] for command in sandbox.commands)
    assert any(content == b"class Model: pass\n" for content in sandbox.files.values())
    evaluation = next(command for command in sandbox.commands if command[:1] == ["python3"])
    assert "--verify-script" in evaluation
    assert evaluation[evaluation.index("--op-name") + 1] == "example"
    assert sandbox.files[evaluation[evaluation.index("--kernel") + 1]] == b"class ModelNew: pass\n"
    assert agent.workdir == "/workspace/AscendKernelBench/agent"
    assert not any(command[:3] == ["ln", "-sfn", "--"] for command in sandbox.commands)
    assert not any(path.endswith("/CLAUDE.md") for path in sandbox.files)


@pytest.mark.cpu
@pytest.mark.level0
def test_task_returns_zero_when_agent_does_not_create_candidate():
    task = TritonOpGeneratorTask(_config())
    sandbox = FakeSandbox()

    class EmptyAgent:
        async def run(self, **_):
            return AgentResult(finished=False)

    task.build_sandbox = lambda: sandbox  # type: ignore[method-assign]
    task.build_agent = lambda: EmptyAgent()  # type: ignore[method-assign]

    result = asyncio.run(task.run())

    assert result.reward == 0.0
    assert result.accuracy == 0.0
    assert result.finished is False
    assert "kernel_code.py" in result.extra_info["error_message"]
    assert not any(command[:1] == ["python3"] for command in sandbox.commands)


@pytest.mark.cpu
@pytest.mark.level0
def test_clean_agent_exit_without_kernel_is_not_marked_finished():
    task = TritonOpGeneratorTask(_config())
    sandbox = FakeSandbox()

    class EmptyAgent:
        async def run(self, **_):
            return AgentResult(info={"exit_code": 0}, finished=True)

    task.build_sandbox = lambda: sandbox  # type: ignore[method-assign]
    task.build_agent = lambda: EmptyAgent()  # type: ignore[method-assign]

    result = asyncio.run(task.run())

    assert result.reward == 0.0
    assert result.finished is False
    assert len(result.extra_info["agent"]["attempts"]) == 1


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("write_on_resume", [False, True])
def test_missing_kernel_resumes_same_claude_session_once(write_on_resume):
    task = TritonOpGeneratorTask(_config(missing_kernel_retries=1))
    sandbox = FakeSandbox(
        final_payload={
            "eval_completed": True,
            "compile_ok": True,
            "correctness_ok": True,
            "passed_cases": 3,
            "total_cases": 3,
            "failed_cases": 0,
            "speedup": 2.0,
        }
    )

    class ContinuableAgent(ClaudeCodeAgent):
        def __init__(self):
            super().__init__(ClaudeCodeConfig())
            self.session_id = None
            self.resume_calls = 0

        async def run_session(self, *, sandbox, messages, session_id, workdir=None):
            self.session_id = session_id
            return AgentResult(info={"exit_code": 0}, finished=True)

        async def resume_session(self, *, sandbox, prompt, session_id, workdir=None):
            assert session_id == self.session_id
            assert "kernel_code.py" in prompt
            assert "Do not stop after describing" in prompt
            self.resume_calls += 1
            if write_on_resume:
                kernel_path = prompt.split("The file ", 1)[1].split(" does not exist", 1)[0]
                await sandbox.write_file(kernel_path, "class ModelNew: pass\n")
            return AgentResult(info={"exit_code": 0}, finished=True)

    agent = ContinuableAgent()
    task.build_sandbox = lambda: sandbox  # type: ignore[method-assign]
    task.build_agent = lambda: agent  # type: ignore[method-assign]

    result = asyncio.run(task.run())

    assert agent.resume_calls == 1
    assert result.finished is write_on_resume
    assert result.reward == (1.0 if write_on_resume else 0.0)
    assert len(result.extra_info["agent"]["attempts"]) == 2
    assert any(command[:2] == ["test", "-f"] for command in sandbox.commands)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("stage", ["initial", "missing", "context", "repair"])
def test_all_agent_stages_keep_reference_and_candidate_entry_points_separate(stage):
    cfg = _config()
    paths = {
        "reference_path": "/tmp/episode/reference.py",
        "kernel_path": "/tmp/episode/output/kernel_code.py",
        "evaluator_path": "/tmp/episode/kernelgym_final_client.py",
    }
    instructions = "/tmp/episode/INSTRUCTIONS.md"
    if stage == "initial":
        prompt = TritonOpGeneratorTask._messages_with_runtime_paths(
            cfg.prompt, reference_path=paths["reference_path"],
            output_dir="/tmp/episode/output", instructions_path=instructions,
        )[0]["content"]
    elif stage == "missing":
        prompt = TritonOpGeneratorTask._missing_kernel_prompt(
            instructions, paths["reference_path"], paths["kernel_path"],
        )
    elif stage == "context":
        prompt = TritonOpGeneratorTask._context_thrash_prompt(
            instructions, paths["reference_path"], paths["kernel_path"],
        )
    else:
        prompt = TritonOpGeneratorTask._evaluation_repair_prompt(
            cfg=cfg, reference_path=paths["reference_path"], kernel_path=paths["kernel_path"],
            instructions_path=instructions,
            result={
                "status": "failed", "compiled": False, "correctness": False,
                "error_code": "VALIDATION_ERROR",
                "error_message": "Kernel code validation failed: Code must contain a 'ModelNew' class",
            },
        )
    workflow = TritonOpGeneratorTask._workflow_instructions(cfg, **paths)
    assert "ModelNew" in prompt
    assert instructions in prompt
    assert "--entry-point ModelNew" not in prompt
    assert "--entry-point ModelNew" not in workflow
    assert "CLAUDE.md" in workflow or "agent markdown" in workflow
    assert "verifier" in workflow
    assert "torch.nn.Module subclass" in workflow
    assert "super().__init__()" in workflow
    assert "Pass torch.Tensor objects directly" in workflow
    assert "Do not shadow npu" in workflow
    assert "triton-op-verifier" in workflow
    assert "triton-latency-optimizer" in workflow
    assert "Do not preload .claude/skills" not in workflow
    assert "KernelGYM" not in workflow
    assert TritonOpGeneratorTask._seed_kernel_rejected("import triton.module\n")
    assert TritonOpGeneratorTask._seed_kernel_rejected("def __call__(self):\n    return A * s\n")
    assert TritonOpGeneratorTask._seed_kernel_rejected("@triton.jit\ndef kernel():\n    pass\n")
    assert not TritonOpGeneratorTask._seed_kernel_rejected(
        "@triton.jit\ndef kernel():\n    pass\nclass ModelNew:\n    def forward(self, A, s):\n        grid = (256,)\n"
    )


@pytest.mark.cpu
@pytest.mark.level0
def test_repair_prompt_retains_failed_npu_call_and_decoy_status():
    prompt = TritonOpGeneratorTask._evaluation_repair_prompt(
        cfg=_config(), reference_path="/tmp/episode/reference.py",
        kernel_path="/tmp/episode/output/kernel_code.py",
        instructions_path="/tmp/episode/INSTRUCTIONS.md",
        result={
            "status": "completed", "compiled": True, "correctness": False,
            "passed_cases": 0, "total_cases": 3, "error_code": "RUNTIME_ERROR",
            "error_message": (
                'Traceback (most recent call last):\n'
                '  File "correctness.py", line 445, in check\n'
                '    model_new = new_model_instance.npu(device=device)\n'
                "TypeError: 'NoneType' object is not callable"
            ),
        },
    )
    assert "new_model_instance.npu(device=device)" in prompt
    assert "passed=0/3" in prompt
    assert "verifier Skill" in prompt
    assert "self.npu = None" in TritonOpGeneratorTask._workflow_instructions(
        _config(),
        reference_path="/tmp/episode/reference.py",
        kernel_path="/tmp/episode/output/kernel_code.py",
        evaluator_path="/tmp/episode/kernelgym_final_client.py",
    )


@pytest.mark.cpu
@pytest.mark.level0
def test_skills_can_be_disabled_for_the_legacy_prompt():
    cfg = _config(use_agent_skills=False)
    workflow = TritonOpGeneratorTask._workflow_instructions(
        cfg,
        reference_path="/tmp/episode/reference.py",
        kernel_path="/tmp/episode/output/kernel_code.py",
        evaluator_path="/tmp/episode/kernelgym_final_client.py",
    )
    prompt = TritonOpGeneratorTask._messages_with_runtime_paths(
        cfg.prompt,
        reference_path="/tmp/episode/reference.py",
        output_dir="/tmp/episode/output",
        instructions_path="/tmp/episode/INSTRUCTIONS.md",
        use_agent_skills=False,
    )[0]["content"]

    assert "Do not preload .claude/skills" in workflow
    assert "Follow that markdown's full skill workflow" not in prompt


@pytest.mark.cpu
@pytest.mark.level0
def test_disabled_skills_keep_claude_workdir_on_agent_checkout():
    task = TritonOpGeneratorTask(_config(use_agent_skills=False))
    sandbox = FakeSandbox(
        final_payload={
            "eval_completed": True,
            "compile_ok": True,
            "correctness_ok": True,
            "passed_cases": 3,
            "total_cases": 3,
            "failed_cases": 0,
            "speedup": 2.0,
        }
    )
    agent = FakeAgent()
    task.build_sandbox = lambda: sandbox  # type: ignore[method-assign]
    task.build_agent = lambda: agent  # type: ignore[method-assign]

    result = asyncio.run(task.run())

    assert result.reward == 1.0
    assert agent.workdir == "/workspace/AscendKernelBench/agent"
    assert not any(command[:3] == ["ln", "-sfn", "--"] for command in sandbox.commands)
    assert not any(path.endswith("/CLAUDE.md") for path in sandbox.files)
