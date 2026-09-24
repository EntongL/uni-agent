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
            return ExecResult(0 if command[2] in self.files else 1, "", "")
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
    async def run(self, *, sandbox, messages, workdir=None):
        prompt = messages[0]["content"]
        output_dir = next(line.split("=", 1)[1] for line in prompt.splitlines() if line.startswith("OUTPUT_DIR="))
        reference_path = next(line.split("=", 1)[1] for line in prompt.splitlines() if line.startswith("REFERENCE_PATH="))
        await sandbox.write_file(reference_path, "tampered reference")
        await sandbox.write_file(f"{output_dir}/kernel_code.py", "class ModelNew: pass\n")
        await sandbox.write_file(f"{output_dir}/generated_impl.json", "{}")
        return AgentResult(info={"exit_code": 0}, finished=True)


def _config(*, cleanup_episode: bool = False, missing_kernel_retries: int = 0) -> TritonOpGeneratorTaskConfig:
    return TritonOpGeneratorTaskConfig(
        sandbox={"provider": "local"},
        agent={"name": "claude_code"},
        prompt=[{"role": "user", "content": "Generate a kernel."}],
        metadata={"instance_id": "level1/example", "operator_src": "class Model: pass\n"},
        agent_workdir="/workspace/AscendKernelBench/agent",
        episode_root="/tmp/uni-agent-triton-episodes",
        cleanup_episode=cleanup_episode,
        missing_kernel_retries=missing_kernel_retries,
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
            "status": "completed",
            "compiled": True,
            "correctness": True,
            "decoy_kernel": False,
            "speedup": 1.25,
        }
    )
    task.build_sandbox = lambda: sandbox  # type: ignore[method-assign]
    task.build_agent = lambda: FakeAgent()  # type: ignore[method-assign]

    result = asyncio.run(task.run())

    assert result.reward == 1.0
    assert result.accuracy == 1.0
    assert result.finished is True
    assert result.extra_info["generated_impl_present"] is True
    assert result.extra_info["speedup"] == 1.25
    assert any(command[:1] == ["python3"] for command in sandbox.commands)
    assert any(content == b"class Model: pass\n" for content in sandbox.files.values())


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
            "status": "completed",
            "compiled": True,
            "correctness": True,
            "decoy_kernel": False,
            "speedup": 1.25,
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
                await sandbox.write_file(kernel_path, "class Model: pass\n")
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
