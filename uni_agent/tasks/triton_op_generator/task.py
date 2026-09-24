"""KernelGYM-backed task for generating Ascend Triton operators."""

from __future__ import annotations

import copy
import json
import logging
import re
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

from pydantic import Field, field_validator

from uni_agent.agents.claude_code.agent import ClaudeCodeAgent

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from .reward import score_kernelgym_result

logger = logging.getLogger(__name__)


def _container_path(value: str, *, field: str, allow_root: bool = False) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or (not allow_root and path == PurePosixPath("/")):
        raise ValueError(f"{field} must be a non-root absolute POSIX path without '..'")
    return str(path)


def _episode_label(instance_id: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance_id).strip("._")
    return label[:80] or "kernel"


class TritonOpGeneratorTaskConfig(TaskConfig):
    """Runtime contract for one AscendKernelBench operator-generation episode."""

    name: str = "triton_op_generator"
    agent_workdir: str = Field(
        default="/workspace/AscendKernelBench/agent",
        description="AscendKernelBench agent directory containing the Claude skills.",
    )
    episode_root: str = Field(
        default="/tmp/uni-agent-triton-episodes",
        description="Task-owned root for isolated reference, candidate, and final-eval files.",
    )
    kernelgym_url: str = Field(
        default="http://127.0.0.1:8082",
        description="KernelGYM API URL as visible from inside the resident sandbox.",
    )
    kernelgym_toolkit: str = Field(default="sandbox_v3")
    kernelgym_backend: str = Field(default="triton")
    entry_point: str = Field(default="Model")
    correctness_trials: int = Field(default=5, ge=1)
    performance_trials: int = Field(default=100, ge=1)
    evaluation_timeout: float = Field(default=900.0, gt=0)
    evaluation_poll_interval: float = Field(default=2.0, gt=0)
    cleanup_episode: bool = Field(
        default=True,
        description="Remove only the task-owned episode directory when the run ends.",
    )
    missing_kernel_retries: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Resume Claude Code this many times if it exits successfully without writing kernel_code.py.",
    )

    @field_validator("agent_workdir", "episode_root")
    @classmethod
    def _validate_container_path(cls, value: str, info) -> str:
        return _container_path(value, field=info.field_name)

    @field_validator("kernelgym_url")
    @classmethod
    def _validate_kernelgym_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("kernelgym_url must start with http:// or https://")
        return value.rstrip("/")

    @field_validator("kernelgym_toolkit", "kernelgym_backend", "entry_point")
    @classmethod
    def _validate_non_empty(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must be non-empty")
        return value


@register_task("triton_op_generator")
class TritonOpGeneratorTask(Task):
    """Run the packaged Claude workflow, then independently score its final kernel."""

    config_model = TritonOpGeneratorTaskConfig

    async def run(self) -> TaskResult:
        cfg: TritonOpGeneratorTaskConfig = self.config  # type: ignore[assignment]
        instance_id, operator_src = self._required_metadata(cfg.metadata)
        episode_dir = f"{cfg.episode_root}/{_episode_label(instance_id)}-{uuid.uuid4().hex}"
        reference_path = f"{episode_dir}/reference.py"
        output_dir = f"{episode_dir}/output"
        kernel_path = f"{output_dir}/kernel_code.py"
        generated_impl_path = f"{output_dir}/generated_impl.json"
        evaluator_path = f"{episode_dir}/kernelgym_final_client.py"
        evaluation_path = f"{episode_dir}/final-eval/result.json"

        async with self.build_sandbox() as sandbox:
            try:
                agent_workdir = await sandbox.exec(["test", "-d", cfg.agent_workdir])
                if agent_workdir.exit_code != 0:
                    raise RuntimeError(
                        f"AscendKernelBench agent workdir {cfg.agent_workdir!r} is unavailable: "
                        f"{agent_workdir.stderr.strip()}"
                    )
                await self._prepare_episode(sandbox, episode_dir, output_dir, reference_path, operator_src)
                agent_result, agent_info = await self._run_agent(
                    sandbox=sandbox,
                    cfg=cfg,
                    reference_path=reference_path,
                    output_dir=output_dir,
                    kernel_path=kernel_path,
                )

                kernel_exists = await self._file_exists(sandbox, kernel_path)
                generated_impl_exists = await self._file_exists(sandbox, generated_impl_path)
                if not kernel_exists:
                    logger.info("Triton operator task %s produced no kernel_code.py", instance_id)
                    return TaskResult(
                        reward=0.0,
                        accuracy=0.0,
                        finished=False,
                        extra_info={
                            "instance_id": instance_id,
                            "resolved": False,
                            "eval_completed": False,
                            "error_message": "agent did not produce output/kernel_code.py",
                            "generated_impl_present": generated_impl_exists,
                            "agent": agent_info,
                        },
                    )

                logger.info(
                    "Triton operator task %s produced %s; running final KernelGYM evaluation",
                    instance_id,
                    kernel_path,
                )
                # The reference is model-visible during the episode. Restore the frozen
                # source before final scoring so edits to it cannot alter the reward.
                await sandbox.write_file(reference_path, operator_src)
                await sandbox.write_file(evaluator_path, self._kernelgym_client_source())
                result = await self._evaluate_final_kernel(
                    sandbox=sandbox,
                    cfg=cfg,
                    instance_id=instance_id,
                    reference_path=reference_path,
                    kernel_path=kernel_path,
                    evaluator_path=evaluator_path,
                    evaluation_path=evaluation_path,
                )
                result.update(
                    {
                        "instance_id": instance_id,
                        "generated_impl_present": generated_impl_exists,
                        "agent": agent_info,
                    }
                )
                return TaskResult(
                    reward=result.pop("reward"),
                    accuracy=result.pop("accuracy"),
                    finished=agent_result.finished if agent_result is not None else False,
                    extra_info=result,
                )
            finally:
                if cfg.cleanup_episode:
                    await sandbox.exec(["rm", "-rf", "--", episode_dir])

    @staticmethod
    def _required_metadata(metadata: dict[str, Any]) -> tuple[str, str]:
        instance_id = metadata.get("instance_id")
        operator_src = metadata.get("operator_src")
        if not isinstance(instance_id, str) or not instance_id.strip():
            raise ValueError("triton_op_generator metadata requires a non-empty string 'instance_id'")
        if not isinstance(operator_src, str) or not operator_src.strip():
            raise ValueError("triton_op_generator metadata requires a non-empty string 'operator_src'")
        return instance_id, operator_src

    @staticmethod
    async def _prepare_episode(sandbox, episode_dir: str, output_dir: str, reference_path: str, operator_src: str) -> None:
        prepared = await sandbox.exec(["mkdir", "-p", output_dir])
        if prepared.exit_code != 0:
            raise RuntimeError(f"failed to create task episode directory {episode_dir!r}: {prepared.stderr.strip()}")
        await sandbox.write_file(reference_path, operator_src)

    async def _run_agent(
        self,
        *,
        sandbox,
        cfg: TritonOpGeneratorTaskConfig,
        reference_path: str,
        output_dir: str,
        kernel_path: str,
    ):
        messages = self._messages_with_runtime_paths(cfg.prompt, reference_path=reference_path, output_dir=output_dir)
        agent = self.build_agent()
        if cfg.missing_kernel_retries and not isinstance(agent, ClaudeCodeAgent):
            raise ValueError("missing_kernel_retries requires the claude_code agent")
        claude_session_id = str(uuid.uuid4()) if cfg.missing_kernel_retries else None
        attempts: list[dict[str, Any]] = []
        try:
            if claude_session_id is None:
                agent_result = await agent.run(sandbox=sandbox, messages=messages, workdir=cfg.agent_workdir)
            else:
                agent_result = await agent.run_session(
                    sandbox=sandbox,
                    messages=messages,
                    session_id=claude_session_id,
                    workdir=cfg.agent_workdir,
                )
            attempts.append({"stage": "initial", **agent_result.info})
            for attempt in range(1, cfg.missing_kernel_retries + 1):
                if not agent_result.finished or await self._file_exists(sandbox, kernel_path):
                    break
                logger.warning(
                    "Triton operator task: Claude exited without %s; resuming session %s (%d/%d)",
                    kernel_path,
                    claude_session_id,
                    attempt,
                    cfg.missing_kernel_retries,
                )
                agent_result = await agent.resume_session(
                    sandbox=sandbox,
                    prompt=self._missing_kernel_prompt(reference_path, kernel_path),
                    session_id=claude_session_id,
                    workdir=cfg.agent_workdir,
                )
                attempts.append({"stage": f"continuation_{attempt}", **agent_result.info})
        except Exception as exc:
            logger.exception("Triton operator agent failed before final evaluation")
            return None, {"error": f"{type(exc).__name__}: {exc}", "attempts": attempts}
        return agent_result, {"error": None, **agent_result.info, "attempts": attempts}

    @staticmethod
    def _missing_kernel_prompt(reference_path: str, kernel_path: str) -> str:
        return (
            "The previous turn ended before the required final artifact was written. "
            f"The file {kernel_path} does not exist. Continue this same task now: "
            "use the packaged triton-op-coding workflow and write a valid Triton-Ascend "
            f"implementation defining Model to {kernel_path}. The reference at "
            f"{reference_path} is immutable; reuse your existing sketch if useful. "
            "Do not stop after describing the next step. Confirm the file exists before finishing. "
            "Use KernelGYM for evaluation; do not run verify.py or benchmark.py."
        )

    @staticmethod
    def _messages_with_runtime_paths(
        messages: list[dict[str, Any]], *, reference_path: str, output_dir: str
    ) -> list[dict[str, Any]]:
        rendered = copy.deepcopy(messages)
        user_messages = [message for message in rendered if message.get("role") == "user"]
        if len(user_messages) != 1:
            raise ValueError("triton_op_generator requires exactly one user prompt for the Claude Code agent")
        user_message = user_messages[0]
        content = user_message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("triton_op_generator user prompt must contain non-empty text")
        user_message["content"] = (
            f"{content.rstrip()}\n\n"
            "Runtime paths for this episode (do not modify the reference):\n"
            f"REFERENCE_PATH={reference_path}\n"
            f"OUTPUT_DIR={output_dir}\n\n"
            "Write the final Triton implementation to OUTPUT_DIR/kernel_code.py. "
            "Use the packaged triton-ascend-kernelgen workflow and KernelGYM only; "
            "do not run verify.py or benchmark.py."
        )
        return rendered

    async def _evaluate_final_kernel(
        self,
        *,
        sandbox,
        cfg: TritonOpGeneratorTaskConfig,
        instance_id: str,
        reference_path: str,
        kernel_path: str,
        evaluator_path: str,
        evaluation_path: str,
    ) -> dict[str, Any]:
        command = [
            "python3",
            evaluator_path,
            "--url",
            cfg.kernelgym_url,
            "--task-id",
            f"uni-agent-{_episode_label(instance_id)}-{uuid.uuid4().hex}",
            "--reference",
            reference_path,
            "--kernel",
            kernel_path,
            "--output",
            evaluation_path,
            "--entry-point",
            cfg.entry_point,
            "--backend",
            cfg.kernelgym_backend,
            "--toolkit",
            cfg.kernelgym_toolkit,
            "--correctness-trials",
            str(cfg.correctness_trials),
            "--performance-trials",
            str(cfg.performance_trials),
            "--timeout",
            str(cfg.evaluation_timeout),
            "--poll-interval",
            str(cfg.evaluation_poll_interval),
            "--enable-profiling",
            "--enable-triton-detection",
        ]
        response = await sandbox.exec(command, timeout=cfg.evaluation_timeout + 60)
        if not await self._file_exists(sandbox, evaluation_path):
            detail = (response.stderr or response.stdout or "no evaluator output").strip()[-2000:]
            raise RuntimeError(f"final KernelGYM invocation produced no result file: {detail}")
        try:
            payload = json.loads((await sandbox.read_file(evaluation_path)).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("final KernelGYM result is not valid UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("final KernelGYM result must be a JSON object")

        result = score_kernelgym_result(payload)
        result["evaluation_command"] = {
            "exit_code": response.exit_code,
            "stdout_tail": (response.stdout or "")[-2000:],
            "stderr_tail": (response.stderr or "")[-2000:],
        }
        return result

    @staticmethod
    async def _file_exists(sandbox, path: str) -> bool:
        return (await sandbox.exec(["test", "-f", path])).exit_code == 0

    @staticmethod
    def _kernelgym_client_source() -> str:
        return Path(__file__).with_name("kernelgym_client.py").read_text(encoding="utf-8")
