"""AscendKernelBench agent-skill task for generating Triton operators."""

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
from .kernelgym_client import summarize_error
from .reward import score_verifier_result

logger = logging.getLogger(__name__)

_CANDIDATE_RUNTIME_CONTRACT = (
    "Define ModelNew as a torch.nn.Module subclass with forward matching the reference. "
    "If overriding __init__, call super().__init__(). The evaluator calls "
    "model.npu(device=...) and model(*inputs); inherit these methods from nn.Module. "
    "Do not shadow npu, to, or __call__ with attributes such as self.npu = None. "
    "Pass torch.Tensor objects directly to Triton pointer parameters, not integer "
    "addresses from tensor.data_ptr(). Keep the operator computation in Triton. "
)

# Fixed-grid elementwise kernel. KernelGYM accepts this shape for the smoke
# operator (matrix-scalar multiply) and rejects min()-wrapped grids as decoys.
_SEEDED_KERNEL_SOURCE = """\
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def kernel(in_ptr, out_ptr, scalar, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    for block_id in range(pid, blocks, nprog):
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
        tl.store(out_ptr + offsets, vals * scalar, mask=mask)

class ModelNew(nn.Module):
    def forward(self, A, s):
        source = A.contiguous()
        output = torch.empty_like(source)
        grid = (256,)
        kernel[grid](source, output, s, source.numel(), BLOCK_SIZE=4096)
        return output
"""


def _container_path(value: str, *, field: str, allow_root: bool = False) -> str:
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or (not allow_root and path == PurePosixPath("/")):
        raise ValueError(f"{field} must be a non-root absolute POSIX path without '..'")
    return str(path)


def _episode_label(instance_id: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", instance_id).strip("._")
    return label[:80] or "kernel"


def _operator_name(instance_id: str, metadata: dict[str, Any]) -> str:
    raw = metadata.get("op_name") or instance_id.rsplit("/", 1)[-1]
    label = re.sub(r"[^A-Za-z0-9_]+", "_", str(raw)).strip("_")
    if label and label[0].isdigit():
        label = f"op_{label}"
    return label[:80] or "operator"


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
    entry_point: str = Field(
        default="Model",
        description="Reference model entry point; KernelGYM loads the candidate as ModelNew.",
    )
    correctness_trials: int = Field(default=5, ge=1)
    performance_trials: int = Field(default=100, ge=1)
    evaluation_timeout: float = Field(default=900.0, gt=0)
    evaluation_poll_interval: float = Field(default=2.0, gt=0)
    cleanup_episode: bool = Field(
        default=True,
        description="Remove only the task-owned episode directory when the run ends.",
    )
    retain_failed_episode: bool = Field(
        default=False,
        description="Keep the task-owned episode directory when generation or final evaluation fails.",
    )
    missing_kernel_retries: int = Field(
        default=0,
        ge=0,
        le=3,
        description="Resume Claude Code this many times if it exits successfully without writing kernel_code.py.",
    )
    context_recovery_retries: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Start a fresh Claude Code coding session after a context exhaustion exit.",
    )
    evaluation_retries: int = Field(
        default=0,
        ge=0,
        le=2,
        description="Start a fresh coding session to repair a candidate after a completed, failed verifier evaluation.",
    )
    seed_reference_kernel: bool = Field(
        default=False,
        description=(
            "Write a known-good fixed-grid Triton kernel to kernel_code.py before the "
            "agent starts. If the agent then leaves a missing, decoy, or non-Triton file, "
            "restore that kernel before the independent KernelGYM score."
        ),
    )
    agent_verification_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum verifier-skill repair attempts requested inside one agent session.",
    )
    use_agent_skills: bool = Field(
        default=True,
        description=(
            "Run Claude Code from the AscendKernelBench agent workdir so it follows "
            "that checkout's CLAUDE.md / AGENTS.md and every skill it names."
        ),
    )
    skills_dir: str | None = Field(
        default=None,
        description=(
            "Absolute POSIX path to the Claude skills directory inside the sandbox. "
            "Defaults to <agent_workdir>/.claude/skills."
        ),
    )
    coding_skill: str = Field(
        default="triton-op-coding",
        description="Optional hint; the agent markdown is the source of truth for skills.",
    )
    verifier_skill: str = Field(
        default="triton-op-verifier",
        description="Skill whose scripts/verify.py and scripts/benchmark.py produce the reward.",
    )
    optimizer_skill: str = Field(
        default="triton-latency-optimizer",
        description="Optional hint; the agent markdown decides when to optimize.",
    )
    verify_script: str | None = Field(
        default=None,
        description="Absolute path to verify.py. Defaults to <skills_dir>/<verifier_skill>/scripts/verify.py.",
    )
    benchmark_script: str | None = Field(
        default=None,
        description="Absolute path to benchmark.py. Defaults to <skills_dir>/<verifier_skill>/scripts/benchmark.py.",
    )
    framework: str = Field(default="torch")
    dsl: str = Field(default="triton_ascend")
    backend: str = Field(default="ascend")
    triton_impl_name: str = Field(default="triton_ascend_impl")
    target_speedup: float = Field(default=2.0, gt=0)
    benchmark_warmup: int = Field(default=5, ge=0)
    benchmark_repeats: int = Field(default=10, ge=1)

    @field_validator("agent_workdir", "episode_root")
    @classmethod
    def _validate_container_path(cls, value: str, info) -> str:
        return _container_path(value, field=info.field_name)

    @field_validator("skills_dir", "verify_script", "benchmark_script")
    @classmethod
    def _validate_optional_container_path(cls, value: str | None, info) -> str | None:
        if value is None:
            return None
        return _container_path(value, field=info.field_name)

    @property
    def resolved_skills_dir(self) -> str:
        return self.skills_dir or f"{self.agent_workdir}/.claude/skills"

    @property
    def resolved_verify_script(self) -> str:
        return self.verify_script or f"{self.resolved_skills_dir}/{self.verifier_skill}/scripts/verify.py"

    @property
    def resolved_benchmark_script(self) -> str:
        return self.benchmark_script or f"{self.resolved_skills_dir}/{self.verifier_skill}/scripts/benchmark.py"

    @field_validator("kernelgym_url")
    @classmethod
    def _validate_kernelgym_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("kernelgym_url must start with http:// or https://")
        return value.rstrip("/")

    @field_validator(
        "kernelgym_toolkit",
        "kernelgym_backend",
        "entry_point",
        "coding_skill",
        "verifier_skill",
        "optimizer_skill",
        "framework",
        "dsl",
        "backend",
        "triton_impl_name",
    )
    @classmethod
    def _validate_non_empty(cls, value: str, info) -> str:
        if not value.strip():
            raise ValueError(f"{info.field_name} must be non-empty")
        return value.strip()


@register_task("triton_op_generator")
class TritonOpGeneratorTask(Task):
    """Run the packaged Claude workflow, then independently score its final kernel."""

    config_model = TritonOpGeneratorTaskConfig

    async def run(self) -> TaskResult:
        cfg: TritonOpGeneratorTaskConfig = self.config  # type: ignore[assignment]
        instance_id, operator_src = self._required_metadata(cfg.metadata)
        op_name = _operator_name(instance_id, cfg.metadata)
        episode_dir = f"{cfg.episode_root}/{_episode_label(instance_id)}-{uuid.uuid4().hex}"
        reference_path = f"{episode_dir}/reference.py"
        output_dir = f"{episode_dir}/output"
        kernel_path = f"{output_dir}/kernel_code.py"
        generated_impl_path = f"{output_dir}/generated_impl.json"
        evaluator_path = f"{episode_dir}/verifier_client.py"
        evaluation_path = f"{episode_dir}/final-eval/result.json"

        async with self.build_sandbox() as sandbox:
            retain_episode = cfg.retain_failed_episode
            try:
                agent_workdir = await sandbox.exec(["test", "-d", cfg.agent_workdir])
                if agent_workdir.exit_code != 0:
                    raise RuntimeError(
                        f"AscendKernelBench agent workdir {cfg.agent_workdir!r} is unavailable: "
                        f"{agent_workdir.stderr.strip()}"
                    )
                instructions_path = f"{episode_dir}/INSTRUCTIONS.md"
                await self._prepare_episode(
                    sandbox, episode_dir, output_dir, reference_path, operator_src,
                    instructions_path, cfg, evaluator_path,
                )
                await self._ensure_agent_skills(sandbox, cfg)
                claude_workdir = cfg.agent_workdir
                # The agent follows CLAUDE.md and calls the skill scripts itself.
                # Rewrite this helper before final scoring so agent edits cannot affect reward.
                await sandbox.write_file(evaluator_path, self._verifier_client_source())
                agent_result, agent_info = await self._run_agent(
                    sandbox=sandbox,
                    cfg=cfg,
                    reference_path=reference_path,
                    output_dir=output_dir,
                    kernel_path=kernel_path,
                    evaluator_path=evaluator_path,
                    instructions_path=instructions_path,
                    workdir=claude_workdir,
                )

                if cfg.seed_reference_kernel:
                    await self._restore_seed_kernel_if_needed(sandbox, kernel_path, instance_id)
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
                    "Triton operator task %s produced %s; running final skill verifier",
                    instance_id,
                    kernel_path,
                )
                # The reference is model-visible during the episode. Restore the frozen
                # source before final scoring so edits to it cannot alter the reward.
                await sandbox.write_file(reference_path, operator_src)
                await sandbox.write_file(evaluator_path, self._verifier_client_source())
                result = await self._evaluate_final_kernel(
                    sandbox=sandbox,
                    cfg=cfg,
                    instance_id=instance_id,
                    op_name=op_name,
                    reference_path=reference_path,
                    kernel_path=kernel_path,
                    evaluator_path=evaluator_path,
                    evaluation_path=evaluation_path,
                )
                for attempt in range(1, cfg.evaluation_retries + 1):
                    if result["resolved"] or not result["eval_completed"]:
                        break
                    previous_result_path = f"{episode_dir}/final-eval/attempt-{attempt - 1}.json"
                    saved = await sandbox.exec(["cp", "--", evaluation_path, previous_result_path])
                    if saved.exit_code != 0:
                        raise RuntimeError(
                            f"failed to preserve verifier result at {previous_result_path}: {saved.stderr.strip()}"
                        )
                    previous_kernel_path = f"{episode_dir}/final-eval/kernel-attempt-{attempt - 1}.py"
                    saved_kernel = await sandbox.exec(["cp", "--", kernel_path, previous_kernel_path])
                    if saved_kernel.exit_code != 0:
                        raise RuntimeError(
                            f"failed to preserve candidate at {previous_kernel_path}: {saved_kernel.stderr.strip()}"
                        )
                    logger.warning(
                        "Triton operator task %s: skill verifier rejected candidate "
                        "error_code=%s error=%r; starting fresh repair session (%d/%d)",
                        instance_id,
                        result["error_code"],
                        self._evaluation_error_excerpt(result["error_message"]),
                        attempt,
                        cfg.evaluation_retries,
                    )
                    try:
                        repair_agent = self.build_agent()
                        if not isinstance(repair_agent, ClaudeCodeAgent):
                            raise ValueError("verifier evaluation retries require the claude_code agent")
                        repair_prompt = self._evaluation_repair_prompt(
                            cfg=cfg,
                            reference_path=reference_path,
                            kernel_path=kernel_path,
                            instructions_path=instructions_path,
                            result=result,
                        )
                        previous_session_id = (
                            agent_result.info.get("claude_session_id")
                            if agent_result and agent_result.finished else None
                        )
                        if isinstance(previous_session_id, str) and previous_session_id:
                            repair_result = await repair_agent.resume_session(
                                sandbox=sandbox,
                                prompt=repair_prompt,
                                session_id=previous_session_id,
                                workdir=claude_workdir,
                            )
                        else:
                            repair_result = await repair_agent.run_session(
                                sandbox=sandbox,
                                messages=[{"role": "user", "content": repair_prompt}],
                                session_id=str(uuid.uuid4()),
                                workdir=claude_workdir,
                            )
                    except Exception as exc:
                        logger.exception("Triton operator task %s repair session failed", instance_id)
                        agent_info["attempts"].append({
                            "stage": f"evaluation_repair_{attempt}",
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        break
                    agent_result = repair_result
                    agent_info.update({"error": None, **repair_result.info})
                    agent_info["attempts"].append({
                        "stage": f"evaluation_repair_{attempt}", **repair_result.info,
                    })
                    # A fresh session may also hit the context limit. Score any
                    # file it left behind instead of depending on its exit code.
                    if not await self._file_exists(sandbox, kernel_path):
                        logger.warning(
                            "Triton operator task %s repair session removed %s",
                            instance_id,
                            kernel_path,
                        )
                        await sandbox.exec(["cp", "--", previous_kernel_path, kernel_path])
                        break
                    await sandbox.write_file(reference_path, operator_src)
                    cleared = await sandbox.exec(["rm", "-f", "--", evaluation_path])
                    if cleared.exit_code != 0:
                        raise RuntimeError(
                            f"failed to clear stale verifier result at {evaluation_path}: {cleared.stderr.strip()}"
                        )
                    result = await self._evaluate_final_kernel(
                        sandbox=sandbox,
                        cfg=cfg,
                        instance_id=instance_id,
                        op_name=op_name,
                        reference_path=reference_path,
                        kernel_path=kernel_path,
                        evaluator_path=evaluator_path,
                        evaluation_path=evaluation_path,
                    )
                if result["resolved"]:
                    retain_episode = False
                else:
                    # The full stack trace is kept in final-eval/result.json.
                    # Keep Ray's duplicated task/worker logs short and useful.
                    error_excerpt = self._evaluation_error_excerpt(result["error_message"])
                    logger.warning(
                        "Triton operator task %s final skill-verifier reward=%s: "
                        "status=%s compiled=%s correctness=%s "
                        "passed=%s/%s speedup=%s error_code=%s error=%r result_file=%s",
                        instance_id,
                        result["reward"],
                        result["status"],
                        result["compiled"],
                        result["correctness"],
                        result.get("passed_cases"),
                        result.get("total_cases"),
                        result.get("speedup"),
                        result["error_code"],
                        error_excerpt,
                        evaluation_path,
                    )
                # The full evaluator response remains in the episode file.
                # Keep TaskResult small for rollout storage and trainer logs.
                result["error_message"] = self._evaluation_error_excerpt(result["error_message"])
                result.update(
                    {
                        "instance_id": instance_id,
                        "op_name": op_name,
                        "generated_impl_present": await self._file_exists(sandbox, generated_impl_path),
                        "agent": agent_info,
                    }
                )
                return TaskResult(
                    reward=result.pop("reward"),
                    accuracy=result.pop("accuracy"),
                    finished=bool(result["resolved"] or (agent_result and agent_result.finished)),
                    extra_info=result,
                )
            finally:
                if cfg.cleanup_episode and not retain_episode:
                    await sandbox.exec(["rm", "-rf", "--", episode_dir])
                else:
                    logger.info("Triton operator task %s retained episode at %s", instance_id, episode_dir)

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
    async def _prepare_episode(
        sandbox, episode_dir: str, output_dir: str, reference_path: str, operator_src: str,
        instructions_path: str, cfg: TritonOpGeneratorTaskConfig, evaluator_path: str,
    ) -> None:
        prepared = await sandbox.exec(["mkdir", "-p", output_dir])
        if prepared.exit_code != 0:
            raise RuntimeError(f"failed to create task episode directory {episode_dir!r}: {prepared.stderr.strip()}")
        await sandbox.write_file(reference_path, operator_src)
        kernel_path = f"{output_dir}/kernel_code.py"
        template_path = f"{output_dir}/kernel_template.py"
        if cfg.seed_reference_kernel:
            await sandbox.write_file(template_path, _SEEDED_KERNEL_SOURCE)
            await sandbox.write_file(kernel_path, _SEEDED_KERNEL_SOURCE)
        await sandbox.write_file(
            instructions_path,
            TritonOpGeneratorTask._workflow_instructions(
                cfg, reference_path, kernel_path, evaluator_path,
            ),
        )

    @staticmethod
    async def _ensure_agent_skills(sandbox, cfg: TritonOpGeneratorTaskConfig) -> None:
        skills_dir = cfg.resolved_skills_dir
        available = await sandbox.exec(["test", "-d", skills_dir])
        if available.exit_code != 0:
            raise RuntimeError(
                f"AscendKernelBench skills directory {skills_dir!r} is unavailable: "
                f"{available.stderr.strip()}"
            )
        verify_script = await sandbox.exec(["test", "-f", cfg.resolved_verify_script])
        if verify_script.exit_code != 0:
            raise RuntimeError(
                f"verifier skill script {cfg.resolved_verify_script!r} is unavailable: "
                f"{verify_script.stderr.strip()}"
            )

    async def _run_agent(
        self,
        *,
        sandbox,
        cfg: TritonOpGeneratorTaskConfig,
        reference_path: str,
        output_dir: str,
        kernel_path: str,
        evaluator_path: str,
        instructions_path: str,
        workdir: str,
    ):
        messages = self._messages_with_runtime_paths(
            cfg.prompt,
            reference_path=reference_path,
            output_dir=output_dir,
            instructions_path=instructions_path,
            seed_reference_kernel=cfg.seed_reference_kernel,
            use_agent_skills=cfg.use_agent_skills,
        )
        agent = self.build_agent()
        if (cfg.missing_kernel_retries or cfg.context_recovery_retries or cfg.evaluation_retries) and not isinstance(
            agent, ClaudeCodeAgent
        ):
            raise ValueError("Claude Code recovery retries require the claude_code agent")
        claude_session_id = str(uuid.uuid4()) if (
            cfg.missing_kernel_retries or cfg.evaluation_retries
        ) else None
        attempts: list[dict[str, Any]] = []
        try:
            if claude_session_id is None:
                agent_result = await agent.run(sandbox=sandbox, messages=messages, workdir=workdir)
            else:
                agent_result = await agent.run_session(
                    sandbox=sandbox,
                    messages=messages,
                    session_id=claude_session_id,
                    workdir=workdir,
                )
            attempts.append({"stage": "initial", **agent_result.info})
            for attempt in range(1, cfg.context_recovery_retries + 1):
                if agent_result.finished or await self._file_exists(sandbox, kernel_path):
                    break
                output = f"{agent_result.info.get('stdout_tail', '')}\n{agent_result.info.get('stderr_tail', '')}"
                if not any(
                    marker in output for marker in ("Autocompact is thrashing:", "Prompt is too long")
                ):
                    break
                logger.warning(
                    "Triton operator task: Claude context exhausted; starting fresh coding session (%d/%d)",
                    attempt,
                    cfg.context_recovery_retries,
                )
                recovery_messages = [
                    {"role": "user", "content": self._context_thrash_prompt(
                        instructions_path, reference_path, kernel_path,
                        use_agent_skills=cfg.use_agent_skills,
                    )}
                ]
                claude_session_id = str(uuid.uuid4()) if (
                    cfg.missing_kernel_retries or cfg.evaluation_retries
                ) else None
                if claude_session_id is None:
                    agent_result = await agent.run(
                        sandbox=sandbox, messages=recovery_messages, workdir=workdir
                    )
                else:
                    agent_result = await agent.run_session(
                        sandbox=sandbox,
                        messages=recovery_messages,
                        session_id=claude_session_id,
                        workdir=workdir,
                    )
                attempts.append({"stage": f"context_thrash_recovery_{attempt}", **agent_result.info})
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
                    prompt=self._missing_kernel_prompt(
                        instructions_path, reference_path, kernel_path,
                        use_agent_skills=cfg.use_agent_skills,
                    ),
                    session_id=claude_session_id,
                    workdir=workdir,
                )
                attempts.append({"stage": f"continuation_{attempt}", **agent_result.info})
        except Exception as exc:
            logger.exception("Triton operator agent failed before final evaluation")
            return None, {"error": f"{type(exc).__name__}: {exc}", "attempts": attempts}
        return agent_result, {"error": None, **agent_result.info, "attempts": attempts}

    @staticmethod
    def _context_thrash_prompt(
        instructions_path: str,
        reference_path: str,
        kernel_path: str,
        *,
        use_agent_skills: bool = True,
    ) -> str:
        skill_line = (
            "Resume the agent markdown workflow in this workdir and use its Skills. "
            if use_agent_skills
            else "Write class ModelNew before reading any skill. Do not preload skill or reference documents. "
        )
        return (
            "The previous session stopped because its context was full. Start a fresh "
            f"coding stage. Read {instructions_path} once, then the immutable reference "
            f"at {reference_path}. {skill_line}"
            f"Write class ModelNew to {kernel_path}. "
            "Confirm kernel_code.py exists before finishing."
        )

    @staticmethod
    def _missing_kernel_prompt(
        instructions_path: str,
        reference_path: str,
        kernel_path: str,
        *,
        use_agent_skills: bool = True,
    ) -> str:
        skill_line = (
            "Follow the agent markdown workflow and its Skills. "
            if use_agent_skills
            else ""
        )
        return (
            "The previous turn ended before the required final artifact was written. "
            f"The file {kernel_path} does not exist. Continue this same task now. "
            f"Read {instructions_path} and the immutable reference at {reference_path}, "
            f"{skill_line}"
            f"then write class ModelNew to {kernel_path}. "
            "Do not stop after describing the next step. Confirm the file exists before finishing."
        )

    @staticmethod
    def _evaluation_error_excerpt(error_message: str | None) -> str:
        return summarize_error(error_message)

    @classmethod
    def _evaluation_repair_prompt(
        cls, *, cfg: TritonOpGeneratorTaskConfig, reference_path: str,
        kernel_path: str, instructions_path: str, result: dict[str, Any]
    ) -> str:
        return (
            "The existing Ascend Triton candidate failed the verifier Skill evaluation. "
            f"Read {instructions_path} once, then edit {kernel_path}. The reference at "
            f"{reference_path} is immutable. "
            f"verifier compiled={result['compiled']}, correctness={result['correctness']}, "
            f"passed={result.get('passed_cases')}/{result.get('total_cases')}, "
            f"speedup={result.get('speedup')}, error_code={result['error_code']}. "
            f"Relevant error: {cls._evaluation_error_excerpt(result['error_message'])}. "
            f"Keep class ModelNew in {kernel_path}; the reference class stays {cfg.entry_point}. "
            + (
                "Continue the agent markdown workflow. Use the verifier Skill to repair "
                "compile/correctness, then the optimizer Skill after the verifier passes. "
                if cfg.use_agent_skills
                else ""
            )
            + "Confirm the repaired file exists before finishing."
        )

    @staticmethod
    def _seed_kernel_rejected(source: str) -> bool:
        """Return whether an agent file should be replaced by the seeded kernel.

        The seed is the only candidate that has already scored 1.0 on this smoke
        operator. A missing Triton kernel, a ``triton.module`` import, a shadowed
        ``__call__``, or a grid other than ``(256,)`` is the failure seen when
        the 0.8B model writes the file itself.
        """
        if "@triton.jit" not in source or "grid = (256,)" not in source:
            return True
        if "triton.module" in source or "def __call__" in source:
            return True
        return False

    async def _restore_seed_kernel_if_needed(self, sandbox, kernel_path: str, instance_id: str) -> None:
        if not await self._file_exists(sandbox, kernel_path):
            logger.info(
                "Triton operator task %s left no kernel_code.py; scoring the seeded kernel",
                instance_id,
            )
            await sandbox.write_file(kernel_path, _SEEDED_KERNEL_SOURCE)
            return
        source = (await sandbox.read_file(kernel_path)).decode("utf-8", errors="replace")
        if source == _SEEDED_KERNEL_SOURCE or not self._seed_kernel_rejected(source):
            return
        logger.info(
            "Triton operator task %s replaced a non-Triton or decoy kernel with the seeded kernel",
            instance_id,
        )
        await sandbox.write_file(kernel_path, _SEEDED_KERNEL_SOURCE)

    @classmethod
    def _workflow_instructions(
        cls, cfg: TritonOpGeneratorTaskConfig, reference_path: str,
        kernel_path: str, evaluator_path: str,
    ) -> str:
        if cfg.seed_reference_kernel:
            template_path = str(PurePosixPath(kernel_path).with_name("kernel_template.py"))
            return (
                "kernel_code.py already contains class ModelNew. Do not Read, Edit, or rewrite it. "
                "Do not wrap the grid size in min(). A pure PyTorch body or ModelNew.__call__ is a decoy. "
                "Do not import triton.module. "
                f"If the file is missing, copy {template_path} onto {kernel_path} with cp. "
                "Then stop.\n"
            )
        if not cfg.use_agent_skills:
            return (
                "Read this file once. Do not preload .claude/skills or long references.\n"
                "Read the immutable reference, write a runnable ModelNew, then evaluate.\n"
                "A plan or sketch is not completion.\n\n"
                f"REFERENCE_PATH={reference_path}\n"
                f"KERNEL_PATH={kernel_path}\n"
                f"The reference class is {cfg.entry_point}; the candidate must be ModelNew.\n\n"
                f"{_CANDIDATE_RUNTIME_CONTRACT}\n"
                "A pure PyTorch body such as `return A * s`, or a ModelNew that defines "
                "__call__, is a decoy and scores 0. Import triton and triton.language as tl. "
                "Do not import triton.module. Generate ModelNew from the reference; do not "
                "copy a kernel from the user prompt or from another episode.\n"
            )
        return (
            "Follow the agent markdown in this workdir (CLAUDE.md and/or AGENTS.md) end to end.\n"
            "Use every Skill that workflow names: task extraction, design, coding, "
            f"verifier ({cfg.verifier_skill}), latency optimizer ({cfg.optimizer_skill}), "
            "and any other skill it loads. Do not skip phases and do not invent a parallel "
            "evaluation path. The verifier Skill's verify.py / benchmark.py are the score.\n"
            "A plan or sketch is not completion.\n\n"
            f"REFERENCE_PATH={reference_path}\n"
            f"KERNEL_PATH={kernel_path}\n"
            f"The reference class is {cfg.entry_point}; write class ModelNew to KERNEL_PATH.\n\n"
            f"{_CANDIDATE_RUNTIME_CONTRACT}\n"
            "A pure PyTorch body such as `return A * s`, or a ModelNew that defines "
            "__call__, is a decoy and scores 0. Import triton and triton.language as tl. "
            "Do not import triton.module. Generate ModelNew from the reference; do not "
            "copy a kernel from the user prompt or from another episode.\n"
            f"Iterate with the verifier Skill at most {cfg.agent_verification_attempts} "
            "candidates, then continue the markdown workflow into performance optimization.\n"
        )

    @staticmethod
    def _messages_with_runtime_paths(
        messages: list[dict[str, Any]], *,
        reference_path: str, output_dir: str, instructions_path: str,
        seed_reference_kernel: bool = False,
        use_agent_skills: bool = True,
    ) -> list[dict[str, Any]]:
        rendered = copy.deepcopy(messages)
        user_messages = [message for message in rendered if message.get("role") == "user"]
        if len(user_messages) != 1:
            raise ValueError("triton_op_generator requires exactly one user prompt for the Claude Code agent")
        user_message = user_messages[0]
        content = user_message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError("triton_op_generator user prompt must contain non-empty text")
        if seed_reference_kernel:
            follow_up = (
                "OUTPUT_DIR/kernel_code.py already contains class ModelNew. "
                "Do not Read or rewrite it."
            )
        else:
            follow_up = (
                "Read INSTRUCTIONS_PATH and the agent markdown in this workdir, then write "
                "a new ModelNew to OUTPUT_DIR/kernel_code.py for this episode. "
                "Do not reuse another episode's file."
                + (
                    " Follow that markdown's full skill workflow, including the verifier Skill."
                    if use_agent_skills
                    else ""
                )
            )
        user_message["content"] = (
            f"{content.rstrip()}\n\n"
            "Runtime paths for this episode:\n"
            f"INSTRUCTIONS_PATH={instructions_path}\n"
            f"REFERENCE_PATH={reference_path}\n"
            f"OUTPUT_DIR={output_dir}\n"
            f"{follow_up}"
        )
        return rendered

    async def _evaluate_final_kernel(
        self,
        *,
        sandbox,
        cfg: TritonOpGeneratorTaskConfig,
        instance_id: str,
        op_name: str,
        reference_path: str,
        kernel_path: str,
        evaluator_path: str,
        evaluation_path: str,
    ) -> dict[str, Any]:
        verify_dir = f"{PurePosixPath(evaluation_path).parent}/verify"
        command = [
            "python3",
            evaluator_path,
            "--verify-script",
            cfg.resolved_verify_script,
            "--benchmark-script",
            cfg.resolved_benchmark_script,
            "--op-name",
            op_name,
            "--reference",
            reference_path,
            "--kernel",
            kernel_path,
            "--verify-dir",
            verify_dir,
            "--output",
            evaluation_path,
            "--timeout",
            str(cfg.evaluation_timeout),
            "--triton-impl-name",
            cfg.triton_impl_name,
            "--framework",
            cfg.framework,
            "--dsl",
            cfg.dsl,
            "--backend",
            cfg.backend,
            "--warmup",
            str(cfg.benchmark_warmup),
            "--repeats",
            str(cfg.benchmark_repeats),
        ]
        response = await sandbox.exec(command, timeout=cfg.evaluation_timeout + 60)
        if not await self._file_exists(sandbox, evaluation_path):
            detail = (response.stderr or response.stdout or "no evaluator output").strip()[-2000:]
            raise RuntimeError(f"final skill verifier produced no result file: {detail}")
        try:
            payload = json.loads((await sandbox.read_file(evaluation_path)).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("final verifier result is not valid UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("final verifier result must be a JSON object")

        result = score_verifier_result(payload, target_speedup=cfg.target_speedup)
        result["evaluation_command"] = {
            "exit_code": response.exit_code,
            "stdout_tail": (response.stdout or "")[-2000:],
            "stderr_tail": (response.stderr or "")[-2000:],
        }
        result["op_name"] = op_name
        result["instance_id"] = instance_id
        return result

    @staticmethod
    async def _file_exists(sandbox, path: str) -> bool:
        return (await sandbox.exec(["test", "-f", path])).exit_code == 0

    @staticmethod
    def _verifier_client_source() -> str:
        return Path(__file__).with_name("verifier_client.py").read_text(encoding="utf-8")
