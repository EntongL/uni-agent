from __future__ import annotations

import asyncio
import uuid
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from .base import ExecResult, Sandbox, _to_str
from .registry import register_sandbox

if TYPE_CHECKING:
    from .base import SandboxConfig


@register_sandbox("docker")
class DockerSandbox(Sandbox):
    """Run an isolated sandbox from an image available to a local Docker daemon."""

    def __init__(
        self,
        *,
        image: str | None = "python:3.12",
        docker_binary: str = "docker",
        container_name: str | None = None,
        container_ref: str | None = None,
        verify_container_image: bool = True,
        run_args: list[str] | None = None,
        pull_policy: str = "missing",
        entrypoint: str = "sleep",
        command: list[str] | None = None,
    ) -> None:
        if container_ref is not None and not container_ref.strip():
            raise ValueError("container_ref must be a non-empty Docker container name or ID")
        if container_ref is not None:
            conflicts = []
            if container_name is not None:
                conflicts.append("container_name")
            if run_args:
                conflicts.append("run_args")
            if pull_policy != "missing":
                conflicts.append("pull_policy")
            if entrypoint != "sleep":
                conflicts.append("entrypoint")
            if command is not None:
                conflicts.append("command")
            if conflicts:
                joined = ", ".join(conflicts)
                raise ValueError(f"container_ref cannot be combined with Docker create options: {joined}")
        elif not image:
            raise ValueError("image is required when container_ref is not set")

        self.image = image
        self.docker_binary = docker_binary
        self.container_name = container_name
        self.container_ref = container_ref
        self.verify_container_image = verify_container_image
        self.run_args = list(run_args or [])
        if pull_policy not in {"always", "missing", "never"}:
            raise ValueError("pull_policy must be one of: 'always', 'missing', 'never'")
        self.pull_policy = pull_policy
        self.entrypoint = entrypoint
        self.command = list(command or ["infinity"])
        self._container_name: str | None = None

    @classmethod
    def from_config(cls, config: SandboxConfig) -> DockerSandbox:
        return cls(image=config.image, **config.sandbox_kwargs)

    async def _run_docker(self, *args: str, timeout: float | None = None) -> ExecResult:
        try:
            proc = await asyncio.create_subprocess_exec(
                self.docker_binary,
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Docker executable {self.docker_binary!r} was not found") from exc

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.communicate()
            raise

        return ExecResult(
            exit_code=int(proc.returncode or 0),
            stdout=_to_str(stdout),
            stderr=_to_str(stderr),
        )

    async def start(self) -> None:
        if self._container_name is not None:
            return

        if self.container_ref is not None:
            inspected = await self._run_docker(
                "inspect",
                "--type",
                "container",
                "--format",
                "{{.Id}}\t{{.State.Running}}\t{{.Image}}",
                self.container_ref,
            )
            if inspected.exit_code != 0:
                detail = inspected.stderr.strip() or inspected.stdout.strip()
                raise RuntimeError(f"Docker container {self.container_ref!r} is unavailable: {detail}")

            fields = inspected.stdout.strip().split("\t")
            if len(fields) != 3 or not fields[0]:
                raise RuntimeError(
                    f"Docker container {self.container_ref!r} returned an invalid inspect response: "
                    f"{inspected.stdout.strip()!r}"
                )
            container_id, running, container_image_id = fields
            if running != "true":
                raise RuntimeError(f"Docker container {self.container_ref!r} is not running")

            if self.verify_container_image and self.image is not None:
                image_inspected = await self._run_docker("image", "inspect", "--format", "{{.Id}}", self.image)
                if image_inspected.exit_code != 0:
                    detail = image_inspected.stderr.strip() or image_inspected.stdout.strip()
                    raise RuntimeError(
                        f"Docker image {self.image!r} is not available locally for verification: {detail}"
                    )
                configured_image_id = image_inspected.stdout.strip()
                if configured_image_id != container_image_id:
                    raise RuntimeError(
                        f"Docker container {self.container_ref!r} uses image {container_image_id!r}, "
                        f"but configured image {self.image!r} resolves to {configured_image_id!r}"
                    )

            self._container_name = container_id
            return

        assert self.image is not None
        if self.pull_policy == "never":
            inspected = await self._run_docker("image", "inspect", self.image)
            if inspected.exit_code != 0:
                detail = inspected.stderr.strip() or inspected.stdout.strip()
                raise RuntimeError(f"Docker image {self.image!r} is not available locally: {detail}")

        name = self.container_name or f"uni-agent-{uuid.uuid4().hex[:12]}"
        args = self._build_run_args(name, include_pull=True)
        started = await self._run_docker(*args)
        if started.exit_code != 0 and self._is_legacy_pull_error(started):
            # Docker added `docker run --pull` in 20.10. Older daemons/CLIs
            # reject the flag before creating a container. Fall back to the
            # legacy command shape: `missing` is the old default behavior,
            # `never` was already checked with image inspect above, and
            # `always` is preserved by pulling explicitly first.
            logger.info(
                "Docker does not support --pull; retrying sandbox %s with legacy run syntax",
                name,
            )
            if self.pull_policy == "always":
                pulled = await self._run_docker("pull", self.image)
                if pulled.exit_code != 0:
                    detail = pulled.stderr.strip() or pulled.stdout.strip()
                    raise RuntimeError(f"Failed to pull Docker image {self.image!r}: {detail}")
            args = self._build_run_args(name, include_pull=False)
            started = await self._run_docker(*args)
        if started.exit_code != 0:
            detail = started.stderr.strip() or started.stdout.strip()
            raise RuntimeError(f"Failed to start Docker sandbox from {self.image!r}: {detail}")
        self._container_name = name

    def _build_run_args(self, name: str, *, include_pull: bool) -> list[str]:
        """Build ``docker run`` arguments for both modern and legacy CLIs."""
        args = ["run", "--rm", "-d", "--name", name]
        if include_pull:
            args.extend(["--pull", self.pull_policy])
        if self.entrypoint:
            args.extend(["--entrypoint", self.entrypoint])
        args.extend(self.run_args)
        assert self.image is not None
        args.append(self.image)
        args.extend(self.command)
        return args

    @staticmethod
    def _is_legacy_pull_error(result: ExecResult) -> bool:
        """Return whether a failed run indicates an old CLI/daemon lacks ``--pull``."""
        detail = f"{result.stdout}\n{result.stderr}".lower()
        return "unknown flag" in detail and "--pull" in detail

    async def stop(self) -> None:
        name, self._container_name = self._container_name, None
        if name is not None and self.container_ref is None:
            await self._run_docker("rm", "-f", name)

    def _require_container(self) -> str:
        if self._container_name is None:
            raise RuntimeError("DockerSandbox not started; call start() first")
        return self._container_name

    async def is_alive(self) -> bool:
        if self._container_name is None:
            return False
        try:
            result = await self._run_docker(
                "inspect",
                "--format",
                "{{.State.Running}}",
                self._container_name,
                timeout=10.0,
            )
            return result.exit_code == 0 and result.stdout.strip() == "true"
        except Exception:
            return False

    async def _exec(
        self,
        argv: list[str],
        *,
        timeout: float | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
    ) -> ExecResult:
        args = ["exec"]
        if workdir:
            args.extend(["--workdir", workdir])
        for key, value in (env or {}).items():
            args.extend(["--env", f"{key}={value}"])
        args.append(self._require_container())
        args.extend(argv)
        return await self._run_docker(*args, timeout=timeout)

    async def upload_file(self, local_file: Path | str, remote_file: str) -> None:
        container = self._require_container()
        parent = str(PurePosixPath(remote_file).parent)
        if parent not in {"", "."}:
            created = await self.exec(["mkdir", "-p", parent])
            if created.exit_code != 0:
                raise RuntimeError(f"Failed to create Docker sandbox directory {parent!r}: {created.stderr.strip()}")
        result = await self._run_docker("cp", str(local_file), f"{container}:{remote_file}")
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to upload {local_file!s} to {remote_file!r}: {result.stderr.strip()}")

    async def download_file(self, remote_file: str, local_file: Path | str) -> None:
        destination = Path(local_file)
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = await self._run_docker(
            "cp",
            f"{self._require_container()}:{remote_file}",
            str(destination),
        )
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to download {remote_file!r}: {result.stderr.strip()}")
