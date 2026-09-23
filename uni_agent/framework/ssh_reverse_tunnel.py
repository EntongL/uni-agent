"""Per-session SSH reverse tunnels for remote Docker sandboxes.

The rollout process owns the Gateway session, while a Docker sandbox may run on
another host.  ``SshReverseTunnel`` opens a reverse ``ssh -R`` forward from the
rollout host to the sandbox host and reports the remote listening port so the
session URL can be rewritten to ``127.0.0.1:<port>`` inside a host-networked
sandbox.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import shlex
import subprocess
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_ALLOCATED_PORT_RE = re.compile(r"Allocated port (\d+) for remote forward", re.IGNORECASE)
_REMOTE_FORWARD_SUCCESS_RE = re.compile(r"remote forward success", re.IGNORECASE)


class SshReverseTunnelError(RuntimeError):
    """Raised when an SSH reverse forward cannot be established."""


@dataclass(frozen=True)
class SshReverseTunnelConfig:
    """Configuration for one per-session SSH reverse forward.

    ``remote_port=0`` asks the SSH server to allocate an unused port.  This is
    the default and avoids collisions when multiple Ray workers start sessions
    concurrently.
    """

    ssh_host: str
    ssh_user: str
    ssh_port: int = 22
    remote_port: int = 0
    bind_address: str = "127.0.0.1"
    identity_file: str | None = None
    known_hosts_file: str | None = None
    connect_timeout: float = 10.0
    startup_timeout: float = 15.0
    server_alive_interval: int = 15
    server_alive_count_max: int = 3
    ssh_binary: str = "ssh"
    ssh_options: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> SshReverseTunnelConfig:
        if not isinstance(value, Mapping):
            raise ValueError("sandbox.sandbox_kwargs.ssh_reverse_tunnel must be a mapping")

        def _required_string(name: str) -> str:
            raw = value.get(name)
            if not isinstance(raw, str) or not raw.strip():
                raise ValueError(f"ssh_reverse_tunnel.{name} must be a non-empty string")
            return raw.strip()

        def _port(name: str, default: int, *, allow_zero: bool = False) -> int:
            raw = value.get(name, default)
            if isinstance(raw, bool):
                raise ValueError(f"ssh_reverse_tunnel.{name} must be an integer")
            try:
                parsed = int(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"ssh_reverse_tunnel.{name} must be an integer") from exc
            minimum = 0 if allow_zero else 1
            if not minimum <= parsed <= 65535:
                raise ValueError(f"ssh_reverse_tunnel.{name} must be between {minimum} and 65535")
            return parsed

        def _positive_float(name: str, default: float) -> float:
            raw = value.get(name, default)
            try:
                parsed = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"ssh_reverse_tunnel.{name} must be positive") from exc
            if parsed <= 0:
                raise ValueError(f"ssh_reverse_tunnel.{name} must be positive")
            return parsed

        raw_options = value.get("ssh_options", ())
        if isinstance(raw_options, str):
            options = (raw_options,)
        elif isinstance(raw_options, (list, tuple)) and all(isinstance(item, str) for item in raw_options):
            options = tuple(raw_options)
        else:
            raise ValueError("ssh_reverse_tunnel.ssh_options must be a string or list of strings")

        identity_file = value.get("identity_file")
        if identity_file is not None and (not isinstance(identity_file, str) or not identity_file.strip()):
            raise ValueError("ssh_reverse_tunnel.identity_file must be a non-empty string when set")
        known_hosts_file = value.get("known_hosts_file")
        if known_hosts_file is not None and (
            not isinstance(known_hosts_file, str) or not known_hosts_file.strip()
        ):
            raise ValueError("ssh_reverse_tunnel.known_hosts_file must be a non-empty string when set")

        return cls(
            ssh_host=_required_string("ssh_host"),
            ssh_user=_required_string("ssh_user"),
            ssh_port=_port("ssh_port", 22),
            remote_port=_port("remote_port", 0, allow_zero=True),
            bind_address=str(value.get("bind_address", "127.0.0.1")),
            identity_file=identity_file.strip() if isinstance(identity_file, str) else None,
            known_hosts_file=known_hosts_file.strip() if isinstance(known_hosts_file, str) else None,
            connect_timeout=_positive_float("connect_timeout", 10.0),
            startup_timeout=_positive_float("startup_timeout", 15.0),
            server_alive_interval=_port("server_alive_interval", 15),
            server_alive_count_max=_port("server_alive_count_max", 3),
            ssh_binary=str(value.get("ssh_binary", "ssh")),
            ssh_options=options,
        )


def _destination_host(hostname: str) -> str:
    """Format an IPv6 gateway hostname for an SSH forwarding specification."""

    return f"[{hostname}]" if ":" in hostname and not hostname.startswith("[") else hostname


def build_ssh_reverse_tunnel_command(
    config: SshReverseTunnelConfig,
    gateway_url: str,
) -> list[str]:
    """Build the non-secret SSH command for one Gateway URL."""

    parsed = urlparse(gateway_url)
    if not parsed.hostname or parsed.port is None:
        raise ValueError(f"gateway URL must contain a host and port, got {gateway_url!r}")
    forward = f"{config.bind_address}:{config.remote_port}:{_destination_host(parsed.hostname)}:{parsed.port}"
    command = [
        config.ssh_binary,
        "-N",
        "-T",
        "-v",
        "-o",
        "BatchMode=yes",
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        f"ConnectTimeout={max(1, int(config.connect_timeout))}",
        "-o",
        f"ServerAliveInterval={config.server_alive_interval}",
        "-o",
        f"ServerAliveCountMax={config.server_alive_count_max}",
        "-p",
        str(config.ssh_port),
    ]
    if config.identity_file:
        command.extend(["-i", config.identity_file])
    if config.known_hosts_file:
        command.extend(["-o", f"UserKnownHostsFile={config.known_hosts_file}"])
    command.extend(config.ssh_options)
    command.extend(["-R", forward, f"{config.ssh_user}@{config.ssh_host}"])
    return command


class SshReverseTunnel:
    """One running SSH reverse forward, closed when its session finishes."""

    def __init__(self, process: asyncio.subprocess.Process, remote_port: int, command: list[str]) -> None:
        self._process = process
        self.remote_port = remote_port
        self.command = command
        self._stderr_tail: list[str] = []
        self._drain_task: asyncio.Task[None] | None = None

    @classmethod
    async def open(cls, gateway_url: str, config: SshReverseTunnelConfig) -> SshReverseTunnel:
        command = build_ssh_reverse_tunnel_command(config, gateway_url)
        logger.info(
            "starting SSH reverse tunnel ssh_host=%s ssh_port=%s remote_port=%s gateway=%s",
            config.ssh_host,
            config.ssh_port,
            config.remote_port,
            _redact_gateway_url(gateway_url),
        )
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise SshReverseTunnelError(f"SSH executable {config.ssh_binary!r} was not found") from exc
        if process.stderr is None:
            await _terminate_process(process)
            raise SshReverseTunnelError("SSH reverse tunnel did not expose stderr for startup diagnostics")

        tunnel = cls(process, config.remote_port, command)
        try:
            await tunnel._wait_until_ready(config.startup_timeout)
        except BaseException:
            await tunnel.close()
            raise
        tunnel._drain_task = asyncio.create_task(tunnel._drain_stderr())
        logger.info("SSH reverse tunnel ready remote_port=%s", tunnel.remote_port)
        return tunnel

    async def _wait_until_ready(self, timeout: float) -> None:
        assert self._process.stderr is not None
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            if self._process.returncode is not None:
                detail = "".join(self._stderr_tail).strip()
                raise SshReverseTunnelError(
                    f"SSH reverse tunnel exited with code {self._process.returncode}: {detail[-2000:]}"
                )
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise SshReverseTunnelError(
                    f"SSH reverse tunnel did not become ready within {timeout:g}s; "
                    f"command={shlex.join(self.command)}"
                )
            try:
                raw_line = await asyncio.wait_for(self._process.stderr.readline(), timeout=remaining)
            except asyncio.TimeoutError as exc:
                raise SshReverseTunnelError(
                    f"SSH reverse tunnel did not become ready within {timeout:g}s; "
                    f"command={shlex.join(self.command)}"
                ) from exc
            if not raw_line:
                returncode = await self._process.wait()
                detail = "".join(self._stderr_tail).strip()
                raise SshReverseTunnelError(
                    f"SSH reverse tunnel exited with code {returncode} before becoming ready: {detail[-2000:]}"
                )
            line = raw_line.decode("utf-8", errors="replace")
            self._remember_stderr(line)
            allocated = _ALLOCATED_PORT_RE.search(line)
            if allocated:
                self.remote_port = int(allocated.group(1))
                return
            if self.remote_port > 0 and _REMOTE_FORWARD_SUCCESS_RE.search(line):
                return

    async def _drain_stderr(self) -> None:
        if self._process.stderr is None:
            return
        try:
            while True:
                raw_line = await self._process.stderr.readline()
                if not raw_line:
                    return
                self._remember_stderr(raw_line.decode("utf-8", errors="replace"))
        except asyncio.CancelledError:
            raise

    def _remember_stderr(self, line: str) -> None:
        self._stderr_tail.append(line)
        del self._stderr_tail[:-20]

    async def close(self) -> None:
        if self._drain_task is not None:
            self._drain_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._drain_task
            self._drain_task = None
        await _terminate_process(self._process)
        logger.info("SSH reverse tunnel closed remote_port=%s", self.remote_port)


async def _terminate_process(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=5.0)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()


def _redact_gateway_url(url: str) -> str:
    parsed = urlparse(url)
    if not parsed.hostname:
        return "<invalid>"
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return f"{parsed.scheme}://{host}{port}{parsed.path}"
