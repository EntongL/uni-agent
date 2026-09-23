# ruff: noqa: E501
"""Agent runner that bridges the framework's gateway sessions to uni_agent tasks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from uni_agent.rl_insight.adapter import task_span
from uni_agent.tasks import TaskConfigResolver, TaskResult, get_task
from uni_agent.tasks.config import _deep_merge

from .ssh_reverse_tunnel import SshReverseTunnel, SshReverseTunnelConfig

if TYPE_CHECKING:
    from uni_agent.gateway.session import SessionHandle

logger = logging.getLogger(__name__)


def _rewrite_gateway_url(gateway_url: str, proxy_port: int) -> str:
    """Rewrite a gateway URL to the sandbox-internal tunnel (``127.0.0.1:<proxy_port>``).

    Replaces host:port with ``127.0.0.1:<proxy_port>`` and keeps the path, so an
    in-sandbox endpoint reaches the gateway through the reverse tunnel. Example:
    ``http://gateway.example:40169/sessions/abc/v1`` ->
    ``http://127.0.0.1:38197/sessions/abc/v1``.
    """
    return f"http://127.0.0.1:{proxy_port}{urlparse(gateway_url).path}"


def _extract_upstream(gateway_url: str) -> str | None:
    """Extract ``host:port`` from a gateway URL (the tunnel's ``upstream``).

    Returns ``None`` when the URL carries no host or port, so callers can fail
    loudly instead of forwarding a ``None:None`` upstream.
    """
    parsed = urlparse(gateway_url)
    if not parsed.hostname or not parsed.port:
        return None
    return f"{parsed.hostname}:{parsed.port}"


def _inject_gateway_tunnel(task: dict[str, Any], base_url: str) -> dict[str, Any]:
    """Fill the runtime side of an openyuanrong gateway reverse tunnel.

    The sandbox config declares its tunnel port via ``sandbox_kwargs.proxy_port``;
    only the runtime-derived pieces are injected here: ``upstream`` (the gateway
    host:port, so the provider knows where to forward the tunnel) and the agent's
    ``model.base_url`` rewritten to the sandbox-internal tunnel address. The agent
    itself stays tunnel-agnostic -- it just sees a base_url that already points at
    ``127.0.0.1:<proxy_port>``.

    The reverse tunnel is currently supported only on the openyuanrong sandbox;
    configuring ``proxy_port`` on any other provider is rejected loudly instead of
    being silently ignored (which would leave the agent pointed at an unreachable
    ``127.0.0.1`` address).
    """
    provider = (task.get("sandbox") or {}).get("provider")
    if provider != "openyuanrong":
        raise ValueError(
            "the gateway reverse tunnel (sandbox.sandbox_kwargs.proxy_port) is currently "
            f"supported only on 'openyuanrong' sandboxes, got provider={provider!r}; "
            "switch the sandbox provider or drop proxy_port"
        )
    upstream = _extract_upstream(base_url)
    if upstream is None:
        raise ValueError(f"cannot derive gateway tunnel upstream from base_url={base_url!r}")
    proxy_port = task["sandbox"]["sandbox_kwargs"]["proxy_port"]
    return _deep_merge(
        task,
        {
            "sandbox": {"sandbox_kwargs": {"upstream": upstream}},
            "agent": {"model": {"base_url": _rewrite_gateway_url(base_url, proxy_port)}},
        },
    )


def _extract_ssh_reverse_tunnel(
    task: dict[str, Any],
) -> tuple[dict[str, Any], SshReverseTunnelConfig | None]:
    """Remove and parse runner-owned SSH tunnel config before task construction."""

    sandbox = task.get("sandbox") or {}
    sandbox_kwargs = sandbox.get("sandbox_kwargs") or {}
    raw_config = sandbox_kwargs.get("ssh_reverse_tunnel")
    if raw_config is None:
        return task, None
    if sandbox.get("provider") != "docker":
        raise ValueError(
            "sandbox.sandbox_kwargs.ssh_reverse_tunnel is currently supported only "
            f"for provider='docker', got provider={sandbox.get('provider')!r}"
        )
    if "proxy_port" in sandbox_kwargs:
        raise ValueError(
            "configure either sandbox.sandbox_kwargs.proxy_port or "
            "sandbox.sandbox_kwargs.ssh_reverse_tunnel, not both"
        )
    cleaned_kwargs = dict(sandbox_kwargs)
    del cleaned_kwargs["ssh_reverse_tunnel"]
    cleaned_task = _deep_merge(task, {"sandbox": {"sandbox_kwargs": cleaned_kwargs}})
    return cleaned_task, SshReverseTunnelConfig.from_mapping(raw_config)


def score_from_runner_result(
    *,
    data_source: str,
    solution_str: str,
    ground_truth: object,
    extra_info: dict[str, Any],
    **_reward_manager_kwargs: Any,
) -> dict[str, int | float | bool]:
    """Adapt the managed Runner result for a VERL RewardLoopWorker scorer.

    The Worker may still apply its own post-processing (for example DAPO's
    overlong penalty), so this is an adapter for the Runner payload rather than
    a bypass of the Worker.
    """
    runner_reward_info = extra_info["runner_reward_info"]
    return {**runner_reward_info["metrics"], "score": float(runner_reward_info["reward"])}


# Keep the conventional VERL callback name available for existing configs.
# New configs should prefer the descriptive ``score_from_runner_result`` name.
compute_score = score_from_runner_result


async def run_task(
    *,
    session: SessionHandle,
    tools_kwargs: dict[str, Any] | None = None,
    raw_prompt: Any = None,
    sample_index: int | None = None,
    task_config_path: str | None = None,
    api_key: str = "EMPTY",
    model_name: str | None = None,
    **_: Any,
) -> TaskResult:
    """Resolve the sample's task, run it against ``session``, and return its result.

    Satisfies the framework's ``AgentRunner`` contract (``session`` / ``raw_prompt``
    / ``sample_index`` / ``tools_kwargs``). The framework's ``raw_prompt`` contains
    the authoritative dataset/source messages and overrides any serialized Task prompt.

    Run-level defaults come from the per-task-name YAML file selected by
    ``task_config_path``. ``TaskConfigResolver`` applies that Task Config, the
    sample values, and the live endpoint in order.
    """
    sample_config = tools_kwargs.get("task") if tools_kwargs else None
    if not isinstance(sample_config, dict):
        raise ValueError("run_task requires tools_kwargs['task'] (the serialized Task Config)")
    sample_config = dict(sample_config)
    sample_config["prompt"] = raw_prompt

    resolver = TaskConfigResolver.from_file(task_config_path) if task_config_path else TaskConfigResolver()
    task = resolver.resolve(
        sample_config,
        runtime_model={
            "base_url": session.base_url,
            "api_key": api_key,
            "model_name": model_name,
        },
    )

    task, ssh_tunnel_config = _extract_ssh_reverse_tunnel(task)
    ssh_tunnel: SshReverseTunnel | None = None

    try:
        if ssh_tunnel_config is not None:
            if not session.base_url:
                raise ValueError("ssh_reverse_tunnel requires a live Gateway session base_url")
            ssh_tunnel = await SshReverseTunnel.open(session.base_url, ssh_tunnel_config)
            task = _deep_merge(
                task,
                {
                    "agent": {
                        "model": {
                            "base_url": _rewrite_gateway_url(session.base_url, ssh_tunnel.remote_port)
                        }
                    }
                },
            )
            logger.info(
                "run_task: SSH reverse tunnel mapped session to sandbox endpoint 127.0.0.1:%s",
                ssh_tunnel.remote_port,
            )

        # openyuanrong reverse tunnel: the sandbox config pins the in-sandbox
        # tunnel port; runtime gateway details are injected from session.base_url.
        tunnel_port = (task.get("sandbox") or {}).get("sandbox_kwargs", {}).get("proxy_port")
        if tunnel_port and session.base_url:
            task = _inject_gateway_tunnel(task, session.base_url)

        task_name = task.get("name")
        logger.info(
            "run_task start: task=%s sample_index=%s session_base_url=%s model_name=%s",
            task_name,
            sample_index,
            session.base_url,
            model_name,
        )

        prompt = task.get("prompt", [])
        with task_span(tools_kwargs, task_name=task_name, prompt=prompt) as span:
            task_instance = get_task(task)
            result = await task_instance.run()
            span.record_result(result, reward_posted=False)
            logger.info(
                "run_task done: task=%s reward=%s acc=%s finished=%s",
                task_name,
                result.reward,
                result.accuracy,
                result.finished,
            )
        return result
    finally:
        if ssh_tunnel is not None:
            await ssh_tunnel.close()
