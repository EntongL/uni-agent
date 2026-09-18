"""Run one AscendKernelBench parquet row through Uni-Agent without verl.

This is the task-level smoke test. It attaches to the configured resident
Docker container, invokes the live policy through the Claude Code Agent, and
prints the final independently computed KernelGYM TaskResult.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
from pathlib import Path
import sys
from typing import Any

# Allow the documented ``python examples/.../run_single_sample.py`` invocation
# from an uninstalled source checkout.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from uni_agent.tasks import TaskConfigResolver, get_task


def load_sample(path: Path, index: int) -> dict[str, Any]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if index < 0 or index >= table.num_rows:
        raise IndexError(f"sample index {index} is outside [0, {table.num_rows})")
    row = table.slice(index, 1).to_pylist()[0]
    if not isinstance(row, dict):
        raise ValueError("parquet row must decode to a mapping")
    return row


def build_task_config(
    row: dict[str, Any], *, task_config_path: str, base_url: str, api_key: str, model_name: str, keep_artifacts: bool
) -> dict[str, Any]:
    extra_info = row.get("extra_info")
    if not isinstance(extra_info, dict):
        raise ValueError("dataset row extra_info must be a mapping")
    tools_kwargs = extra_info.get("tools_kwargs")
    if not isinstance(tools_kwargs, dict) or not isinstance(tools_kwargs.get("task"), dict):
        raise ValueError("dataset row requires extra_info.tools_kwargs.task")
    prompt = row.get("prompt")
    if not isinstance(prompt, list):
        raise ValueError("dataset row prompt must be a message list")

    sample_task = dict(tools_kwargs["task"])
    sample_task["prompt"] = prompt
    resolver = TaskConfigResolver.from_file(task_config_path)
    resolved = resolver.resolve(
        sample_task,
        runtime_model={"base_url": base_url, "api_key": api_key, "model_name": model_name},
    )
    if keep_artifacts:
        resolved["cleanup_episode"] = False
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one AscendKernelBench Uni-Agent task")
    parser.add_argument("--data", type=Path, required=True, help="kernelbench_level1.parquet")
    parser.add_argument("--task-config", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--base-url", required=True, help="Anthropic-compatible policy endpoint or Gateway session /v1 URL")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--api-key-env", default="MODEL_API_KEY")
    parser.add_argument("--keep-artifacts", action="store_true")
    args = parser.parse_args()

    row = load_sample(args.data.expanduser(), args.index)
    config = build_task_config(
        row,
        task_config_path=str(args.task_config.expanduser()),
        base_url=args.base_url,
        api_key=os.environ.get(args.api_key_env, "EMPTY"),
        model_name=args.model_name,
        keep_artifacts=args.keep_artifacts,
    )
    result = asyncio.run(get_task(config).run())
    print(json.dumps(dataclasses.asdict(result), ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
