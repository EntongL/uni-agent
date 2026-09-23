# Ascend Triton single-sample Gateway smoke

This recipe runs one `triton_op_generator` row from AscendKernelBench through
the same Uni-Agent Gateway path used by RL: the rollout engine creates a
session, Claude Code uses its Anthropic Messages endpoint, and the task
independently scores `kernel_code.py` with KernelGYM. Finalized trajectories
carry the generated token IDs, masks, log probabilities, and task reward.

Before running it, verify the prepared resident container contains:

- the AscendKernelBench checkout and its `agent/.claude/skills` directory;
- a running KernelGYM API reachable from *inside the container*;
- the `claude` CLI and the Ascend Triton runtime.

Adjust `agent_workdir`, `episode_root`, and `kernelgym_url` in
`task_config_claude_code.yaml` if the checkout is not at
`/workspace/AscendKernelBench`.

## Gateway smoke

Run this from the Ray head / rollout host, not from inside
`uni-agent-resident`. The resident container is where Claude Code and
KernelGYM evaluation run. The rollout host must be able to reach that Docker
container, and the container must be able to reach the rollout host's Ray-node
IP: Gateway binds a per-session port on that IP.

The command below selects exactly parquet row `0`. Change `--start-index` to
run another row. `MODEL_API_KEY` and `--base-url` are intentionally absent:
the policy endpoint is the Gateway session, not an externally served API.

```bash
export DEVICE=npu
export VERL_PLATFORM=huawei
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export GLOBAL_CONCURRENCY=1
unset LOCAL_RANK

# Single-node example: start Ray after LOCAL_RANK has been cleared.
ray start --head \
  --port=6379 \
  --resources='{"NPU": 1}' \
  --disable-usage-stats
export RAY_ADDRESS=127.0.0.1:6379

python3 examples/inference/parallel_infer_verl.py \
  --data-path /path/to/AscendKernelBench/uniagent-training-data/kernelbench_level1.parquet \
  --model-path /path/to/local-model \
  --served-model-name glm-5.2 \
  --task-config examples/ascend_triton/task_config_claude_code.yaml \
  --engine vllm \
  --tool-parser hermes \
  --nnodes 1 \
  --n-gpus-per-node 1 \
  --tensor-parallel-size 1 \
  --gateway-count 1 \
  --concurrency 1 \
  --n 1 \
  --start-index 0 \
  --limit 1 \
  --log-dir /mnt/shared/uni-agent-ascend-triton-smoke \
  --result-path /mnt/shared/uni-agent-ascend-triton-smoke/result.json
```

Set `--tool-parser` to the parser used by the rollout model's chat template;
for example, use `hermes` for standard Qwen3 models whose template emits
`<tool_call>{"name": ..., "arguments": ...}</tool_call>`. Qwen3-Coder models
may instead use the function/parameter XML format; select `qwen3_xml` or
`qwen3_coder` according to the actual checkpoint and installed parser. It must
agree with the model's output format. Use
the same `--model-path`, parser, and hardware settings that the later RL job
will use.

For a GLM checkpoint, do not assume `hermes` or `qwen3_coder`. Inspect the
parsers installed in the rollout environment and choose the GLM-specific one
when available:

```bash
python3 - <<'PY'
from vllm.tool_parsers import ToolParserManager
print("eager:", sorted(getattr(ToolParserManager, "tool_parsers", {})))
print("lazy:", sorted(getattr(ToolParserManager, "lazy_parsers", {})))
for name in ("hermes", "qwen3_coder", "qwen3_xml", "glm45", "glm47"):
    try:
        parser = ToolParserManager.get_tool_parser(name)
    except Exception as exc:
        print(f"{name}: unavailable ({type(exc).__name__}: {exc})")
    else:
        print(f"{name}: {parser}")
PY
```

The recipe enables temporary Claude diagnostics. The full Claude debug stream
is written inside the resident container at
`/tmp/uni-agent-claude-debug.log`; inspect it with:

```bash
docker exec uni-agent-resident sh -lc \
  'tail -n 300 /tmp/uni-agent-claude-debug.log'
```

`--served-model-name` is the model string sent by Claude Code to the
Anthropic-compatible Gateway. It does not select the local checkpoint; the
checkpoint is selected by `--model-path`. In the current resident-router setup,
`glm-5.2` is the alias that reached the Gateway without the `Model not exist`
400 seen with `Qwen3-0.6B` and `claude-sonnet-4-5`. It is therefore valid for
the smoke even when `--model-path` points to Qwen3-0.6B; the parser must still
follow the actual checkpoint (`hermes` for the standard Qwen3 JSON tool-call
template). Prefer the exact ID returned by the
router's `/v1/models` if that endpoint is available.

The smoke passes when the summary reports `1 / 1` scored session and
`result.json` has one score. `reward=0` is a valid model outcome. Inspect
`<log-dir>/<session-id>/task.log` for the final KernelGYM status and the
matching `framework.log`, `trajectory.json`, and `trajectory.npz`. For the first investigation, set
`cleanup_episode: false` in `task_config_claude_code.yaml` to retain the
task-owned candidate files, then restore it before multi-episode runs.

## Cross-host resident tunnel

When the resident Docker host cannot route to the Ray/Gateway host, add
`sandbox.sandbox_kwargs.ssh_reverse_tunnel` to the task config. The rollout
process starts one SSH reverse forward per Gateway session and closes it after
the task finishes. `remote_port: 0` asks the SSH server to allocate a unique
remote port, which is safe for concurrent rollouts.

```yaml
sandbox:
  provider: docker
  sandbox_kwargs:
    container_ref: uni-agent-resident
    ssh_reverse_tunnel:
      ssh_host: <atlas-41-address>
      ssh_user: root
      remote_port: 0
      identity_file: /root/.ssh/id_ed25519
      known_hosts_file: /root/.ssh/known_hosts
```

The training container must have an `ssh` client and key-based access to
`ssh_host`; the SSH server must allow `AllowTcpForwarding`. The resident
container should use host networking so its `127.0.0.1:<allocated-port>` is the
atlas host loopback. Logs include the allocated tunnel port without printing
the SSH key. If the SSH server does not report an allocated port for
`remote_port: 0`, set a unique fixed `remote_port` per concurrent worker.

## Direct endpoint smoke (optional)

`run_single_sample.py` remains useful for evaluating an already hosted
Anthropic-compatible model API, but it bypasses Gateway and does not produce
training trajectories:

```bash
export MODEL_API_KEY='<policy-api-key>'

python3 examples/ascend_triton/run_single_sample.py \
  --data /path/to/AscendKernelBench/uniagent-training-data/kernelbench_level1.parquet \
  --task-config examples/ascend_triton/task_config_claude_code.yaml \
  --base-url http://<policy-host>:<port>/v1 \
  --model-name <served-model> \
  --index 0 \
  --keep-artifacts
```

The recipe deliberately keeps `CONCURRENCY=1`: a resident container retains
filesystem state across episodes. `--keep-artifacts` retains the task-owned
directory for the direct smoke; the recipe otherwise cleans it automatically
and is safe to reuse for automated RL runs.
