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
  --tool-parser qwen3_xml \
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
the smoke even though `--model-path` points to Qwen3-Coder-30B-A3B-Instruct;
the parser must still follow the actual output. Recent resident-container runs emitted
`<function=...><parameter=...>` calls, so the command above uses `qwen3_xml`.
If the checkpoint/template instead emits JSON inside `<tool_call>`, use `hermes`.
Prefer the exact ID returned by the
router's `/v1/models` if that endpoint is available.

The example now targets a 64K-token smoke window for the actual
`Qwen3-Coder-30B-A3B-Instruct` checkpoint:
`CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536` and
`CLAUDE_CODE_MAX_OUTPUT_TOKENS=4096` and
`CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS=4096`. It also enables
`CLAUDE_CODE_SIMPLE_SYSTEM_PROMPT=1`, which reduces system/tool-schema overhead
while keeping the skills and tools available. The checkpoint declares a 256K
native window; 64K is a smaller rollout budget for the one-sample smoke.
The custom `glm-5.2` route keeps Claude Code's proactive compaction enabled.
If an Anthropic-format request reaches the Gateway trajectory limit, the
Gateway returns a recognizable `Prompt is too long` error as a fallback.

Set `CLAUDE_CODE_MAX_CONTEXT_TOKENS` to the actual `--model-path` context, and
set `agent.model.max_total_tokens` so the Gateway's trajectory budget
(`4096 + max_total_tokens` in this inference recipe) matches the intended
rollout length. The sample values `4096 + 61440 = 65536` match the 64K
smoke budget. The `glm-5.2` served-model alias does not determine either
limit. In a training job, the Gateway instead uses the rollout's configured
`prompt_length + response_length`; changing this task YAML does not enlarge a
training rollout. Keep that sum aligned with the actual vLLM `max_model_len`.
In the resident container, inspect the effective Claude window with:

```bash
docker exec uni-agent-resident sh -lc \
  'grep -E "autocompact:.*effectiveWindow|rapid-refill breaker" /tmp/uni-agent-claude-debug.log | tail -n 30'
```

The task log now prints the context settings passed to Claude Code. If Claude
still exits with `Autocompact is thrashing`, the example starts one fresh
coding-stage session rather than resuming the saturated conversation. The
prompt asks Claude to write its first candidate before evaluation. If Claude
reports an output-token-limit error, the Gateway
actor logs the actual `prompt_tokens`, `output_tokens`, requested/effective
`max_tokens`, and trajectory capacity at WARNING level.

`1 / 1` scored sessions only means the rollout produced a score; it does not
prove that a kernel was written or evaluated. The example config resumes the
same Claude conversation once when a clean exit leaves `output/kernel_code.py`
missing (`missing_kernel_retries: 1`). If the file is still missing, the task
returns `finished=False` and reward 0. A generated file can also receive
reward 0 if KernelGYM reports a compile, correctness, or service failure.
The task log prints those fields and the first 4000 characters of a failed
KernelGYM result. The example keeps failed episode artifacts at the path logged
as `retained episode`: inspect `output/kernel_code.py` and
`final-eval/result.json`. Set `retain_failed_episode: false` for large runs
after debugging, to avoid accumulating failed episode directories. Matching
`framework.log`, `trajectory.json`, and `trajectory.npz` contain rollout data.

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
