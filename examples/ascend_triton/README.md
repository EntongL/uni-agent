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

python3 examples/inference/parallel_infer_verl.py \
  --data-path /path/to/AscendKernelBench/uniagent-training-data/kernelbench_level1.parquet \
  --model-path /path/to/local-model \
  --task-config examples/ascend_triton/task_config_claude_code.yaml \
  --engine vllm \
  --tool-parser <parser-matching-the-model> \
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
for example, current Uni-Agent recipes use `qwen3_coder` for Qwen3-Coder and
`hermes` for generic Qwen3. It must agree with the vLLM tool-call parser. Use
the same `--model-path`, parser, and hardware settings that the later RL job
will use.

The smoke passes when the summary reports `1 / 1` scored session and
`result.json` has one score. `reward=0` is a valid model outcome. Inspect
`<log-dir>/<session-id>/task.log` for the final KernelGYM status and the
matching `framework.log`, `trajectory.json`, and `trajectory.npz`. For the first investigation, set
`cleanup_episode: false` in `task_config_claude_code.yaml` to retain the
task-owned candidate files, then restore it before multi-episode runs.

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
