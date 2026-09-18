# Ascend Triton single-sample smoke

This recipe runs one `triton_op_generator` row from AscendKernelBench through
Uni-Agent's Claude Code Agent, then scores the generated `kernel_code.py` with
an independent final KernelGYM request.

Before running it, verify the prepared resident container contains:

- the AscendKernelBench checkout and its `agent/.claude/skills` directory;
- a running KernelGYM API reachable from *inside the container*;
- the `claude` CLI and the Ascend Triton runtime.

Adjust `agent_workdir`, `episode_root`, and `kernelgym_url` in
`task_config_claude_code.yaml` if the checkout is not at
`/workspace/AscendKernelBench`.

Run the smoke from the Uni-Agent checkout on the Docker host:

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

The smoke passes when the JSON result includes a final `status`, compilation
and correctness fields, and a scalar `reward`. `reward=0` is a valid model
outcome; inability to produce a final KernelGYM result is an infrastructure
failure and raises instead of being mislabeled as a model failure.

The recipe deliberately keeps `CONCURRENCY=1`: a resident container retains
filesystem state across episodes. `--keep-artifacts` retains the task-owned
directory for this smoke; the recipe otherwise cleans it automatically and is
safe to reuse for automated RL runs.
