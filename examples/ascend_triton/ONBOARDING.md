# Ascend Triton Megatron 训练接手指南

本文档面向从零接手本项目的同学，覆盖当前机器上已跑通的完整链路：

**环境准备 → 训练数据准备 → Sandbox / Resident 流程 → Megatron Smoke（1 步）→ Megatron 多步（3 步）**

更细的 Gateway / Claude 诊断说明见同目录 [`README.md`](./README.md)。

---

## 0. 整体架构（先建立心智模型）

```
训练 / Ray 节点（本机 Notebook）                  Ascend 宿主机 atlas-41 (10.1.30.41)
┌─────────────────────────────────┐              ┌──────────────────────────────────┐
│ venv + Ray + Megatron + vLLM    │   SSH Docker │ Docker: uni-agent-resident        │
│ Gateway 开 session              │ ───────────► │ Claude Code + Ascend Triton       │
│ 写 trajectory / 算 GRPO        │              │ episode 目录 + KernelGYM 调评      │
│                                 │ ◄─ reverse ─ │ Claude 经 127.0.0.1:<port> 回调   │
└─────────────────────────────────┘   tunnel     └──────────────────────────────────┘
                                                        │
                                                        ▼
                                                 KernelGYM :19001
```

要点：

- **训练与 Rollout 跑在本机 Ray head**，不要进 `uni-agent-resident` 里起训练。
- **Resident 容器**只负责 Claude Code + KernelGYM 评测；通过本机 `/home/naie/.local/bin/docker`（SSH 到 `root@10.1.30.41`）attach。
- 任务类型：`triton_op_generator`。参考类名 `Model`，候选必须是 `ModelNew`。
- 当前 recipe 使用 **常驻容器**（`container_ref: uni-agent-resident`），不是每次新建 ephemeral 容器。

---

## 1. 目录与路径约定

工作区：`/home/naie/work` → `/home/work/user-job-dir/app`

| 路径 | 用途 |
|------|------|
| `lib/py/uni-agent/` | 代码仓库根（所有脚本的 `repo_root`） |
| `lib/py/uni-agent/examples/ascend_triton/` | 本 recipe：脚本、task yaml、本文档 |
| `lib/py/venv/` | Python 3.12 训练环境（含 ray / torch_npu 等） |
| `model/Qwen3.5-0.8B` | Smoke 默认模型 |
| `model/Qwen3.5-4B` | 多步训练默认模型 |
| `input/input/data/uni_agent/ascend_triton_rl_smoke_20261009152557/` | 源 parquet（1 条 Matrix scalar 样例） |
| `input/data/uni_agent/ascend_triton_megatron_smoke/` | Smoke 切片后的 train/test |
| `input/data/uni_agent/ascend_triton_megatron_4b/` | 4B 多步切片数据 |
| `input/logs/<PROJECT>/<EXP>/` | 训练日志、session、`task.log`、`trajectory.*` |
| `input/ckpts/<PROJECT>/<EXP>/` | `global_step_*` checkpoint |
| `tmp_no_sync/` | Ray 临时目录等（不入同步配额；重启可能丢） |

本 recipe 关键文件：

| 文件 | 作用 |
|------|------|
| `task_config_claude_code.yaml` | Sandbox / Claude / KernelGYM / SSH tunnel |
| `train_megatron_smoke.sh` | Qwen3.5-0.8B × **1 step** Megatron GRPO |
| `train_megatron_4b_multistep.sh` | Qwen3.5-4B × **3 steps**（TP=2 PP=4） |
| `run_single_sample.py` | 不经 VeRL 训练、直连 policy API 的单样本任务烟测 |
| `matrix_scalar_mul_core_limited.py` | 已知可过评测的金标 kernel（调试用） |

底层训练入口：

- `examples/quickstart/training/train_npu_qwen35_megatron_smoke.sh`
- `examples/quickstart/training/make_npu_smoke_data.py`
- `examples/quickstart/training/runtime_env_npu_smoke.yaml`

---

## 2. 环境准备

### 2.1 激活 Python 环境

```bash
source /home/naie/work/lib/py/venv/bin/activate
cd /home/naie/work/lib/py/uni-agent
```

### 2.2 必设环境变量

每次开新 shell / 起 Ray / 提交训练前执行：

```bash
export DEVICE=npu
export VERL_PLATFORM=huawei
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
unset LOCAL_RANK   # 关键：容器里 LOCAL_RANK 常是 "0,1,...,15" 字符串，会弄崩整数解析
```

可选：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export RAY_DATA_HOME=/home/naie/work   # 日志进 input/logs，ckpt 进 input/ckpts
```

### 2.3 检查 NPU

```bash
python -c 'import torch, torch_npu; assert torch.npu.is_available(); print(torch.npu.device_count())'
# 期望：True，本机 16 卡
```

Ascend 栈在 `/usr/local/Ascend/{ascend-toolkit,cann-*,driver,nnal}`。

### 2.4 Ray（资源名必须是 `NPU`）

Smoke / 多步脚本默认 `START_RAY=True`，会自动起 head。手动起集群时：

```bash
# 先清掉错误资源名的旧集群
ray stop -f

ray start --head \
  --port=6379 \
  --dashboard-host=0.0.0.0 \
  --resources='{"NPU": 16}' \
  --disable-usage-stats

export RAY_GCS_ADDRESS=127.0.0.1:6379
export RAY_ADDRESS=http://127.0.0.1:8265
```

若 Ray 已启动但没有 `NPU` 资源，训练脚本会直接退出并提示 `ray stop -f` 后重建。

### 2.5 Docker 代理与 Resident 容器

本机无本地 docker daemon，CLI 走 SSH：

```text
/home/naie/.local/bin/docker  →  ssh -i ~/.ssh/id_ed25519 root@10.1.30.41 docker ...
```

健康检查：

```bash
/home/naie/.local/bin/docker ps --format '{{.Names}} {{.Status}}'
# 期望看到：uni-agent-resident ... Up ...

/home/naie/.local/bin/docker exec uni-agent-resident sh -lc \
  'test -d /mnt/model/lientong/AscendKernelBench/agent && which claude && \
   curl -s -o /dev/null -w "%{http_code}\n" http://10.1.30.41:19001/'
# 期望：目录存在、claude 在 PATH、KernelGYM HTTP 200
```

Resident 容器内需要具备：

- AscendKernelBench + `agent/.claude/skills`
- `claude` CLI
- Ascend Triton 运行时
- 从容器内可访问 `http://10.1.30.41:19001`（KernelGYM）
- 建议 **host network**，以便 Claude 用 `127.0.0.1:<tunnel-port>` 回连 Gateway

SSH 反向隧道要求：`AllowTcpForwarding`、密钥登录可用；task yaml 里 `remote_port: 0` 表示每次分配独立端口。

### 2.6 模型权重

确认本地 HF 目录存在：

```bash
ls /home/naie/work/model/Qwen3.5-0.8B
ls /home/naie/work/model/Qwen3.5-4B
```

缺失时按平台规范放到 `model/`（大文件注意同步配额，可参考工作区根目录 `README.md`）。

---

## 3. 训练数据准备

### 3.1 数据是什么

- 来源：AscendKernelBench Uni-Agent parquet（任务 `triton_op_generator`）
- 上游正式名常见为：`AscendKernelBench/uniagent-training-data/kernelbench_level1.parquet`
- **本机默认源**（已切成 1 行 smoke）：

```text
/home/naie/work/input/input/data/uni_agent/ascend_triton_rl_smoke_20261009152557/train.parquet
```

当前样例：`level1/1_Matrix_scalar_multiplication`。

### 3.2 Schema（VERL 行格式）

列大致为：`data_source`, `ability`, `prompt`, `reward_model`, `extra_info`

关键字段：

- `ability`: `triton_op_generator`
- `prompt`: chat messages（题面 + CLAUDE.md 阶段说明）
- `reward_model.ground_truth`: 参考 `Model` 的 PyTorch 源码
- `extra_info.tools_kwargs.task`: 任务名、sandbox 元数据、`instance_id` / `op_name` 等

### 3.3 怎么切 smoke 数据

训练脚本会自动调用；也可手动：

```bash
cd /home/naie/work/lib/py/uni-agent
source /home/naie/work/lib/py/venv/bin/activate

python3 examples/quickstart/training/make_npu_smoke_data.py \
  --source /home/naie/work/input/input/data/uni_agent/ascend_triton_rl_smoke_20261009152557/train.parquet \
  --target /home/naie/work/input/data/uni_agent/ascend_triton_megatron_smoke/train_one.parquet \
  --rows 1
```

说明：

- 只截断 / 不足时重复行，**不改 schema**
- 换数据源：`export ASCEND_KERNELBENCH_DATA=/path/to/your.parquet`
- 注意：`claude_code_kernel_task` 的 DrKernel preprocess 是**另一条**数据管线；本 recipe 走 AscendKernelBench + KernelGYM

---

## 4. Sandbox 流程

### 4.1 一次 episode 在做什么

1. Gateway 在 Ray 节点开 session（Anthropic-compatible 给 Claude）
2. Task 通过 Docker provider **attach** 已有容器 `uni-agent-resident`（不 start/stop）
3. 可选 SSH reverse tunnel：把 Gateway 映射到 atlas-41 的 `127.0.0.1:<port>`
4. 在容器内建 episode 目录，例如：  
   `/tmp/uni-agent-triton-episodes/<label>-<uuid>/`  
   含 `reference.py`、`INSTRUCTIONS.md`、`output/kernel_code.py`、`final-eval/`
5. Claude Code（`permission_mode: dontAsk`，工具仅 Read/Write/Bash）写 `ModelNew`，可调 KernelGYM
6. Task **独立**再跑一次 KernelGYM 打分 → reward（常见：`1.0` / `0.3` / `0.1` / `0.0`）

### 4.2 Task 配置（本机实际值）

文件：`examples/ascend_triton/task_config_claude_code.yaml`

| 项 | 值 |
|----|-----|
| `sandbox.provider` | `docker` |
| `container_ref` | `uni-agent-resident` |
| `docker_binary` | `/home/naie/.local/bin/docker` |
| `verify_container_image` | `false` |
| `ssh_reverse_tunnel.ssh_host` | `10.1.30.41` |
| `agent_workdir` | `/mnt/model/lientong/AscendKernelBench/agent` |
| `episode_root` | `/tmp/uni-agent-triton-episodes` |
| `kernelgym_url` | `http://10.1.30.41:19001` |
| `entry_point` | `Model`（参考类；候选永远是 `ModelNew`） |
| `CONCURRENCY` | **必须为 1**（常驻容器共享可变文件系统） |

Claude context（须与训练 `prompt_length + response_length = 24064` 对齐）：

- `CLAUDE_CODE_MAX_CONTEXT_TOKENS=26112`
- `CLAUDE_CODE_MAX_OUTPUT_TOKENS=2048`
- `CLAUDE_CODE_FILE_READ_MAX_OUTPUT_TOKENS=1024`
- `CLAUDE_CODE_SIMPLE_SYSTEM_PROMPT=1`

### 4.3 不跑训练、只测任务（可选）

**A. Gateway 推理烟测**（与 RL 同路径，无 GRPO 更新）见 [`README.md`](./README.md) 的 `parallel_infer_verl.py` 命令。

**B. 直连已有 policy API：**

```bash
export MODEL_API_KEY='<policy-api-key>'
python3 examples/ascend_triton/run_single_sample.py \
  --data /home/naie/work/input/input/data/uni_agent/ascend_triton_rl_smoke_20261009152557/train.parquet \
  --task-config examples/ascend_triton/task_config_claude_code.yaml \
  --base-url http://<policy-host>:<port>/v1 \
  --model-name glm-5.2 \
  --index 0 --keep-episodes
```

### 4.4 调试 Claude / 评测

```bash
# Claude 调试日志（容器内）
/home/naie/.local/bin/docker exec uni-agent-resident sh -lc \
  'tail -n 300 /tmp/uni-agent-claude-debug.log'

# 失败 episode 若 retain_failed_episode=true，可在容器内看
# /tmp/uni-agent-triton-episodes/.../output/kernel_code.py
# /tmp/uni-agent-triton-episodes/.../final-eval/result.json
```

成功标志（`task.log`）：

```text
run_task done: task=triton_op_generator reward=1.0 acc=1.0 finished=True
```

注意：**`1/1 scored` ≠ kernel 正确**；要看 `reward=`，以及是否 `decoy_kernel`（纯 `return A * s` 得 0）。

---

## 5. Smoke 训练（0.8B × 1 step）

目标：用最小拓扑验证 **Gateway + Sandbox + Megatron 一步 GRPO** 闭环。

### 5.1 启动

```bash
source /home/naie/work/lib/py/venv/bin/activate
cd /home/naie/work/lib/py/uni-agent

export DEVICE=npu VERL_PLATFORM=huawei
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
unset LOCAL_RANK

bash examples/ascend_triton/train_megatron_smoke.sh
```

### 5.2 默认配置摘要

| 项 | 默认 |
|----|------|
| 模型 | `/home/naie/work/model/Qwen3.5-0.8B` |
| NPU | `NGPUS_PER_NODE=2` |
| Context | `22016 + 2048 = 24064` |
| `n` responses | `2` |
| `TOOL_PARSER` | `qwen3_xml` |
| `SERVED_MODEL_NAME` | `glm-5.2`（路由别名，**不是** checkpoint 名） |
| Session timeout | `1800s` |
| Steps | `trainer.total_training_steps=1` |
| PROJECT | `Uni-Agent-Qwen3.5-0.8B-megatron-npu-smoke` |
| 数据目录 | `input/data/uni_agent/ascend_triton_megatron_smoke/` |

脚本会：切片 parquet → 调 `train_npu_qwen35_megatron_smoke.sh` → `ray job submit`。

常用覆盖：

```bash
MODEL_PATH=/path/to/model \
ASCEND_KERNELBENCH_DATA=/path/to/train.parquet \
NGPUS_PER_NODE=2 \
bash examples/ascend_triton/train_megatron_smoke.sh
```

### 5.3 成功长什么样

- 日志：`/home/naie/work/input/logs/Uni-Agent-Qwen3.5-0.8B-megatron-npu-smoke/<EXP_NAME>/step_1/session-.../`
  - `task.log`：tunnel + `run_task done` + reward
  - `trajectory.json` / `trajectory.npz`
- Checkpoint：`/home/naie/work/input/ckpts/.../global_step_1`
- 启动阶段应看到 `device=npu`、`ray_resource=NPU`、`torch.npu.is_available()`

---

## 6. 多步训练（4B × 3 steps）

目标：在 16 卡上用 TP+PP 跑通 **多步** Megatron GRPO（本机已跑通 `TOTAL_TRAINING_STEPS=3`）。

### 6.1 启动

```bash
source /home/naie/work/lib/py/venv/bin/activate
cd /home/naie/work/lib/py/uni-agent

export DEVICE=npu VERL_PLATFORM=huawei
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
unset LOCAL_RANK

# 默认 3 步；可改步数
TOTAL_TRAINING_STEPS=3 bash examples/ascend_triton/train_megatron_4b_multistep.sh
```

### 6.2 与 Smoke 的差异

| 方面 | Smoke 0.8B | Multistep 4B |
|------|------------|--------------|
| 模型 | Qwen3.5-0.8B | Qwen3.5-4B |
| 卡数 | 2 | **16** |
| 并行 | TP=PP=CP=1 | **TP=2, PP=4, CP=1**（DP=2），`sequence_parallel=True` |
| Rollout TP | 1 | **GEN_TP=2**，`ROLLOUT_GPU_MEM_UTIL=0.40` |
| Steps | 1 | **默认 3** |
| Session timeout | 1800s | **3600s** |
| 数据目录 | `.../ascend_triton_megatron_smoke` | `.../ascend_triton_megatron_4b` |
| PROJECT | `...-0.8B-...-smoke` | `Uni-Agent-Qwen3.5-4B-megatron-npu-triton` |

为何要 TP+PP：Qwen3.5 Megatron 走 **padded BSHD**（无 fla_npu / remove_padding）；**CP 与 BSHD 不兼容**；4B + 24k 在 4 卡 TP=PP=CP=1 会 OOM。

Batch 约束：`real_train_batch = TRAIN_PROMPT_BSZ * N_RESP_PER_PROMPT` 必须能被 DP 整除。当前 `1 * 2 = 2 == DP`。

### 6.3 成功长什么样

参考已跑通实验：`20261022194344_exp`

- 日志：`input/logs/Uni-Agent-Qwen3.5-4B-megatron-npu-triton/<EXP>/step_{1,2,3}/...`
- Checkpoint：`input/ckpts/.../global_step_{1,2,3}`，以及 `latest_checkpointed_iteration.txt`
- 单 step 内 reward 可能混杂（`1.0` 正确 / `0.3` 仅编译 / `0.0` 超时），**只要 step 推进并写出 ckpt 即训练闭环成功**

查看某次 session：

```bash
EXP=20261022194344_exp
ls /home/naie/work/input/logs/Uni-Agent-Qwen3.5-4B-megatron-npu-triton/${EXP}/
grep -R "run_task done" \
  /home/naie/work/input/logs/Uni-Agent-Qwen3.5-4B-megatron-npu-triton/${EXP}/step_1/ \
  --include='task.log' | head
```

---

## 7. 推荐上手顺序（Checklist）

按顺序打勾：

1. [ ] `source lib/py/venv/bin/activate`，`cd lib/py/uni-agent`
2. [ ] `export DEVICE/VERL_PLATFORM/...`，`unset LOCAL_RANK`
3. [ ] `torch.npu.is_available()` 为 True，卡数符合预期
4. [ ] `docker ps` 看到 `uni-agent-resident` Up；容器内 KernelGYM 200；`claude` 可用
5. [ ] SSH 到 `10.1.30.41` 密钥登录正常（docker 代理 + reverse tunnel 都依赖）
6. [ ] 源 parquet、`Qwen3.5-0.8B` / `Qwen3.5-4B` 路径存在
7. [ ] （可选）Gateway 或 `run_single_sample.py` 单样本先过
8. [ ] `bash examples/ascend_triton/train_megatron_smoke.sh` → 有 `global_step_1`
9. [ ] `bash examples/ascend_triton/train_megatron_4b_multistep.sh` → 有 `global_step_1..3`

---

## 8. 常见坑（踩过再写）

1. **在 resident 容器里起训练** → 架构反了；训练只在 Ray 节点跑。
2. **忘记 `unset LOCAL_RANK`** → 整数解析 / 设备绑定异常。
3. **Ray 资源名写成 GPU** → 脚本找不到 NPU；`ray stop -f` 后按上文重建。
4. **`CONCURRENCY>1` + 常驻容器** → episode 目录互相踩；保持 1。
5. **`entry_point` 改成 `ModelNew`** → 参考校验失败；必须保持 `Model`。
6. **纯 PyTorch decoy**（`return A * s`）→ reward 0；必须有 `@triton.jit` + `tl.load`/`tl.store`。
7. **`SERVED_MODEL_NAME=glm-5.2` 与 parser 不匹配** → tool call 解析失败；parser 跟**真实 checkpoint** 模板走（当前 Qwen3.5 用 `qwen3_xml`）。
8. **Claude YAML context 与训练 `prompt+response` 不对齐** → Autocompact thrash / `Prompt is too long`。训练侧实际预算是 Hydra 的长度和，不是只改 YAML 就能放大。
9. **Ray `runtime_env` 里加 `pip:`** → NPU 环境会被克隆到 `/tmp`，易把盘打满；本机 yaml **故意无 pip**，包装在 base venv。
10. **4B 不切 TP/PP 硬跑** → actor update OOM。
11. **KernelGYM 超时 / NPU fatal** → 该 episode reward 0，但训练 step 仍可能完成；查 `task.log` 与 `final-eval`。
12. **工作区软链接** → 平台 S3 同步不支持；训练作业勿依赖 softlink。

---

## 9. 一天内最短命令块（本机）

```bash
source /home/naie/work/lib/py/venv/bin/activate
cd /home/naie/work/lib/py/uni-agent

export DEVICE=npu VERL_PLATFORM=huawei
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
unset LOCAL_RANK

python -c 'import torch,torch_npu; assert torch.npu.is_available()'
/home/naie/.local/bin/docker ps | grep uni-agent-resident

# 1-step smoke
bash examples/ascend_triton/train_megatron_smoke.sh

# 3-step 4B
bash examples/ascend_triton/train_megatron_4b_multistep.sh
```

---

## 10. 相关文档索引

| 文档 | 内容 |
|------|------|
| [`examples/ascend_triton/README.md`](./README.md) | Gateway smoke、tool-parser、Claude context 数学、tunnel |
| `docs/source/concepts/sandbox.md` | `container_ref` attach 语义 |
| `docs/source/quickstart/launch-sandbox.md` | Docker attach 模式 |
| `docs/source/quickstart/npu-training.md` | 通用 NPU / `RAY_DATA_HOME`（偏 VeOmni，路径可对照） |
| `/home/naie/work/README.md` | 平台路径、上传配额、`tmp_no_sync` |
| `examples/claude_code_kernel_task/` | **另一条** ephemeral verifier 镜像流程，勿与本 resident+KernelGYM 混淆 |

---

*文档基于本机已跑通的 smoke 与 `Uni-Agent-Qwen3.5-4B-megatron-npu-triton/20261022194344_exp`（3 step）整理。若主机 IP、容器路径或模型目录变更，优先改 `task_config_claude_code.yaml` 与脚本顶部的默认环境变量。*
