#!/usr/bin/env bash
set -euo pipefail

# One Megatron GRPO update for the resident KernelGYM Triton task on Qwen3.5-0.8B.
# The actor follows examples/quickstart/training/train_npu_qwen35_megatron_smoke.sh
# (MindSpeed, padded BSHD, PyTorch GDN). Context and session timeout follow
# examples/claude_code_kernel_task: the Gateway budget must cover Claude's
# system prompt, and a stuck session must not hold the trainer step.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

export DEVICE=${DEVICE:-npu}
export VERL_PLATFORM=${VERL_PLATFORM:-huawei}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}
unset LOCAL_RANK

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
RAY_DATA_HOME=${RAY_DATA_HOME:-/home/naie/work}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/model/Qwen3.5-0.8B"}
ASCEND_KERNELBENCH_DATA=${ASCEND_KERNELBENCH_DATA:-/home/naie/work/input/input/data/uni_agent/ascend_triton_rl_smoke_20261009152557/train.parquet}
SMOKE_DIR=${SMOKE_DIR:-/home/naie/work/input/data/uni_agent/ascend_triton_megatron_smoke}
RUNTIME_ENV=${RUNTIME_ENV:-examples/quickstart/training/runtime_env_npu_smoke.yaml}
PROJECT_NAME=${PROJECT_NAME:-Uni-Agent-Qwen3.5-0.8B-megatron-npu-smoke}
EXP_NAME=${EXP_NAME:-"$(date +%Y%m%d%H%M%S)_exp"}

# 22016 + 2048 = 24064. Claude's first request is about 21k tokens. Qwen3.5
# Megatron stays in padded BSHD, so a larger window OOMs the actor update.
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-22016}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export SESSION_TIMEOUT_SECONDS=${SESSION_TIMEOUT_SECONDS:-1800}
export NUM_WARMUP_BATCHES=${NUM_WARMUP_BATCHES:-1}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-1}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-2}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1}
export TEST_FREQ=${TEST_FREQ:--1}
export TOOL_PARSER=${TOOL_PARSER:-qwen3_xml}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm-5.2}
export GATEWAY_COUNT=${GATEWAY_COUNT:-1}
export CONCURRENCY=${CONCURRENCY:-1}
export TASK_CONFIG=${TASK_CONFIG:-examples/ascend_triton/task_config_claude_code.yaml}
export START_RAY=${START_RAY:-True}
export RAY_GCS_ADDRESS=${RAY_GCS_ADDRESS:-127.0.0.1:6379}
export RAY_ADDRESS=${RAY_ADDRESS:-http://127.0.0.1:8265}

mkdir -p "${SMOKE_DIR}"
TRAIN_FILE="${SMOKE_DIR}/train_one.parquet"
TEST_FILE="${SMOKE_DIR}/test_one.parquet"
python3 examples/quickstart/training/make_npu_smoke_data.py \
    --source "${ASCEND_KERNELBENCH_DATA}" \
    --target "${TRAIN_FILE}" \
    --rows "${TRAIN_PROMPT_BSZ}"
python3 examples/quickstart/training/make_npu_smoke_data.py \
    --source "${ASCEND_KERNELBENCH_DATA}" \
    --target "${TEST_FILE}" \
    --rows 1

export MODEL_PATH TRAIN_FILE TEST_FILE RUNTIME_ENV PROJECT_NAME EXP_NAME
export RAY_DATA_HOME NNODES NGPUS_PER_NODE

echo "Starting Qwen3.5-0.8B Megatron Triton smoke"
echo "  model:    ${MODEL_PATH}"
echo "  data:     ${TRAIN_FILE}"
echo "  context:  ${MAX_PROMPT_LENGTH} + ${MAX_RESPONSE_LENGTH}"
echo "  topology: ${NNODES} x ${NGPUS_PER_NODE} NPU"
echo "  logs:     ${RAY_DATA_HOME}/input/logs/${PROJECT_NAME}/${EXP_NAME}"

bash examples/quickstart/training/train_npu_qwen35_megatron_smoke.sh \
    trainer.total_training_steps=1 \
    trainer.total_epochs=1 \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    'trainer.logger=["console"]' \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}" \
    "$@"
