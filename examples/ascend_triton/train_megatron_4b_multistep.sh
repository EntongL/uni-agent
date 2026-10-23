#!/usr/bin/env bash
set -euo pipefail

# Multi-step Ascend Triton Megatron GRPO on Qwen3.5-4B.
#
# Constraints on this NPU stack:
# - Qwen3.5 Megatron stays in padded BSHD (no fla_npu / remove_padding).
# - CP is rejected with BSHD ("Use THD or disable CP").
# - 4B + 24k BSHD OOMs actor update on 4 NPUs with TP=PP=CP=1.
# So shard the actor with TP+PP across the full 16-NPU node.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

export DEVICE=${DEVICE:-npu}
export VERL_PLATFORM=${VERL_PLATFORM:-huawei}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}
unset LOCAL_RANK

NNODES=${NNODES:-1}
# TP=2 PP=4 CP=1 on 16 NPUs => DP=2. real_train_batch (bsz * n) must % DP == 0.
NGPUS_PER_NODE=${NGPUS_PER_NODE:-16}
TRAIN_TP=${TRAIN_TP:-2}
TRAIN_PP=${TRAIN_PP:-4}
TRAIN_CP=${TRAIN_CP:-1}
RAY_DATA_HOME=${RAY_DATA_HOME:-/home/naie/work}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/model/Qwen3.5-4B"}
ASCEND_KERNELBENCH_DATA=${ASCEND_KERNELBENCH_DATA:-/home/naie/work/input/input/data/uni_agent/ascend_triton_rl_smoke_20261009152557/train.parquet}
SMOKE_DIR=${SMOKE_DIR:-/home/naie/work/input/data/uni_agent/ascend_triton_megatron_4b}
RUNTIME_ENV=${RUNTIME_ENV:-examples/quickstart/training/runtime_env_npu_smoke.yaml}
PROJECT_NAME=${PROJECT_NAME:-Uni-Agent-Qwen3.5-4B-megatron-npu-triton}
EXP_NAME=${EXP_NAME:-"$(date +%Y%m%d%H%M%S)_exp"}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-3}

# 22016 + 2048 = 24064. Keep the Claude first-request budget.
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-22016}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
export SESSION_TIMEOUT_SECONDS=${SESSION_TIMEOUT_SECONDS:-3600}
export NUM_WARMUP_BATCHES=${NUM_WARMUP_BATCHES:-1}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-1}
# 1 * 2 = 2 == DP with TP=2 PP=4 on 16 NPUs.
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-2}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1}
export TEST_FREQ=${TEST_FREQ:--1}
export TOOL_PARSER=${TOOL_PARSER:-qwen3_xml}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm-5.2}
export GATEWAY_COUNT=${GATEWAY_COUNT:-1}
export CONCURRENCY=${CONCURRENCY:-1}
export TASK_CONFIG=${TASK_CONFIG:-examples/ascend_triton/task_config_claude_code.yaml}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.40}
# Shard rollout weights across 2 NPUs to leave HBM for the Megatron actor.
export GEN_TP=${GEN_TP:-2}
export START_RAY=${START_RAY:-True}
export RAY_GCS_ADDRESS=${RAY_GCS_ADDRESS:-127.0.0.1:6379}
export RAY_ADDRESS=${RAY_ADDRESS:-http://127.0.0.1:8265}

total_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))

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

echo "Starting Qwen3.5-4B Megatron Triton multi-step"
echo "  model:    ${MODEL_PATH}"
echo "  data:     ${TRAIN_FILE}"
echo "  context:  ${MAX_PROMPT_LENGTH} + ${MAX_RESPONSE_LENGTH}"
echo "  topology: ${NNODES} x ${NGPUS_PER_NODE} NPU (TP=${TRAIN_TP} PP=${TRAIN_PP} CP=${TRAIN_CP} gen_tp=${GEN_TP})"
echo "  steps:    ${TOTAL_TRAINING_STEPS}"
echo "  logs:     ${RAY_DATA_HOME}/input/logs/${PROJECT_NAME}/${EXP_NAME}"

bash examples/quickstart/training/train_npu_qwen35_megatron_smoke.sh \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.total_epochs="${TOTAL_TRAINING_STEPS}" \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    'trainer.logger=["console"]' \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}" \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
    actor_rollout_ref.actor.megatron.context_parallel_size="${TRAIN_CP}" \
    actor_rollout_ref.actor.megatron.sequence_parallel=True \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
    actor_rollout_ref.ref.megatron.context_parallel_size="${TRAIN_CP}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${total_len}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${total_len}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${total_len}" \
    "$@"
