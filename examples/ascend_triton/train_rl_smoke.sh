#!/usr/bin/env bash
set -euo pipefail

# One trainer update for the Ascend Triton task. Run from any directory on the
# NPU/Ray head host after exporting the required paths and topology variables.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen3-Coder-30B-A3B-Instruct checkpoint}"
: "${ASCEND_KERNELBENCH_DATA:?Set ASCEND_KERNELBENCH_DATA to kernelbench_level1.parquet}"
: "${NNODES:?Set NNODES to the training node count}"
: "${NGPUS_PER_NODE:?Set NGPUS_PER_NODE to the Ascend NPU count per node}"

# Match verl's Huawei platform detection and Ray device-binding mode before
# checking or starting the local cluster.
export DEVICE=${DEVICE:-npu}
export VERL_PLATFORM=${VERL_PLATFORM:-huawei}
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}
unset LOCAL_RANK

if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    visible_npus=""
    for ((npu_index = 0; npu_index < NGPUS_PER_NODE; npu_index++)); do
        visible_npus+="${visible_npus:+,}${npu_index}"
    done
    export ASCEND_RT_VISIBLE_DEVICES="${visible_npus}"
fi

# The single-node smoke can create its own Ray head. A multi-node run must use
# the cluster started by the user's NPU launcher on every node.
RAY_GCS_ADDRESS=${RAY_GCS_ADDRESS:-127.0.0.1:6379}
if ! ray status --address="${RAY_GCS_ADDRESS}" >/dev/null 2>&1; then
    if [[ "${NNODES}" != 1 ]]; then
        echo "No Ray cluster at ${RAY_GCS_ADDRESS}; start the multi-node Ray cluster on all ${NNODES} nodes first." >&2
        exit 1
    fi
    ray start --head \
        --port=6379 \
        --dashboard-host=0.0.0.0 \
        --resources="{\"NPU\": ${NGPUS_PER_NODE}}" \
        --disable-usage-stats
elif ! ray status --address="${RAY_GCS_ADDRESS}" 2>/dev/null | grep -q "NPU"; then
    echo "Ray cluster at ${RAY_GCS_ADDRESS} has no NPU resource; start it with the NPU resource enabled." >&2
    exit 1
fi

python3 -c '
import torch
import torch_npu  # noqa: F401
from verl.plugin.platform import get_platform

platform = get_platform()
print(f"verl platform: {platform.__class__.__name__}, device={platform.device_name}, ray_resource={platform.ray_resource_name()}")
assert platform.device_name == "npu", platform.device_name
assert platform.ray_resource_name() == "NPU", platform.ray_resource_name()
assert torch.npu.is_available(), "torch.npu.is_available() is false"
'

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
RUNTIME_ENV=${RUNTIME_ENV:-"examples/quickstart/training/runtime_env_npu_smoke.yaml"}
SMOKE_DIR=${SMOKE_DIR:-"${RAY_DATA_HOME}/data/uni_agent/ascend_triton_rl_smoke"}
PROJECT_NAME=${PROJECT_NAME:-"Uni-Agent-Ascend-Triton-RL-Smoke"}
EXP_NAME=${EXP_NAME:-"$(date +%Y%m%d%H%M%S)_smoke"}

for required_file in \
    "${MODEL_PATH}/config.json" \
    "${ASCEND_KERNELBENCH_DATA}" \
    "${RUNTIME_ENV}" \
    "examples/ascend_triton/task_config_claude_code.yaml" \
    "examples/quickstart/training/train_npu_qwen3_moe.sh"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done

TRAIN_FILE="${SMOKE_DIR}/train_one.parquet"
TEST_FILE="${SMOKE_DIR}/test_one.parquet"
mkdir -p "${SMOKE_DIR}"
python3 examples/quickstart/training/make_npu_smoke_data.py \
    --source "${ASCEND_KERNELBENCH_DATA}" \
    --target "${TRAIN_FILE}" \
    --rows 1
python3 examples/quickstart/training/make_npu_smoke_data.py \
    --source "${ASCEND_KERNELBENCH_DATA}" \
    --target "${TEST_FILE}" \
    --rows 1

export TRAIN_FILE TEST_FILE RUNTIME_ENV PROJECT_NAME EXP_NAME
export TASK_CONFIG=examples/ascend_triton/task_config_claude_code.yaml
export TOOL_PARSER=${TOOL_PARSER:-qwen3_xml}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-glm-5.2}
export GATEWAY_COUNT=${GATEWAY_COUNT:-1}
export CONCURRENCY=${CONCURRENCY:-1}
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-1}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-2}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1}
export MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-61440}
export TEST_FREQ=-1

echo "Starting one-step Ascend Triton RL smoke"
echo "  model:       ${MODEL_PATH}"
echo "  dataset:     ${TRAIN_FILE} (first row; also used as val placeholder)"
echo "  topology:    ${NNODES} node(s) x ${NGPUS_PER_NODE} NPU(s)"
echo "  context:     ${MAX_PROMPT_LENGTH} + ${MAX_RESPONSE_LENGTH} tokens"
echo "  parser/name: ${TOOL_PARSER} / ${SERVED_MODEL_NAME}"
echo "  checkpoint:  ${RAY_DATA_HOME}/ckpts/${PROJECT_NAME}/${EXP_NAME}"

bash examples/quickstart/training/train_npu_qwen3_moe.sh \
    trainer.total_training_steps=1 \
    trainer.total_epochs=1 \
    trainer.save_freq=1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    'trainer.logger=["console"]'
