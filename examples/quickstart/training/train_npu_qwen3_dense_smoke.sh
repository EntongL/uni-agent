#!/usr/bin/env bash
set -xeuo pipefail

# Single-node dense Qwen3 smoke recipe.
# It intentionally keeps the same VeOmni + Agent Framework path as the full
# NPU recipe, but removes MoE-only settings and uses tiny sequence/batch sizes.

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}

project_name=${PROJECT_NAME:-"Uni-Agent-Qwen3-0.6B-veomni-npu-smoke"}
exp_name=${EXP_NAME:-"$(date +%Y%m%d%H%M%S)_exp"}

RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-0.6B"}
RUNTIME_ENV=${RUNTIME_ENV:-"examples/quickstart/training/runtime_env_npu_smoke.yaml"}
TASK_CONFIG=${TASK_CONFIG:-"examples/quickstart/training/task_config_react_npu_smoke.yaml"}

# If TRAIN_FILE / TEST_FILE are not supplied, make tiny files from the normal
# Uni-Agent parquet files before submitting the Ray job.
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/uni_agent/npu_smoke_train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/uni_agent/npu_smoke_test.parquet"}
SOURCE_TRAIN_FILE=${SOURCE_TRAIN_FILE:-"${RAY_DATA_HOME}/data/uni_agent/swe_rebench_filtered_1150.parquet"}
SOURCE_TEST_FILE=${SOURCE_TEST_FILE:-"${SOURCE_TRAIN_FILE}"}
SMOKE_TRAIN_ROWS=${SMOKE_TRAIN_ROWS:-1}
SMOKE_TEST_ROWS=${SMOKE_TEST_ROWS:-1}

CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
AGENT_LOG_DIR=${AGENT_LOG_DIR:-"${RAY_DATA_HOME}/logs/${project_name}/${exp_name}"}

# Generic Qwen3 uses a different tool-call format from Qwen3-Coder.
TOOL_PARSER=${TOOL_PARSER:-"hermes"}
GATEWAY_COUNT=${GATEWAY_COUNT:-1}
CONCURRENCY=${CONCURRENCY:-1}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-"$(basename "${MODEL_PATH}")"}

# Keep Qwen3's thinking mode off for the first tool-calling smoke test. Set to
# True if you specifically want to test reasoning-mode chat templates.
ENABLE_THINKING=${ENABLE_THINKING:-False}

rollout_mode=${ROLLOUT_MODE:-"async"}
rollout_name=${ROLLOUT_NAME:-"vllm"}

# Standard GRPO baseline; no router replay or rollout correction in smoke mode.
adv_estimator=${ADV_ESTIMATOR:-grpo}
loss_mode=${LOSS_MODE:-vanilla}
use_kl_in_reward=${USE_KL_IN_REWARD:-False}
kl_coef=${KL_COEF:-0.0}
use_kl_loss=${USE_KL_LOSS:-False}
kl_loss_coef=${KL_LOSS_COEF:-0.0}
clip_ratio_low=${CLIP_RATIO_LOW:-0.2}
clip_ratio_high=${CLIP_RATIO_HIGH:-0.28}

# Qwen3-0.6B has a 32K context window. These defaults are deliberately small.
max_prompt_length=${MAX_PROMPT_LENGTH:-1024}
max_response_length=${MAX_RESPONSE_LENGTH:-2048}
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))

# Dense-model parallelism: one NPU by default. Increase only after the one-NPU
# path works; USP and TP must divide the available NPU topology.
usp_size=${USP_SIZE:-1}
gen_tp=${GEN_TP:-1}
infer_dp=${INFER_DP:-1}
use_dynamic_bsz=${USE_DYNAMIC_BSZ:-True}
offload=${OFFLOAD:-True}
attn_impl=${ATTN_IMPL:-flash_attention_2}
rollout_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.60}

# Small colocate_async batch. This still performs rollout, reward, and update.
train_prompt_bsz=${TRAIN_PROMPT_BSZ:-1}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-2}
train_prompt_mini_bsz=${PPO_MINI_BATCH_SIZE:-1}
num_warmup_batches=${NUM_WARMUP_BATCHES:-1}
test_freq=${TEST_FREQ:-1}

if [[ ! -f "${TRAIN_FILE}" ]]; then
    python3 examples/quickstart/training/make_npu_smoke_data.py \
        --source "${SOURCE_TRAIN_FILE}" \
        --target "${TRAIN_FILE}" \
        --rows "${SMOKE_TRAIN_ROWS}"
fi

if [[ ! -f "${TEST_FILE}" ]]; then
    python3 examples/quickstart/training/make_npu_smoke_data.py \
        --source "${SOURCE_TEST_FILE}" \
        --target "${TEST_FILE}" \
        --rows "${SMOKE_TEST_ROWS}"
fi

ray job submit --no-wait --runtime-env "${RUNTIME_ENV}" \
    -- env RAY_OVERRIDE_JOB_RUNTIME_ENV=1 \
    python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=colocate_async \
    trainer.v1.colocate_async.num_warmup_batches=${num_warmup_batches} \
    transfer_queue.enable=True \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    model_engine=veomni \
    actor_rollout_ref.actor.veomni.param_offload=True \
    actor_rollout_ref.actor.veomni.optimizer_offload=${offload} \
    actor_rollout_ref.actor.veomni.enable_full_shard=True \
    actor_rollout_ref.actor.veomni.ulysses_parallel_size=${usp_size} \
    actor_rollout_ref.actor.veomni.attn_implementation=${attn_impl} \
    actor_rollout_ref.actor.veomni.rms_norm_implementation=npu \
    actor_rollout_ref.actor.veomni.rotary_pos_emb_implementation=npu \
    actor_rollout_ref.actor.veomni.swiglu_mlp_implementation=eager \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    ++actor_rollout_ref.rollout.multi_turn.format=${TOOL_PARSER} \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
    ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT} \
    ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${AGENT_LOG_DIR}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${CONCURRENCY} \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG}" \
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name="${SERVED_MODEL_NAME}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_mem_util} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.data_parallel_size=${infer_dp} \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.8 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.8 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.nccl_timeout=9600 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    reward.reward_manager.name=dapo \
    reward.custom_reward_function.path=pkg://uni_agent.framework.task_runner \
    reward.custom_reward_function.name=score_from_runner_result \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=False \
    trainer.device=npu \
    trainer.save_freq=1 \
    trainer.total_epochs=1 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.test_freq=${test_freq} \
    "$@"
