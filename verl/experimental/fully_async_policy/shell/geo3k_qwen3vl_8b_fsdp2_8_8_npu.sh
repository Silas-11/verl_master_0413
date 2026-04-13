set -x

# ===================================== Environment & Paths =====================================
export CUDA_DEVICE_MAX_CONNECTIONS=1  # For megatron communication/computation overlapping

HF_MODEL_PATH=${HF_MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-VL-8B-Instruct"}
train_path=$HOME/data/geo3k/train.parquet
test_path=$HOME/data/geo3k/test.parquet

# ===================================== Rollout Mode =====================================
rollout_mode="async"
rollout_name="vllm"  # sglang or vllm

return_raw_chat="False"
if [ "$rollout_mode" = "async" ]; then
    export VLLM_USE_V1=1
    return_raw_chat="True"
fi

# ===================================== GPU Allocation =====================================
n_gpus_rollout=16
n_gpus_training=16
n_nodes_rollout=1
n_nodes_train=1

# ===================================== Data =====================================
train_prompt_bsz=0
gen_prompt_bsz=1
n_resp_per_prompt=4

DATA_CONFIG="
    data.train_files=${train_path} \
    data.val_files=${test_path} \
    data.train_batch_size=${train_prompt_bsz} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=${return_raw_chat}"

# ===================================== Actor Model & Optim =====================================
train_prompt_mini_bsz=128

ACTOR_CONFIG="
    actor_rollout_ref.model.path=${HF_MODEL_PATH} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True"

# ===================================== Ref Config =====================================
REF_CONFIG="
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=5120"

# ===================================== Rollout Config =====================================
ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=5120 \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0"

# ===================================== Algorithm =====================================
ALGORITHM_CONFIG="
    algorithm.use_kl_in_reward=False \
    algorithm.adv_estimator=grpo"

# ===================================== Trainer =====================================
total_epochs=200
test_freq=5
total_rollout_steps=$(( 512 * 100 ))

TRAINER_CONFIG="
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo_example_geo3k' \
    trainer.experiment_name='qwen3_vl_8b_fsdp2_async' \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs} \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.resume_mode=auto \
    trainer.nnodes=${n_nodes_train} \
    trainer.n_gpus_per_node=${n_gpus_training} \
    rollout.nnodes=${n_nodes_rollout} \
    rollout.n_gpus_per_node=${n_gpus_rollout} \
    rollout.total_rollout_steps=${total_rollout_steps}"

# ===================================== Async Training =====================================
staleness_threshold=0.1
trigger_parameter_sync_step=4
require_batches=2
partial_rollout=True

ASYNC_CONFIG="
    async_training.staleness_threshold=${staleness_threshold} \
    async_training.trigger_parameter_sync_step=${trigger_parameter_sync_step} \
    async_training.require_batches=${require_batches} \
    async_training.partial_rollout=${partial_rollout}"

# ===================================== Launch =====================================
python -m verl.experimental.fully_async_policy.fully_async_main \
    --config-path=config \
    --config-name='fully_async_ppo_trainer.yaml' \
    $DATA_CONFIG \
    $ACTOR_CONFIG \
    $REF_CONFIG \
    $ROLLOUT_CONFIG \
    $ALGORITHM_CONFIG \
    $TRAINER_CONFIG \
    $ASYNC_CONFIG