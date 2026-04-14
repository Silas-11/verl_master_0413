set -x

# ===================================== Environment & Paths =====================================
ENGINE=${1:-vllm}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-VL-8B-Instruct"}

project_name='GRPO-Qwen3_vl'
exp_name='GRPO-Qwen3_vl-8B-npu'
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}

HF_MODEL_PATH=/mnt/chubao/lujiawei30/hw_dwq/full_async/ckpt/Qwen3-VL-8B-Instruct
train_path=/mnt/chubao/lujiawei30/hw_dwq/full_async/data/geo3k/train.parquet
test_path=/mnt/chubao/lujiawei30/hw_dwq/full_async/data/geo3k/test.parquet

# ===================================== GPU Allocation =====================================
n_gpus_per_node=16
n_nodes=1

# ===================================== Data =====================================
train_prompt_bsz=512
n_resp_per_prompt=4
train_prompt_mini_bsz=128

DATA_CONFIG="
    data.train_files=${train_path} \
    data.val_files=${test_path} \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.shuffle=False"

# ===================================== Actor Model & Optim =====================================
ACTOR_CONFIG="
    actor_rollout_ref.model.path=${HF_MODEL_PATH} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=5120 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
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
gen_tp=1

ROLLOUT_CONFIG="
    actor_rollout_ref.rollout.name=${ENGINE} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
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
rollout_is=sequence
rollout_is_threshold=2.0
rollout_is_batch_normalize=true
rollout_rs=token_k1
rollout_rs_threshold=0.6_1.6

ALGORITHM_CONFIG="
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.rollout_is=${rollout_is} \
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold} \
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize} \
    algorithm.rollout_correction.rollout_rs=${rollout_rs} \
    algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold}"

# ===================================== Trainer =====================================
total_epochs=200
test_freq=5

TRAINER_CONFIG="
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name='verl_grpo_example_geo3k' \
    trainer.experiment_name='qwen3_vl_8b_fsdp2_async' \
    trainer.n_gpus_per_node=${n_gpus_per_node} \
    trainer.nnodes=${n_nodes} \
    trainer.default_local_dir=${CKPTS_DIR} \
    trainer.resume_mode=auto \
    trainer.val_before_train=False \
    trainer.save_freq=-1 \
    trainer.test_freq=${test_freq} \
    trainer.total_epochs=${total_epochs}"

# ===================================== Launch =====================================
python3 -m verl.trainer.main_ppo \
    $DATA_CONFIG \
    $ACTOR_CONFIG \
    $REF_CONFIG \
    $ROLLOUT_CONFIG \
    $ALGORITHM_CONFIG \
    $TRAINER_CONFIG