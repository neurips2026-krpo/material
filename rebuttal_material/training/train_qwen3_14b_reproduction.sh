#!/usr/bin/env bash
set -euo pipefail

: "${EASYR1_ROOT:?Set EASYR1_ROOT}"
: "${MODEL_PATH:?Set MODEL_PATH to the Qwen3-14B base model}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${EMBEDDING_API_URL:?Set EMBEDDING_API_URL}"
export EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen3-Embedding-8B}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "${EASYR1_ROOT}"
python3 -m verl.trainer.main \
  config="${EASYR1_ROOT}/tcm_example/config.yaml" \
  data.train_files="${EASYR1_ROOT}/datasets/tcm_train_2nd_datasets.jsonl" \
  data.val_files="${EASYR1_ROOT}/datasets/tcm_val_2nd_datasets.jsonl" \
  data.prompt_key=instruction \
  data.answer_key=output \
  data.max_response_length=16384 \
  worker.actor.model.model_path="${MODEL_PATH}" \
  worker.actor.fsdp.torch_dtype=bf16 \
  worker.actor.optim.strategy=adamw_bf16 \
  worker.actor.optim.lr=2e-6 \
  worker.rollout.n=10 \
  worker.rollout.temperature=0.8 \
  worker.rollout.tensor_parallel_size=4 \
  worker.reward.reward_function="${EASYR1_ROOT}/examples/reward_function/tcm_reward_v2.py:compute_score" \
  algorithm.online_filtering=false \
  trainer.total_epochs=20 \
  trainer.max_steps=100 \
  trainer.n_gpus_per_node=8 \
  worker.rollout.gpu_memory_utilization=0.4 \
  worker.actor.offload.offload_params=true \
  trainer.logger='["file"]' \
  trainer.save_checkpoint_path="${OUTPUT_DIR}" \
  trainer.experiment_name="qwen3_14b_tcm_2k_reproduction"
