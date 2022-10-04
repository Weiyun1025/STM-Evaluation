#!/usr/bin/env bash

set -x
mkdir logs

PARTITION=VC
MODEL="conv_halo_v2_tiny"

JOB_NAME=${MODEL}
PROJECT_NAME="${MODEL}"

GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
QUOTA_TYPE="auto"

CPUS_PER_TASK=${CPUS_PER_TASK:-12}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA_TYPE} \
    --async \
    --output="logs/${PROJECT_NAME}.out" \
    --error="logs/${PROJECT_NAME}.err" \
    ${SRUN_ARGS} \
    python -u cam.py \
    --model ${MODEL} \
    --ckpt_dir "/mnt/petrelfs/share_data/shimin/share_checkpoint/halonet/halonet_v2_tiny" \
    --data_dir "/mnt/cache/share/images/" \
    --output_dir "/mnt/petrelfs/${USER}/model_evaluation/cam" \
    --cam --save_raw --cat
    
# sh shell/1k_pretrain/convnext_tiny_1k_224.sh