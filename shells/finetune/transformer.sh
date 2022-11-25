#!/usr/bin/env bash

set -x
mkdir logs

PARTITION=VC
MODEL=$1
DESC="unified_config" 

# key hyperparameters
TOTAL_BATCH_SIZE="1024"
LR="4e-5"
INIT_LR="4e-8"
END_LR="4e-7"

JOB_NAME=${MODEL}
PROJECT_NAME="${MODEL}_22k_${DESC}"

UPDATE_FREQ=1
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
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
    python -u main.py \
    --model ${MODEL} \
    --epochs 30 \
    --update_freq ${UPDATE_FREQ} \
    --batch_size $((TOTAL_BATCH_SIZE/GPUS/UPDATE_FREQ)) \
    --lr ${LR} \
    --warmup_init_lr ${INIT_LR} \
    --min_lr ${END_LR} \
    --data_set IMNET1k \
    --data_path "/mnt/cache/share/images/" \
    --output_dir "/mnt/petrelfs/wangweiyun/model_evaluation/${PROJECT_NAME}" \
    --warmup_epochs 5 \
    --input_size 384 \
    --model_ema false \
    --model_ema_eval false \
    --weight_decay 1e-8 \
    --use_checkpoint true
