#!/usr/bin/env bash

set -x
mkdir logs

PARTITION=VC
MODEL=$1
CKPT="/mnt/petrelfs/wangweiyun/model_ckpt_22k/${MODEL}.pth"
DESC="unified_config" 

# key hyperparameters
TOTAL_BATCH_SIZE="512"
LR="5e-5"
INIT_LR="0"
END_LR="1e-6"

JOB_NAME=${MODEL}
PROJECT_NAME="${MODEL}_22k_ft_${DESC}"

UPDATE_FREQ=2
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
    --finetune ${CKPT} \
    --epochs 30 \
    --update_freq ${UPDATE_FREQ} \
    --batch_size $((TOTAL_BATCH_SIZE/GPUS/UPDATE_FREQ)) \
    --lr ${LR} \
    --warmup_init_lr ${INIT_LR} \
    --min_lr ${END_LR} \
    --data_set CEPH22k \
    --data_path "s3://image22k/" \
    --output_dir "/mnt/petrelfs/wangweiyun/model_evaluation/${PROJECT_NAME}" \
    --warmup_epochs 0 \
    --input_size 384 \
    --model_ema false \
    --model_ema_eval false \
    --weight_decay 1e-8 \
    --mixup 0 \
    --cutmix 0 \
    --use_amp true \
    --use_checkpoint false \
    