#!/usr/bin/env bash

set -x
mkdir logs

PARTITION=VC
MODEL=$1
CKPT="/mnt/petrelfs/wangweiyun/model_ckpt/${MODEL}.pth"

# key hyperparameters
TOTAL_BATCH_SIZE="128"

JOB_NAME=${MODEL}

GPUS=${GPUS:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-1}
QUOTA_TYPE="spot"


srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA_TYPE} \
    python -u main.py \
    --model ${MODEL} \
    --eval true \
    --resume ${CKPT} \
    --batch_size $((TOTAL_BATCH_SIZE/GPUS_PER_NODE)) \
    --input_size 224 \
    --data_set IMNET1k \
    --data_path /mnt/cache/share/images/ \
    --data_on_memory false \
    --nb_classes 1000 \
    --use_amp false \
