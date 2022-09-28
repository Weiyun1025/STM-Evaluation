#!/usr/bin/env bash

set -x
mkdir logs

PARTITION=VC
MODEL="conv_halo_v2_no_train_mask_timm_tiny"
DESC="eval" 

# key hyperparameters
TOTAL_BATCH_SIZE="1024"

JOB_NAME=${MODEL}
PROJECT_NAME="${MODEL}_1k_${DESC}"

GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
QUOTA_TYPE="spot"

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
    --eval true \
    --resume "/mnt/petrelfs/wangweiyun/model_evaluation/conv_halo_v2_timm_tiny_1k_unified_config/checkpoint-best.pth" \
    --batch_size $((TOTAL_BATCH_SIZE/GPUS_PER_NODE)) \
    --model_ema true \
    --model_ema_decay 0.9999 \
    --model_ema_eval true \
    --input_size 224 \
    --data_set IMNET1k \
    --data_path /mnt/cache/share/images/ \
    --data_on_memory false \
    --nb_classes 1000 \
    --use_amp true \
    --output_dir "/mnt/petrelfs/${USER}/model_evaluation/eval/${PROJECT_NAME}"
