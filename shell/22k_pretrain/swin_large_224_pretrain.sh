#!/usr/bin/env bash

set -x
mkdir logs

PARTITION=VC
#MODEL=$1
MODEL="unified_swin_large_224_w7"
DESC="unified_config_run_test" 

# key hyperparameters
TOTAL_BATCH_SIZE="512"
LR="1e-3"
INIT_LR="1e-6"
END_LR="1e-5"

JOB_NAME=${MODEL}
PROJECT_NAME="${MODEL}_1k_${DESC}"

UPDATE_FREQ=1
GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
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
    --output="logs/${PROJECT_NAME}.out" \
    --error="logs/${PROJECT_NAME}.err" \
    ${SRUN_ARGS} \
    python -u main.py \
    --model ${MODEL} \
    --epochs 300 \
    --update_freq ${UPDATE_FREQ} \
    --batch_size $((TOTAL_BATCH_SIZE/GPUS/UPDATE_FREQ)) \
    --lr ${LR} \
    --warmup_init_lr ${INIT_LR} \
    --min_lr ${END_LR} \
    --use_checkpoint True \
    --data_set "CEPH22k" \
    --data_path "s3://image22k/" \
    --nb_classes 1000 \
    --output_dir "/mnt/petrelfs/shimin/model_evaluation/${PROJECT_NAME}"

# sh shell/22k_pretrain/swin_large_224_pretrain.sh