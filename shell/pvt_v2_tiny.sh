#!/usr/bin/env bash

set -x
mkdir logs

PARTITION=VC
MODEL="conv_pvt_v2_tiny"
DESC="erf_analysis" 
INPUT_SIZE=448

# key hyperparameters
JOB_NAME=${MODEL}
PROJECT_NAME="${MODEL}_1k_${DESC}_${INPUT_SIZE}"
QUOTA_TYPE="spot"

CPUS_PER_TASK=${CPUS_PER_TASK:-8}
SRUN_ARGS=${SRUN_ARGS:-""}

srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA_TYPE} \
    --async \
    --output="logs/${PROJECT_NAME}.out" \
    --error="logs/${PROJECT_NAME}.err" \
    ${SRUN_ARGS} \
    python -u valset_erf.py \
    --model ${MODEL} \
    --input_size ${INPUT_SIZE} \
    --data_path /mnt/cache/share/images/ \
    --weights "/mnt/petrelfs/share_data/shimin/share_checkpoint/pvt/conv_pvt_v2_tiny_1k_unified_config/checkpoint-best.pth" \
    --num_data 200
    #--output_dir "/mnt/petrelfs/${USER}/model_evaluation/${PROJECT_NAME}"
    
# sh ./shell/pvt_v2_tiny.sh