#!/usr/bin/env bash

set -x
mkdir logs

PARTITION=VC

#MODEL="conv_halo_v2_timm_tiny" 
#CKPT="/mnt/petrelfs/share_data/shimin/share_checkpoint/halonet/halonet_v2_tiny/checkpoint-best.pth"

MODEL="conv_swin_tiny" 
CKPT="/mnt/petrelfs/share_data/shimin/share_checkpoint/swin/swin_tiny/checkpoint-best.pth"

#MODEL="conv_convnext_v2_tiny" 
#CKPT="/mnt/petrelfs/share_data/shimin/share_checkpoint/convnext/convnext_tiny/checkpoint-best-ema.pth"

# MODEL="dcn_v3_tiny" 
# CKPT="/mnt/petrelfs/share_data/shimin/share_checkpoint/dcnv3/dcnv3_tiny/checkpoint-best-ema.pth"

DESC="eval_invariance" 

# key hyperparameters
TOTAL_BATCH_SIZE="256"

JOB_NAME=${MODEL}
PROJECT_NAME="${MODEL}_1k_${DESC}"

GPUS=${GPUS:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}
QUOTA_TYPE="reserved"

CPUS_PER_TASK=${CPUS_PER_TASK:-12}

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
    python -u invariance_eval_all.py \
    --model ${MODEL} \
    --resume ${CKPT} \
    --batch_size $((TOTAL_BATCH_SIZE/GPUS_PER_NODE)) \
    --input_size 224 \
    --crop_pct 0.875 \
    --data_set IMNET1k \
    --data_path /mnt/cache/share/images/ \
    --data_on_memory false \
    --nb_classes 1000 \
    --use_amp false \
    --output_dir "/mnt/petrelfs/${USER}/model_evaluation/invariance/${PROJECT_NAME}"
