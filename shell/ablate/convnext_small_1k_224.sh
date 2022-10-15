#!/usr/bin/env bash

set -x
mkdir logs

PARTITION=VC
MODEL=$1
DESC="unified_config" 

# key hyperparameters
TOTAL_BATCH_SIZE="4096"
LR="4e-3"
INIT_LR="0"
END_LR="1e-6"
DROP_PATH="0.4"

JOB_NAME=${MODEL}
PROJECT_NAME="${MODEL}_1k_${DESC}"

GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE="reserved"

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
    --epochs 300 \
    --batch_size $((TOTAL_BATCH_SIZE/GPUS)) \
    --warmup_epochs 20 \
    --lr ${LR} \
    --warmup_init_lr ${INIT_LR} \
    --min_lr ${END_LR} \
    --opt adamw \
    --clip_grad 5.0 \
    --drop_path ${DROP_PATH} \
    --weight_decay 0.05 \
    --layer_scale_init_value 1e-6 \
    --smoothing 0.1 \
    --model_ema true \
    --model_ema_decay 0.9999 \
    --model_ema_eval true \
    --input_size 224 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --mixup_prob 1.0 \
    --mixup_switch_prob 0.5 \
    --aa rand-m9-mstd0.5-inc1 \
    --repeated_aug false \
    --reprob 0.25 \
    --color_jitter 0.4 \
    --crop_pct 0.875 \
    --data_set IMNET1k \
    --data_path /mnt/cache/share/images/ \
    --data_on_memory false \
    --nb_classes 1000 \
    --use_amp true \
    --save_ckpt true \
    --enable_wandb true \
    --project 'model evaluation' \
    --name ${PROJECT_NAME} \
    --output_dir "/mnt/petrelfs/${USER}/model_evaluation/${PROJECT_NAME}"
    
# sh shell/1k_pretrain/convnext_small_1k_224.sh