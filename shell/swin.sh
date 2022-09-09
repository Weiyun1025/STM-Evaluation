#!/usr/bin/env bash
# swin_[ tiny | small | base | large ]

set -x
mkdir logs

PARTITION=VC
MODEL="swin_tiny"
DESC="unified" 

JOB_NAME=${MODEL}
PROJECT_NAME="${MODEL}_1k_${DESC}"

GPUS=${GPUS:-2}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
QUOTA_TYPE="spot"

CPUS_PER_TASK=${CPUS_PER_TASK:-12}
SRUN_ARGS=${SRUN_ARGS:-""}

# todo: dropout
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    --quotatype=${QUOTA_TYPE} \
    ${SRUN_ARGS} \
    python -u main.py \
    --model ${MODEL} \
    --epochs 300 \
    --batch_size 512 \
    --warmup_epochs 20 \
    --lr 1e-3\
    --warmup_init_lr 1e-6\
    --min_lr 1e-6\
    --opt adamw \
    --clip_grad 5.0 \
    --drop_path 0.2 \
    --weight_decay 0.05 \
    --layer_scale_init_value 1e-5 \
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
    --reprob 0.25 \
    --color_jitter 0.4 \
    --crop_pct 0.875 \
    --data_set IMNET1k \
    --data_path /mnt/cache/share/images/ \
    --nb_classes 1000 \
    --use_amp false \
    --save_ckpt true \
    --output_dir "backbone_outputdir/${PROJECT_NAME}" \
    --async \
    -o logs/"${PROJECT_NAME}.out" \
    -e logs/"${PROJECT_NAME}.err"

# sh train.sh
